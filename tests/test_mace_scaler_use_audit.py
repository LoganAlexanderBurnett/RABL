"""Real frozen-checkpoint replay tests, including unequal X/Y coordinate frames."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
from _mace_floor_study import calibrate, fingerprint, write_json
from _mace_scaler_audit import verify_scaler
from _mace_scaler_use import Checks, independent_rollout, observed_ensemble, selected_profiles, target_to_input
from run_mace_scaler_use_audit import METHODS, production_modules, run


def fixture(root, *, unequal=False):
    """Small real LSTMs and HDF5 inputs; production generates historical forecasts."""
    torch.set_num_threads(1)
    pipe, bag, cp, _ds, _uq = production_modules()
    rng = np.random.default_rng(19)
    names = ["TN2", "x_steam_out"]
    profiles = {s: [f"{s}_{i}" for i in range(count)] for s, count in
                [("train", 3), ("val", 1), ("cal", 20), ("test", 4)]}
    physical = {}
    lo, hi = np.array([300., .8]), np.array([325., 1.])
    for split, ids in profiles.items():
        for name in ids:
            x = np.empty((4, 3, 3), dtype=np.float64)
            x[:, :, :2] = lo+rng.random((4, 3, 2))*(hi-lo)
            x[:, :, 2] = 10+rng.random((4, 3))*40
            y = lo+rng.random((4, 2))*(hi-lo)
            if split == "train":
                x[0, :, :2] = y[0] = lo
                x[1, :, :2] = y[1] = hi
                x[0, :, 2], x[1, :, 2] = 10, 50
            if unequal:
                x[:, :, 0] = 301+(x[:, :, 0]-300)*.95
            physical[name] = (x, y)
    stats = {}
    for axis, position in (("x", 0), ("y", 1)):
        values = np.concatenate([physical[name][position].reshape(-1, 3 if axis == "x" else 2) for name in profiles["train"]])
        minimum, maximum = values.min(axis=0), values.max(axis=0)
        stats[axis] = {"min": minimum, "max": maximum, "span": maximum-minimum}
    exp = root/"experiment"
    exp.mkdir()
    raw_path, external_path, training_path = root/"raw.h5", root/"external.h5", root/"training.h5"
    attrs = {"state_feature_names": np.asarray(names, dtype="S"), "control_feature_name": "drumAngleDeg"}
    with h5py.File(raw_path, "w") as raw, h5py.File(external_path, "w") as scaled:
        for k, v in attrs.items():
            raw.attrs[k] = v
            scaled.attrs[k] = v
        scaled.attrs["scaling_type"] = "minmax"
        for axis, st in stats.items():
            for k, a in st.items():
                scaled.create_dataset("scaling/"+axis+"_"+k, data=a)
        for split, ids in profiles.items():
            for name in ids:
                for axis, field, value in (("x", "X", physical[name][0]), ("y", "Y", physical[name][1])):
                    raw.create_dataset(f"files/{name}/{field}", data=value)
                    scaled.create_dataset(f"{split}/files/{name}/{field}", data=(value-stats[axis]["min"])/stats[axis]["span"])
    shutil.copyfile(external_path, training_path)
    scaler_path = root/"scaler.json"
    write_json(scaler_path, {"type": "minmax", **{a: {k: v.astype(np.float32) for k, v in st.items() if k in ("min", "span")} for a, st in stats.items()}})
    architecture = {"n_lstm": 1, "lstm_hidden": 4, "lstm_dropout": 0., "n_fc": 1, "fc_hidden": [3]}
    models = []
    for i in range(10):
        torch.manual_seed(101+i)
        model = pipe.LSTMRegressor(3, 3, 2, **architecture).eval()
        models.append(model)
        folder = exp/f"ensemble/model_{i}"
        folder.mkdir(parents=True)
        torch.save(model.state_dict(), folder/"model.pt")
        write_json(folder/"member_metadata.json", {"member_index": i, "architecture": architecture, "device": "cpu"})
    run_stats = pipe._load_scaling_stats(external_path)
    forecasts, arrays = {}, {}
    for split in ("cal", "test"):
        file = root/(split+".h5")
        forecasts[split] = []
        with h5py.File(file, "w") as f:
            f.attrs["target_names"] = np.asarray(names, dtype="S")
            f.attrs["member_count"] = 10
            f.attrs["ensemble_std_ddof"] = 0
            for name, x, y in pipe.ProfileDataset(external_path, profiles[split], split):
                stack, mean, spread = bag.ensemble_member_predictions_scaled(models, x.numpy(), state_dim=2)
                y = y.numpy()
                true = pipe._descale_targets_from_stats(run_stats, y)
                pred = pipe._descale_targets_from_stats(run_stats, mean)
                u = pipe._descale_feature_from_stats(run_stats, x.numpy()[:, -1, 2], 2)
                values = {"member_predictions_scaled": stack, "ensemble_mean_scaled": mean, "ensemble_std_scaled": spread,
                          "y_true_scaled": y, "y_true_physical": true, "ensemble_mean_physical": pred,
                          "t": np.arange(4, dtype=np.float32), "control": u}
                physical_members = pipe._descale_targets_from_stats(run_stats, stack)
                half = 2.*physical_members.std(axis=0, ddof=0)
                values['ensemble_lower_2sigma_physical'] = pred-half
                values['ensemble_upper_2sigma_physical'] = pred+half
                for k, a in values.items():
                    f.create_dataset(name+"/"+k, data=a)
                forecasts[split].append({"profile": name, "t": values["t"], "u": u, "y_true_scaled": y,
                    "member_predictions_scaled": stack, "mean_scaled": mean, "spread_scaled": spread, "y_true": true, "y_pred": pred})
        arrays[split] = {k: np.array([e[v] for e in forecasts[split]]) for k, v in
                         [("truth", "y_true_scaled"), ("mean", "mean_scaled"), ("spread", "spread_scaled")]}
    prior_config = {"alpha": .05, "expected_profile_counts": {s: len(v) for s, v in profiles.items()},
        "ensemble_experiment_dir": str(exp), "external_scaled_h5_path": str(external_path),
        "training_scaled_h5_path": str(training_path), "raw_h5_path": str(raw_path),
        "reference_scaler_json_path": str(scaler_path), "calibration_forecasts_h5_path": str(root/"cal.h5"),
        "test_forecasts_h5_path": str(root/"test.h5"), "saved_uq_comparison_dir": str(root/"uq")}
    input_hashes = {k: fingerprint(Path(v)) for k, v in prior_config.items() if k.endswith("_path")}
    input_hashes.update({f"checkpoint_{i}": fingerprint(exp/f"ensemble/model_{i}/model.pt") for i in range(10)})
    study = root/"study"
    study.mkdir()
    write_json(study/"study_metadata.json", {"completed": True, "no_training_or_inference": True,
        "input_file_sizes_and_mtimes_unchanged": True, "config": prior_config, "profile_names": profiles,
        "member_count": 10, "steps": 4, "target_names": names, "input_hashes": input_hashes,
        "coverage_convention": "Inclusive float32 scaled endpoints"})
    report, _ = verify_scaler(external_path=external_path, training_path=training_path,
        raw_path=raw_path, reference_json_path=scaler_path, manifests=profiles, target_names=names)
    assert report["match"]
    write_json(study/"scaler_verification.json", report)
    qs = {"absolute": calibrate(arrays["cal"], alpha=.05, floor=None),
          "mace": calibrate(arrays["cal"], alpha=.05, floor=.001)}
    with h5py.File(study/"recalibration.h5", "w") as f:
        f.attrs["target_names"] = np.asarray(names, dtype="S")
        f.attrs["calibration_profile_names"] = np.asarray(profiles["cal"], dtype="S")
        f.attrs["quantile_rank"] = 20
        for kind, label in (("absolute", "absolute"), ("mace", "floor_0.001")):
            for k, a in qs[kind].items():
                f.create_dataset(label+"/"+k, data=a)
    for method, (kind, mode) in METHODS.items():
        folder = root/"uq"/method
        folder.mkdir(parents=True)
        qkey = "q_by_target" if mode == "trajectory" else "q_by_horizon_target"
        cal = {"alpha": .05, "calibration_profile_names": profiles["cal"], qkey: qs[kind][mode], "sigma_floor": .001}
        write_json(folder/"calibration_metadata.json", cal)
        for split, filename in (("cal", "calibration_forecasts.h5"), ("test", "test_forecasts.h5")):
            entries = [cp.apply_uq_method(e, method_id=method, calibration_result=cal, scaling_stats=run_stats, alpha=.05) for e in forecasts[split]]
            cp.save_uq_forecasts_hdf5(entries, output_path=folder/filename, metadata={}, target_names=names)
    return {"floor_study_dir": str(study), "output_dir": str(root/"audit"),
            "profiles_per_split": 1, "device": "cpu", "torch_threads": 1}, models, stats


class ScalerUseTests(unittest.TestCase):
    def test_equal_frame_mapping_is_exact_identity_and_unequal_frame_changes_feedback(self):
        p = np.array([[.2, .4]], dtype=np.float32)
        stats = {"x": {"min": np.array([10., 20., 30.]), "span": np.array([2., 4., 5.])},
                 "y": {"min": np.array([10., 20.]), "span": np.array([2., 4.])}}
        np.testing.assert_array_equal(target_to_input(p, stats), p)
        stats["x"]["min"][0] += 1
        np.testing.assert_array_equal(target_to_input(p, stats), np.array([[-.3, .4]], dtype=np.float32))

    def test_manifest_positions_not_metric_selection(self):
        self.assertEqual(selected_profiles(list('abcdefg'), 3), ['a', 'd', 'g'])
        for count in [0, 8, True, 1.5]:
            with self.assertRaises(ValueError):
                selected_profiles(list('abcdefg'), count)

    def test_float64_float32_comparison_reports_conversion_and_same_dtype_one_ulp_fails(self):
        checks = Checks()
        a = np.array([.3], np.float32)
        b = a.astype(float)+1e-10
        row = checks.add('precision', a, b)
        self.assertTrue(row['match'])
        self.assertEqual(row['conversion'], 'both arrays -> float32')
        row = checks.add('ulp', a, np.nextafter(a, np.float32(1)))
        self.assertFalse(row['match'])

    def test_real_checkpoint_end_to_end_exact_and_inputs_immutable(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            config, _, _ = fixture(Path(tmp))
            before = {p: fingerprint(p) for p in Path(tmp).rglob('*') if p.is_file()}
            out = run(config)
            metadata = json.loads((out/'audit_metadata.json').read_text())
            self.assertEqual(metadata['status'], 'passed')
            self.assertTrue(metadata['completed'])
            self.assertTrue(metadata['input_sizes_and_mtimes_unchanged'])
            self.assertTrue(metadata['sources_unchanged'])
            for p, state in before.items():
                self.assertEqual(fingerprint(p), state)
            effects = (out/'replay_coverage_effects.csv').read_text()
            self.assertIn('changed_coverage_events', effects)
            with h5py.File(out/'replays.h5') as f:
                self.assertEqual(f['test/test_0/member_predictions_scaled'].shape, (10, 4, 2))
                self.assertEqual(f['test/test_0/model_0/actual_model_inputs'].shape, (3, 3, 3))
            self.assertTrue((out/'scaler_verification.json').exists())
            with self.assertRaises(FileExistsError):
                run(config)

    def test_exact_historical_replay_does_not_hide_feedback_coordinate_bug(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            config, _, _ = fixture(Path(tmp), unequal=True)
            out = run(config)
            metadata = json.loads((out/'audit_metadata.json').read_text())
            self.assertTrue(metadata['completed'])
            self.assertFalse(metadata['passed'])
            self.assertEqual(metadata['status'], 'input_output_coordinate_mismatch')
            self.assertEqual(metadata['check_summary']['invariant']['mismatches'], 0)
            self.assertEqual(metadata['check_summary']['replay']['mismatches'], 0)
            self.assertGreater(metadata['check_summary']['coordinate']['mismatches'], 0)

    def test_changed_original_checkpoint_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config, _, _ = fixture(root)
            with (root/'experiment/ensemble/model_0/model.pt').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Scientific input differs'):
                run(config)
            metadata = json.loads((root/'audit/audit_metadata.json').read_text())
            self.assertFalse(metadata['completed'])

    def test_control_corruption_detected_even_if_prior_hash_record_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config, _, _ = fixture(root)
            with h5py.File(root/'external.h5', 'r+') as f:
                f['test/files/test_0/X'][0, -1, -1] += .1
            path = root/'study/study_metadata.json'
            prior = json.loads(path.read_text())
            prior['input_hashes']['external_scaled_h5_path'] = fingerprint(root/'external.h5')
            write_json(path, prior)
            with self.assertRaisesRegex(ValueError, 'raw-to-scaled checks failed'):
                run(config)
            self.assertFalse(json.loads((root/'audit/scaler_verification.json').read_text())['match'])

    def test_skip_extrema_reuses_hash_guarded_prior_audit(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            config, _, _ = fixture(Path(tmp))
            config['recompute_train_extrema'] = False
            out = run(config)
            metadata = json.loads((out/'audit_metadata.json').read_text())
            self.assertEqual(metadata['status'], 'passed')
            self.assertIn('Reused', metadata['train_extrema_scope'])

    def test_runtime_observation_detects_teacher_forcing_and_changed_control(self):
        torch.set_num_threads(1)
        pipe, _, _, _, _ = production_modules()
        torch.manual_seed(30)
        models = [pipe.LSTMRegressor(3, 3, 2, lstm_hidden=4, fc_hidden=(3,)).eval()]
        x = np.random.default_rng(2).random((4, 3, 3), dtype=np.float32)
        stats = {'x': {'min': np.zeros(3), 'span': np.ones(3)},
                 'y': {'min': np.zeros(2), 'span': np.ones(2)}}
        for corrupt_control in [False, True]:
            def wrong_production(models, x, state_dim):
                # Deliberately uses future true states instead of own predictions.
                z = x.copy()
                if corrupt_control:
                    z[:, :, 2] += .1
                with torch.no_grad():
                    member = np.array([models[0](torch.from_numpy(v[None])).numpy()[0] for v in z])
                stack = member[None]
                return stack, stack.mean(axis=0), stack.std(axis=0)
            checks = Checks()
            with contextlib.redirect_stdout(io.StringIO()):
                observed_ensemble(models, x, stats, checks, split='test', profile='sample',
                                  production=wrong_production, target_names=['TN2', 'x_steam_out'])
            self.assertGreater(checks.summary()['invariant']['mismatches'], 0)
            row = next(r for r in checks.rows if r['check'].startswith('actual_model_inputs_vs'))
            if corrupt_control:
                self.assertGreater(row['max_absolute_difference_by_feature']['drumAngleDeg'], 0)

    def test_saved_interval_one_ulp_mismatch_is_completed_failure_not_tolerance_pass(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config, _, _ = fixture(root)
            file = root/'uq/ensemble_conformal_target_horizon/test_forecasts.h5'
            with h5py.File(file, 'r+') as f:
                a = f['test_0/lower_scaled'][()]
                a[0, 0] = np.nextafter(a[0, 0], np.float32(np.inf))
                f['test_0/lower_scaled'][...] = a
            out = run(config)
            metadata = json.loads((out/'audit_metadata.json').read_text())
            self.assertTrue(metadata['completed'])
            self.assertEqual(metadata['status'], 'audit_failed')
            self.assertFalse(metadata['passed'])


if __name__ == '__main__':
    unittest.main()
