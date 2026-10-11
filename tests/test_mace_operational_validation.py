"""Operational inputs, corrected real recurrence and phase-aware paired UQ."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from scipy.io import loadmat, savemat
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
from _mace_floor_study import evaluate, fingerprint, write_json
from _mace_operational_forecasts import (FEEDBACK, corrected_rollout, ensemble, forecast_stage, load_forecast,
                                         operational_input, read_simulation)
from _mace_operational_metrics import (comparison_metrics, draw_weights, evaluate_study,
                                       quantiles, sufficient, summarize)
from _mace_operational_profiles import (FAMILIES, PHASES, TIME, control, draw_commands,
                                        generate, load_manifest, mat_table, phase_masks)
from _mace_scaler_use import target_to_input
from run_mace_operational_validation import (PRE_FIX_SOURCE_HASHES, configuration, generated_inputs,
                                             run, simulate_stage, source_hashes)
from rabl.machine_learning.lstm_pipeline import LSTMRegressor, rolling_forecast


def scaling():
    return dict(x=dict(min=np.array([301., .7, 10.]), span=np.array([19., .3, 40.])),
                y=dict(min=np.array([300., .8]), span=np.array([20., .2])))


def synthetic(n, rng):
    shape = (n, 501, 2)
    mean = rng.uniform(.3, .7, shape).astype(np.float32)
    truth = (mean+rng.normal(0, .025, shape)).astype(np.float32)
    spread = rng.uniform(.001, .02, shape).astype(np.float32)
    stats = scaling()
    return dict(mean=mean, truth=truth, spread=spread,
                truth_physical=truth.astype(float)*stats["y"]["span"]+stats["y"]["min"])


class OperationalTests(unittest.TestCase):
    def test_independent_generation_disjoint_complete_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            m = generate(a, calibration_count=1000, test_count=2000, seed=20261010, settling_seconds=20)
            other = generate(b, calibration_count=1000, test_count=2000, seed=20261010, settling_seconds=20)
            self.assertEqual([p["moves"] for p in m["profiles"]], [p["moves"] for p in other["profiles"]])
            self.assertEqual(len(set(json.dumps(p["moves"], sort_keys=True) for p in m["profiles"])), 3000)
            for split, count in (("cal", 1000), ("test", 2000)):
                ps = [p for p in m["profiles"] if p["split"] == split]
                self.assertEqual(len(ps), count)
                self.assertEqual(set(p["family"] for p in ps), set(FAMILIES))
            self.assertEqual(load_manifest(a)["identity"], m["identity"])
            self.assertEqual(TIME[[0, -1]].tolist(), [0., 200.])
            for p in m["profiles"][::31]:
                table = loadmat(p["mat_path"])["profile"]
                np.testing.assert_array_equal(table, mat_table(p["moves"]))
                self.assertEqual(table.shape[1], 4)
                # Position interpolates exactly from the MAT table's knots.
                a1, v = control(p["moves"])
                self.assertLess(np.max(abs(np.interp(TIME, table[:, 0], table[:, 1])-a1)), 1e-12)
                self.assertLessEqual(max(abs(v)), .3)
                self.assertTrue(np.all(np.diff(table[:, 0]) >= 0))
            with Path(m["profiles"][0]["mat_path"]).open("ab") as f:
                f.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "MAT changed"):
                load_manifest(a)

    def test_phase_partition_boundaries_and_short_hold_override(self):
        moves = [dict(start_s=10., end_s=20.), dict(start_s=25., end_s=30.)]
        t = np.array([0., 10., 19.6, 20., 24.6, 25., 30., 49.6, 50., 200.])
        masks = phase_masks(moves, settling_seconds=20, time=t)
        np.testing.assert_array_equal(sum(masks[k].astype(int) for k in PHASES[1:]), np.ones(len(t)))
        self.assertTrue(masks["motion"][1])
        self.assertTrue(masks["early_settling"][3])
        self.assertTrue(masks["motion"][5])
        self.assertTrue(masks["late_hold"][8])
        self.assertTrue(masks["initial_hold"][0])

    def test_real_corrected_recurrence_uses_each_members_mapped_output(self):
        torch.set_num_threads(1)
        torch.manual_seed(83)
        model = LSTMRegressor(3, 3, 2, lstm_hidden=4, fc_hidden=(3,)).eval()
        x = np.random.default_rng(2).random((5, 3, 3), dtype=np.float32)
        stats = scaling()
        observed = []
        hook = model.register_forward_pre_hook(lambda _m, args: observed.append(args[0].detach().cpu().numpy().copy()))
        try:
            pred = corrected_rollout(model, x, stats)
        finally:
            hook.remove()
        np.testing.assert_array_equal(observed[0][0], x[0])
        for t in range(1, len(x)):
            np.testing.assert_array_equal(observed[t][0, -1, :2], target_to_input(pred[t-1], stats))
            np.testing.assert_array_equal(observed[t][0, :, 2], x[t, :, 2])
        changed = x.copy()
        changed[1:, :, :2] = 100
        np.testing.assert_array_equal(corrected_rollout(model, changed, stats), pred)
        self.assertGreater(np.max(abs(pred-rolling_forecast(model, x, state_dim=2))), 0)
        stats["x"]["min"][:2] = stats["y"]["min"]
        stats["x"]["span"][:2] = stats["y"]["span"]
        np.testing.assert_array_equal(corrected_rollout(model, x, stats), rolling_forecast(model, x, state_dim=2))

    def test_control_windows_match_original_shifted_builder(self):
        seed = np.tile([310., .95, 45.], (3, 1))
        angle = np.array([45., 46., 47., 48.])
        x = operational_input(angle, seed, scaling())
        np.testing.assert_array_equal(x[0, :, :2], np.repeat(x[0, :1, :2], 3, axis=0))
        expected = (np.array([[45., 45., 45.], [45., 45., 46.], [45., 46., 47.], [46., 47., 48.]])-10)/40
        np.testing.assert_array_equal(x[:, :, 2], expected.astype(np.float32))

    def test_sufficient_stats_pool_points_not_profile_means_and_exclude_empty_phases(self):
        masks = np.array([[True, False, False], [True, True, True], [False, False, False]])
        result = dict(mask=np.array([[[False], [True], [True]], [[True], [True], [True]], [[True], [True], [True]]]),
                      width_scaled=np.array([[[10.], [0.], [0.]], [[2.], [2.], [2.]], [[100.], [100.], [100.]]]),
                      score_scaled=np.ones((3, 3, 1)))
        st = sufficient(result, masks)
        out = summarize(st)
        self.assertEqual(out["mean_full_width"].item(), 4.)
        self.assertEqual(out["pointwise_coverage"].item(), .75)
        self.assertEqual(out["target_trajectory_coverage"].item(), .5)
        self.assertEqual(out["joint_trajectory_coverage"].item(), .5)

    def test_bootstrap_draws_are_paired_batch_invariant_and_match_direct_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, digest = draw_weights(Path(tmp)/"a.h5", n=8, replicates=31, seed=42, batch_size=3)
            wb, db = draw_weights(Path(tmp)/"b.h5", n=8, replicates=31, seed=42, batch_size=11)
            np.testing.assert_array_equal(w, wb)
            self.assertEqual(digest, db)
            rng = np.random.default_rng(7)
            result = evaluate(synthetic(8, rng), np.array([.1, .2], np.float32), floor=None, alpha=.05, span=scaling()["y"]["span"])
            mask = rng.random((8, 501)) > .5
            st = sufficient(result, mask)
            samples = summarize(st, w)
            with h5py.File(Path(tmp)/"a.h5") as f:
                indices = f["test_profile_indices"][()]
            for i, ids in enumerate(indices):
                direct = summarize({k: v[ids] for k, v in st.items()})
                for field in samples:
                    np.testing.assert_allclose(samples[field][i], direct[field], rtol=1e-14, atol=1e-14)
            paired = comparison_metrics({k: v[..., 0] for k, v in samples.items()}, {k: v[..., 0] for k, v in samples.items()})
            np.testing.assert_array_equal(paired["mean_interval_score_difference"], np.zeros(31))
            np.testing.assert_array_equal(paired["width_ratio"], np.ones(31))

    def test_quantiles_recalibrated_for_both_sources_and_joint_max_is_before_quantile(self):
        rng = np.random.default_rng(32)
        a, b = synthetic(20, rng), synthetic(20, rng)
        b["truth"] += np.float32(.5)
        qa = quantiles(a, alpha=.05, floor=.001)
        qb = quantiles(b, alpha=.05, floor=.001)
        self.assertGreater(float(qb["mace"]["target_trajectory"].mean()), float(qa["mace"]["target_trajectory"].mean()))
        for kind, q in qa.items():
            score = abs(a["truth"]-a["mean"])
            if kind == "mace":
                score /= a["spread"]+np.float32(.001)
            np.testing.assert_array_equal(q["target_maxima"], score.max(axis=1))
            np.testing.assert_array_equal(q["joint_maxima"], score.max(axis=(1, 2)))
            self.assertEqual(q["joint_trajectory"], np.sort(score.max(axis=(1, 2)))[19])
        # Disjoint rare target failures: max(q_j) would miss the joint objective.
        truth = np.ones((100, 501, 2), np.float32)
        truth[:4, 0, 0] = 50
        truth[4:8, 0, 1] = 50
        scores = quantiles(dict(truth=truth, mean=np.zeros_like(truth), spread=np.zeros_like(truth)), alpha=.05, floor=.001)
        self.assertEqual(scores["absolute"]["target_trajectory"].max(), 1)
        self.assertEqual(scores["absolute"]["joint_trajectory"], 50)

    def test_stored_forecast_checks_reject_single_ulp_and_legacy_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"saved.h5"
            rng, stats = np.random.default_rng(99), scaling()
            members = rng.random((10, 501, 2), dtype=np.float32)
            physical = rng.uniform([300., .8], [320., 1.], (501, 2))
            with h5py.File(path, "w") as f:
                f.attrs.update(completed=True, feedback=FEEDBACK, manifest_identity="id", member_count=10,
                               ensemble_std_ddof=0, target_names=np.asarray(["TN2", "x_steam_out"], dtype="S"))
                for k, v in dict(member_predictions_scaled=members, mean=members.mean(axis=0),
                                 spread=members.std(axis=0), truth_physical=physical,
                                 truth=((physical-stats["y"]["min"])/stats["y"]["span"]).astype(np.float32),
                                 time_seconds=TIME).items():
                    f.create_dataset("p/"+k, data=v)
            args = dict(names=["TN2", "x_steam_out"], profile_names=["p"], manifest_identity="id", stats=stats)
            arrays, checks = load_forecast(path, **args)
            self.assertEqual(arrays["truth"].dtype, np.float32)
            truth_check = next(r for r in checks if r["field"] == "truth")
            self.assertEqual(truth_check["conversion"], "both arrays -> float32")
            self.assertGreater(truth_check["max_absolute_difference"], 0)
            with h5py.File(path, "r+") as f:
                v = f["p/mean"][0, 0]
                f["p/mean"][0, 0] = np.nextafter(v, np.float32(np.inf))
            with self.assertRaisesRegex(ValueError, "array mismatch"):
                load_forecast(path, **args)
            with h5py.File(path, "r+") as f:
                f.attrs["feedback"] = "historical-direct-feedback"
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_forecast(path, **args)

    def test_real_ten_member_ensemble_retains_member_histories_and_ddof_zero(self):
        torch.set_num_threads(1)
        models = []
        for i in range(10):
            torch.manual_seed(30+i)
            models.append(LSTMRegressor(3, 3, 2, lstm_hidden=4, fc_hidden=(3,)).eval())
        x = np.random.default_rng(91).random((4, 3, 3), dtype=np.float32)
        members, mean, spread = ensemble(models, x, scaling())
        for i, model in enumerate(models):
            np.testing.assert_array_equal(members[i], corrected_rollout(model, x, scaling()))
        np.testing.assert_array_equal(mean, members.mean(axis=0))
        np.testing.assert_array_equal(spread, members.std(axis=0, ddof=0))
        self.assertEqual(members.dtype, np.float32)

    def test_forecast_stage_streams_original_and_operational_sets_with_strict_checkpoints(self):
        """Production-shaped IO; synthetic physics and fast inference, real loaders.

        Recurrence is tested separately with real forward calls above. Hard links
        keep this 1,000-profile fixture small; it is not a scientific dataset.
        """
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root, stats = Path(tmp), scaling()
            out = root/"run"
            manifest = generate(out, calibration_count=20, test_count=8, seed=81, settling_seconds=20)
            names = ["TN2", "x_steam_out"]
            ids = [f"cal_{i}" for i in range(1000)]
            raw_path, scaled_path = root/"raw.h5", root/"scaled.h5"
            x = np.tile([310., .95, 45.], (501, 3, 1))
            y = np.tile([310., .95], (501, 1))
            with h5py.File(raw_path, "w") as r, h5py.File(scaled_path, "w") as s:
                s.attrs.update(state_feature_names=np.asarray(names, dtype="S"), control_feature_name="drumAngleDeg", scaling_type="minmax")
                for axis in ("x", "y"):
                    for k, v in stats[axis].items():
                        s.create_dataset(f"scaling/{axis}_{k}", data=v)
                raw_group = r.create_group("files/"+ids[0])
                scaled_group = s.create_group("cal/files/"+ids[0])
                for field, axis, value in (("X", "x", x), ("Y", "y", y)):
                    raw_group.create_dataset(field, data=value)
                    scaled_group.create_dataset(field, data=(value-stats[axis]["min"])/stats[axis]["span"])
                for name in ids[1:]:
                    r["files/"+name] = raw_group
                    s["cal/files/"+name] = scaled_group
            paths = dict(raw_h5_path=raw_path, external_scaled_h5_path=scaled_path)
            arch = dict(n_lstm=1, lstm_hidden=4, lstm_dropout=0., n_fc=1, fc_hidden=[3])
            for i in range(10):
                paths[f"checkpoint_{i}"] = root/f"model_{i}.pt"
                paths[f"member_metadata_{i}"] = root/f"metadata_{i}.json"
                torch.manual_seed(i)
                torch.save(LSTMRegressor(3, 3, 2, **arch).state_dict(), paths[f"checkpoint_{i}"])
                write_json(paths[f"member_metadata_{i}"], dict(member_index=i, architecture=arch))
            hashes = {k: fingerprint(p) for k, p in paths.items()}
            results = {}
            for p in manifest["profiles"]:
                path = out/"simulation"/p["split"]/(p["id"]+".csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "drumAngleDeg"]+names)
                    w.writerows(np.column_stack([TIME, control(p["moves"])[0], y]))
                results[p["id"]] = fingerprint(path)
            write_json(out/"simulation_metadata.json", dict(completed=True, manifest_identity=manifest["identity"], result_hashes=results))
            prior = dict(steps=501, target_names=names, profile_names=dict(cal=ids))
            calls = []
            def fast(models, frame, native):
                self.assertEqual(len(models), 10)
                calls.append(frame.shape)
                members = np.tile(np.array([.5, .75], np.float32), (10, 501, 1))
                members += np.arange(10, dtype=np.float32)[:, None, None]*np.float32(.001)
                return members, members.mean(axis=0), members.std(axis=0)
            config = dict(output_dir=str(out), device="cpu", torch_threads=1)
            with mock.patch("_mace_operational_forecasts.frozen_inputs", return_value=(prior, paths, hashes)), mock.patch("_mace_operational_forecasts.ensemble", side_effect=fast):
                meta = forecast_stage(config, manifest, root=ROOT)
            self.assertTrue(meta["completed"])
            self.assertEqual(len(calls), 1028)
            self.assertTrue(all(shape == (501, 3, 3) for shape in calls))
            self.assertEqual(meta["feedback"], FEEDBACK)
            same = {k: fingerprint(p) for k, p in paths.items()}
            self.assertEqual(same, hashes)
            for split, expected in (("original_cal", ids), ("operational_cal", [p["id"] for p in manifest["profiles"] if p["split"] == "cal"]),
                                    ("operational_test", [p["id"] for p in manifest["profiles"] if p["split"] == "test"])):
                arrays, checks = load_forecast(out/"forecasts"/(split+".h5"), names=names, profile_names=expected,
                                               manifest_identity=manifest["identity"], stats=stats)
                self.assertEqual(len(arrays["truth"]), len(expected))
                self.assertTrue(all(r["match"] for r in checks))
            with self.assertRaises(FileExistsError):
                forecast_stage(config, manifest, root=ROOT)

    def test_full_time_phase_metrics_and_units_end_to_end_without_physics(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root, rng = Path(tmp), np.random.default_rng(57)
            profiles = [dict(id=f"p{i}", family=family, moves=draw_commands(family, rng))
                        for i, family in enumerate(FAMILIES*2)]
            cal = synthetic(20, rng)
            test = synthetic(8, rng)
            meta = evaluate_study(root, calibrations=dict(original=cal, operational=cal), test=test,
                                  profiles=profiles, names=["TN2", "x_steam_out"], stats=scaling(),
                                  alpha=.05, floor=.001, settling_seconds=20, replicates=31, seed=3, batch_size=7)
            self.assertTrue(meta["completed"])
            self.assertTrue(meta["saved_endpoint_coverage_exact"])
            with (root/"uq_method_metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(set(r["mode"] for r in rows), {"horizon", "target_trajectory", "joint_trajectory"})
            self.assertEqual(set(r["phase"] for r in rows), set(PHASES))
            self.assertEqual(set(r["family"] for r in rows), {"pooled", *FAMILIES})
            self.assertEqual({r["units"] for r in rows if r["scope"] == "TN2" and r["metric"] == "mean_full_width"}, {"K"})
            with (root/"paired_comparisons.csv").open() as f:
                comparisons = list(csv.DictReader(f))
            identical = [r for r in comparisons if r["comparison"] == "mace:operational_minus_original" and r["metric"] == "mean_interval_score_difference"]
            self.assertTrue(identical)
            self.assertTrue(all(float(r["estimate"]) == float(r["ci_lower"]) == float(r["ci_upper"]) == 0 for r in identical))
            with self.assertRaises(FileExistsError):
                evaluate_study(root, calibrations={}, test={}, profiles=[], names=[], stats={}, alpha=.05,
                               floor=.001, settling_seconds=20, replicates=31, seed=3, batch_size=7)

    def test_simulation_grid_control_and_failed_or_missing_output_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"result.csv"
            moves = draw_commands("B", np.random.default_rng(10))
            angle = control(moves)[0]
            def save(t, a):
                with path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "drumAngleDeg", "TN2"])
                    w.writerows(zip(t, a, np.full(len(t), 300.)))
            save(TIME, angle)
            y, actual, report = read_simulation(path, names=["TN2"], moves=moves)
            self.assertEqual(y.shape, (501, 1))
            np.testing.assert_array_equal(actual, angle)
            self.assertEqual(report["angle_max_difference_degrees"], 0)
            save(TIME, angle+.01)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_simulation(path, names=["TN2"], moves=moves)
            save(TIME[:-1], angle[:-1])
            with self.assertRaisesRegex(ValueError, "501"):
                read_simulation(path, names=["TN2"], moves=moves)

    def test_reported_discrepancy_reproduced_with_verified_float32_time_angle_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(np.random.SeedSequence(20261010).spawn(2)[0])
            family = str(rng.choice(FAMILIES))
            moves = draw_commands(family, rng)
            expected = control(moves)[0]
            extracted = np.interp(TIME, TIME.astype(np.float32).astype(float), expected.astype(np.float32).astype(float))
            self.assertEqual(float(abs(extracted-expected).max()), 2.107193843414734e-6)
            path, raw = Path(tmp)/"results_drum_profile_10001.csv", Path(tmp)/"run_10001.mat"
            def save(angle):
                with path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "drumAngleDeg", "TN2"])
                    w.writerows(zip(TIME, angle, np.full(501, 300.)))
            save(extracted)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_simulation(path, names=["TN2"], moves=moves)
            data = np.stack([TIME, expected]).astype(np.float32)
            savemat(raw, {"data_2": data}, format="4")
            _y, angle, report = read_simulation(path, names=["TN2"], moves=moves)
            np.testing.assert_array_equal(angle, extracted)  # No CSV signal rewriting.
            self.assertEqual(report["raw_dymola_result"]["trajectory_dtype"], "float32")
            self.assertEqual(report["observed_vs_float32_reference_max_difference_degrees"], 0.)
            self.assertLessEqual(report["angle_max_fraction_of_pointwise_bound"], 1)
            save(extracted+.0001)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_simulation(path, names=["TN2"], moves=moves)
            save(extracted)
            savemat(raw, {"data_2": data.astype(np.float64)}, format="4")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_simulation(path, names=["TN2"], moves=moves)

    def test_validation_fix_migration_is_narrow_and_keeps_physics_and_profile_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual = source_hashes()
            def fixture(label, separator="/"):
                out = Path(tmp)/label
                c = configuration(dict(output_dir=str(out), floor_study_dir=str(Path(tmp)/"study"),
                                       operational_calibration_count=20, operational_test_count=8, plots=False))
                m = generate(out, calibration_count=20, test_count=8, seed=c["profile_seed"], settling_seconds=20)
                current = {k.replace("/", separator): v for k, v in actual.items()}
                before = current.copy()
                for k, hashes in PRE_FIX_SOURCE_HASHES.items():
                    before[k.replace("/", separator)] = sorted(hashes)[0]
                # Exercise both of the whitelisted Windows CRLF digests explicitly.
                if separator == "\\":
                    before["scripts\\run_mace_operational_validation.py"] = "96faa98189335fa6ee095c6f3153559c5e778e92ae431c405108ff35850c11a8"
                    before["scripts\\_mace_operational_forecasts.py"] = "8cf234cfabfeda54a99ca0f20a391d60df07ef01174967e2a343f8760ac80aff"
                write_json(out/"run_configuration.json", dict(configuration=c, source_hashes=before, git=dict(commit="be406101")))
                identity = dict(manifest_identity=m["identity"], modelica_source_hashes={"package.mo": "physics"},
                                runner_source_sha256="runner", output_interval=.4, model_name="MicroreactorPK.Experiments.RunOneProfile")
                write_json(out/"simulation_metadata.json", dict(**identity, completed=False, config=c, result_hashes={}))
                return out, c, m, current, identity
            for i, separator in enumerate(("/", "\\")):
                out, c, m, current, identity = fixture(f"good{i}", separator)
                hashes = {p["mat_path"]: fingerprint(Path(p["mat_path"])) for p in m["profiles"]}
                with mock.patch("run_mace_operational_validation.source_hashes", return_value=current), mock.patch("run_mace_operational_validation.simulation_identity", return_value=identity):
                    with self.assertRaisesRegex(ValueError, "source changed"):
                        generated_inputs(c)
                    migrated = generated_inputs(c, resume_after_validation_fix=True)
                    self.assertEqual(migrated, m)
                    self.assertTrue((out/"run_configuration_before_validation_fix.json").exists())
                    self.assertTrue((out/"simulation_metadata_before_validation_fix.json").exists())
                    after = json.loads((out/"run_configuration.json").read_text())
                    self.assertEqual(after["source_hashes"], current)
                    self.assertEqual(after["configuration"], c)
                    self.assertFalse(after["validation_fix_migration"]["regenerated_profiles"])
                    generated_inputs(c, resume_after_validation_fix=True)  # Idempotent after migration.
                self.assertEqual({p: fingerprint(Path(p)) for p in hashes}, hashes)
            for kind in ("unrelated_source", "physics", "forecast_started", "configuration"):
                out, c, m, current, identity = fixture(kind)
                if kind == "unrelated_source":
                    payload = json.loads((out/"run_configuration.json").read_text())
                    payload["source_hashes"]["scripts/_mace_operational_metrics.py"] = "edited"
                    write_json(out/"run_configuration.json", payload)
                elif kind == "physics":
                    identity = {**identity, "modelica_source_hashes": {"package.mo": "changed"}}
                elif kind == "forecast_started":
                    (out/"forecasts").mkdir()
                else:
                    c = {**c, "profile_seed": c["profile_seed"]+1}
                with mock.patch("run_mace_operational_validation.source_hashes", return_value=current), mock.patch("run_mace_operational_validation.simulation_identity", return_value=identity), self.assertRaises(ValueError):
                    generated_inputs(c, resume_after_validation_fix=True)
                self.assertFalse((out/"run_configuration_before_validation_fix.json").exists())
            with self.assertRaisesRegex(ValueError, "only"):
                run({}, "all", resume_after_validation_fix=True)

    def test_simulation_stage_resume_keeps_manifest_and_checks_every_result(self):
        """Exercise runner contract/resumption with synthetic CSVs, not Dymola."""
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp)/"run"
            manifest = generate(out, calibration_count=20, test_count=8, seed=39, settling_seconds=20)
            package = root/"MicroreactorPK/package.mo"
            package.parent.mkdir()
            package.write_text("package MicroreactorPK end MicroreactorPK;\n")
            runner_source = root/"src/rabl/interface/pymola.py"
            runner_source.parent.mkdir(parents=True)
            runner_source.write_text("# runner contract fixture\n")
            names, closed, first = ["TN2", "x_steam_out"], [], [True]
            class Runner:
                def __init__(self, cfg):
                    self.cfg = cfg
                def start(self):
                    pass
                def close(self):
                    closed.append(self.cfg.out_dir)
                def run_all(self):
                    self_test.assertEqual(self.cfg.output_interval, .4)
                    self_test.assertEqual(self.cfg.table_name, "profile")
                    for i, p in enumerate(manifest["profiles"]):
                        if Path(p["mat_path"]).parent != Path(self.cfg.profiles_dir):
                            continue
                        path = Path(self.cfg.out_dir)/(p["id"]+".csv")
                        if self.cfg.skip_existing and path.exists():
                            continue
                        with path.open("w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["t", "drumAngleDeg"]+names)
                            w.writerows(np.column_stack([TIME, control(p["moves"])[0], np.tile([310., .95], (501, 1))]))
                        if first[0] and i == 3:
                            first[0] = False
                            raise RuntimeError("synthetic connection interruption")
            self_test = self
            fake = types.ModuleType("rabl.interface.pymola")
            fake.BatchConfig, fake.DymolaBatchRunner = lambda **kwargs: types.SimpleNamespace(**kwargs), Runner
            c = dict(output_dir=str(out), package_mo=str(package), floor_study_dir="unused")
            with mock.patch.dict(sys.modules, {"rabl.interface.pymola": fake}), mock.patch("run_mace_operational_validation.ROOT", root), mock.patch("run_mace_operational_validation.validate_prior_study", return_value=dict(target_names=names)):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    simulate_stage(c, manifest)
                self.assertFalse(json.loads((out/"simulation_metadata.json").read_text())["completed"])
                self.assertEqual(len(closed), 1)
                result = simulate_stage(c, manifest)
                self.assertTrue(result["completed"])
                self.assertEqual(len(result["result_hashes"]), 28)
                self.assertEqual(len(closed), 3)
                # If only completion bookkeeping was interrupted, validate all
                # cached splits without opening another Dymola instance.
                write_json(out/"simulation_metadata.json", {**result, "completed": False})
                cached = simulate_stage(c, manifest)
                self.assertTrue(cached["completed"])
                self.assertEqual(len(closed), 3)
                with self.assertRaises(FileExistsError):
                    simulate_stage(c, manifest)

    def test_configuration_and_generation_source_guards(self):
        for edit in ({"common_floor": .01}, {"alpha": .1}, {"operational_calibration_count": 10}, {"bootstrap_replicates": True}):
            with self.assertRaises(ValueError):
                configuration(dict(floor_study_dir="unused", output_dir="unused2", **edit))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            payload = dict(floor_study_dir=str(Path(tmp)/"study"), output_dir=str(Path(tmp)/"run"),
                           operational_calibration_count=20, operational_test_count=8, plots=False)
            out = run(payload, "generate")
            self.assertEqual(len(generated_inputs(configuration(payload))["profiles"]), 28)
            changed = configuration({**payload, "early_settling_seconds": 30})
            with self.assertRaisesRegex(ValueError, "Configuration"):
                generated_inputs(changed)
            with self.assertRaises(FileExistsError):
                run(payload, "generate")


if __name__ == "__main__":
    unittest.main()
