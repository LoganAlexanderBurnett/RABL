"""Independent arithmetic and HDF5 integration tests; no torch/model loading."""

from __future__ import annotations

import ast
import json
import shutil
import sys
import tempfile
import unittest
from math import ceil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _mace_floor_study import (FLOORS, bounds, calibrate, coverage, evaluate, fingerprint,
                             legacy_metrics, reconcile_saved, wilson, write_json)
from _mace_scaler_audit import compare_arrays, verify_scaler
from run_lstm_floor_sensitivity import run


class ArithmeticTests(unittest.TestCase):
    def test_correct_profile_max_and_order_statistic(self):
        truth = np.asarray([[[8,2],[0,1]], [[0,1],[8,4]], [[0,5],[0,0]]], dtype=np.float32)
        spread = np.ones_like(truth)
        cal = {"truth":truth, "mean":np.zeros_like(truth), "spread":spread}
        q = calibrate(cal, alpha=.5, floor=1)
        scores = truth/(spread+np.float32(1))
        np.testing.assert_array_equal(q["horizon"], np.sort(scores, axis=0)[1])
        np.testing.assert_array_equal(q["trajectory_scores"], scores.max(axis=1))
        np.testing.assert_array_equal(q["trajectory"], np.sort(scores.max(axis=1),axis=0)[1])
        self.assertEqual(int(q["rank"]), 2)
        self.assertFalse(np.array_equal(q["trajectory"], q["horizon"].max(axis=0)))
        self.assertFalse(np.array_equal(q["trajectory"], calibrate(cal, alpha=.5, floor=.001)["trajectory"]))

    def test_exact_parity_with_existing_calibrators_at_all_floors(self):
        # Extract the pure numerical functions from the existing module to avoid
        # importing its unrelated torch/Dymola stack in a saved-forecast study.
        path = ROOT / "src/rabl/machine_learning/conformal_prediction.py"
        tree = ast.parse(path.read_text())
        wanted = {"_sigma_floor_vector", "_quantile_and_index",
                  "calibrate_ensemble_normalized_conformal", "calibrate_absolute_conformal"}
        functions = [ast.get_source_segment(path.read_text(), n) for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name in wanted]
        namespace = {"np":np, "ceil":ceil, "Any":Any}
        exec("from __future__ import annotations\n" + "\n\n".join(functions), namespace)
        rng = np.random.default_rng(10)
        mean = rng.random((20,5,2), dtype=np.float32)
        truth = rng.random((20,5,2), dtype=np.float32)
        spread = rng.random((20,5,2), dtype=np.float32)*np.float32(.03)
        cal = {"mean":mean,"truth":truth,"spread":spread}
        entries = [{"profile":str(i), "mean_scaled":mean[i], "y_true_scaled":truth[i],
                    "spread_scaled":spread[i]} for i in range(20)]
        for floor in (*FLOORS, None):
            actual = calibrate(cal, alpha=.05, floor=floor)
            for mode, key in (("trajectory","q_by_target"),("per_horizon","q_by_horizon_target")):
                fn = "calibrate_absolute_conformal" if floor is None else "calibrate_ensemble_normalized_conformal"
                kwargs = {} if floor is None else {"sigma_floor":floor}
                ref = namespace[fn](entries,alpha=.05,temporal_mode=mode,**kwargs)
                np.testing.assert_array_equal(actual["trajectory" if mode=="trajectory" else "horizon"],ref[key])

    def test_large_floor_limit_and_no_fixed_quantile_shortcut(self):
        rng = np.random.default_rng(17)
        cal = {"truth":rng.random((50,3,2), dtype=np.float32),
               "mean":np.zeros((50,3,2),np.float32),
               "spread":rng.random((50,3,2), dtype=np.float32)*np.float32(1e-5)}
        test = {k:v[:2] for k,v in cal.items()}
        for mode in ("horizon","trajectory"):
            aq=calibrate(cal,alpha=.05,floor=None)[mode]
            q=calibrate(cal,alpha=.05,floor=100)[mode]
            al,au=bounds(test,aq,None)
            lo,hi=bounds(test,q,100)
            np.testing.assert_allclose(hi,au,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(lo,al,rtol=1e-6,atol=1e-7)
            fixed=calibrate(cal,alpha=.05,floor=1e-6)[mode]
            self.assertGreater(float(bounds(test,fixed,100)[1].max()),1000)

    def test_precision_checks_have_no_tolerance(self):
        a=np.asarray([1,2],dtype=np.float32)
        b=a.copy(); b[0]=np.nextafter(b[0],np.float32(2))
        self.assertFalse(compare_arrays(a,b)["match"])
        c=a.astype(np.float64); c[0]+=1e-9
        report=compare_arrays(a,c)
        self.assertEqual(report["status"],"precision_converted_match")
        self.assertGreater(report["max_absolute_difference"],0)
        self.assertEqual(report["unequal_count_after_conversion"],0)

    def test_inclusive_scaled_coverage_and_profile_event(self):
        truth=np.asarray([[[.1],[.2]],[[.3],[.4]]],dtype=np.float32)
        test={"truth":truth,"mean":np.zeros_like(truth),"spread":np.zeros_like(truth)}
        result=evaluate(test,np.asarray([.3],np.float32),floor=None,alpha=.05,span=np.asarray([2.]))
        self.assertEqual(result["coverage"]["pointwise"],.75)
        self.assertEqual(result["coverage"]["mean_targetwise_trajectory"],.5)
        self.assertAlmostEqual(result["mean_physical_width_by_target"][0],1.2,places=6)
        lo,hi=wilson(1900,2000)
        self.assertLess(lo,.95); self.assertGreater(hi,.95)

    def test_reconciliation_exposes_physical_rounding_coverage_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            q=np.asarray([.3],dtype=np.float32)
            y=np.asarray([[[np.nextafter(q[0],np.float32(1))]]],dtype=np.float32)
            stats={"y":{"span":np.asarray([.1]),"min":np.asarray([1e6])}}
            physical=y*np.float32(.1)+np.float32(1e6)
            test={"truth":y,"mean":np.zeros_like(y),"spread":np.zeros_like(y),"truth_physical":physical}
            # Force a scaled endpoint of .3 with a quantile consistent with tau.
            q=np.asarray([300],dtype=np.float32)
            result=evaluate(test,q,floor=.001,alpha=.05,span=stats["y"]["span"])
            pmask=((result["lower"]*np.float32(.1)+np.float32(1e6)<=physical)
                   &(physical<=result["upper"]*np.float32(.1)+np.float32(1e6)))
            method=root/"ensemble_conformal_target_trajectory";method.mkdir()
            write_json(method/"calibration_metadata.json",{"q_by_target":q,"alpha":.05})
            metrics=legacy_metrics(result["coverage"],coverage(pmask),["x_steam_out"],"trajectory")
            metrics["mean_scaled_interval_width_overall"]=result["mean_scaled_width"]
            miss=np.maximum(result["lower"]-y,0)+np.maximum(y-result["upper"],0)
            metrics["mean_scaled_interval_score_overall"]=float((result["width_scaled"]+40*miss).mean())
            write_json(method/"coverage_metrics.json",metrics)
            report=reconcile_saved({"trajectory":result},{"trajectory":q},saved_dir=root,test=test,
                stats=stats,profiles=["sample"],target_names=["x_steam_out"])["ensemble_conformal_target_trajectory"]
            self.assertEqual(report["coverage_disagreement_count"],1)
            self.assertEqual(report["canonical_scaled_coverage"]["pointwise"],0)
            self.assertEqual(report["legacy_float32_physical_coverage"]["pointwise"],1)
            self.assertEqual(report["saved_interval_arrays"]["status"],"not_available")


class IntegrationTests(unittest.TestCase):
    def fixture(self, root):
        exp=root/"ensemble"; exp.mkdir()
        counts={"train":2,"val":1,"cal":20,"test":5}
        manifests={s:[f"{s}_{i:02d}" for i in range(n)] for s,n in counts.items()}
        names=["TN2","x_steam_out"]
        stats={"x":{"min":np.asarray([10,.5,30.]),"span":np.asarray([8,.5,4.])},
               "y":{"min":np.asarray([10,.5]),"span":np.asarray([8,.5])}}
        rng=np.random.default_rng(91)
        raw_path=root/"raw.h5"; scaled_path=root/"scaled.h5"
        with h5py.File(raw_path,"w") as raw,h5py.File(scaled_path,"w") as scaled:
            for f in (raw,scaled):
                f.attrs["state_feature_names"]=np.asarray(names,dtype="S")
                f.attrs["control_feature_name"]="drumAngleDeg"
            scaled.attrs["split_mode"]="profile"
            scaled.attrs["scaling_type"]="minmax"
            g=scaled.create_group("scaling")
            for axis in stats:
                for key,a in stats[axis].items(): g.create_dataset(f"{axis}_{key}",data=a)
                g.create_dataset(f"{axis}_max",data=stats[axis]["min"]+stats[axis]["span"])
            for split,profiles in manifests.items():
                for name in profiles:
                    # Training covers exactly 0..1. Other splits intentionally
                    # extend the range; including them would fail train-only fit.
                    base=np.asarray([0,.5,1.]) if split=="train" else rng.random(3)*2
                    ys=np.column_stack([base,base])
                    xs=np.repeat(np.column_stack([base,base,base])[:,None,:],2,axis=1)
                    rg=raw.create_group(f"files/{name}")
                    sg=scaled.create_group(f"{split}/files/{name}")
                    for axis,key,a in (("x","X",xs),("y","Y",ys)):
                        physical=a*stats[axis]["span"]+stats[axis]["min"]
                        rg.create_dataset(key,data=physical)
                        sg.create_dataset(key,data=(physical-stats[axis]["min"])/stats[axis]["span"])
        write_json(root/"scaler.json",{"type":"minmax", **{axis:{key:a.astype(np.float32) for key,a in st.items()} for axis,st in stats.items()}})
        for i in range(10):
            (exp/f"ensemble/model_{i}").mkdir(parents=True)
            (exp/f"ensemble/model_{i}/model.pt").write_bytes(b"fixed-checkpoint")
        arrays={}
        for split in ("cal","test"):
            ys=[];means=[];spreads=[];physical=[]
            with h5py.File(root/f"{split}.h5","w") as f,h5py.File(scaled_path,"r") as scaled:
                f.attrs["target_names"]=np.asarray(names,dtype="S")
                f.attrs["member_count"]=10; f.attrs["ensemble_std_ddof"]=0
                for name in manifests[split]:
                    sg=scaled[f"{split}/files/{name}"]
                    y=sg["Y"][()].astype(np.float32)
                    members=y[None,:,:]+rng.normal(0,.01,(10,3,2)).astype(np.float32)
                    mean=members.mean(axis=0);spread=members.std(axis=0)
                    phys=y*stats["y"]["span"].astype(np.float32)+stats["y"]["min"].astype(np.float32)
                    ys.append(y);means.append(mean);spreads.append(spread);physical.append(phys)
                    g=f.create_group(name)
                    for key,a in {"y_true_scaled":y,"member_predictions_scaled":members,
                                  "ensemble_mean_scaled":mean,"ensemble_std_scaled":spread,
                                  "y_true_physical":phys,"t":np.arange(3,dtype=np.float32),
                                  "control":sg["X"][:,-1,-1].astype(np.float32)*4+30}.items():
                        g.create_dataset(key,data=a)
            arrays[split]={"truth":np.asarray(ys),"mean":np.asarray(means),"spread":np.asarray(spreads),"truth_physical":np.asarray(physical)}
        saved=root/"saved";saved.mkdir()
        for floor,prefix in ((None,"absolute"),(.001,"ensemble")):
            cal=arrays["cal"]
            scores=np.abs(cal["truth"]-cal["mean"])
            if floor is not None: scores/=cal["spread"]+np.float32(floor)
            quantiles={"horizon":np.sort(scores,axis=0)[19],"trajectory":np.sort(scores.max(axis=1),axis=0)[19]}
            for mode,q in quantiles.items():
                method=saved/f"{prefix}_conformal_target_{mode}";method.mkdir()
                data={"alpha":.05,"calibration_profile_names":manifests["cal"],
                      "q_by_target" if mode=="trajectory" else "q_by_horizon_target":q}
                if floor is not None:data["sigma_floor"]=np.full(2,np.float32(floor))
                write_json(method/"calibration_metadata.json",data)
                test=arrays["test"]
                half=q*(test["spread"]+np.float32(floor)) if floor is not None else q
                lower,upper=test["mean"]-half,test["mean"]+half
                mask=(lower<=test["truth"])&(test["truth"]<=upper)
                pmask=(lower*stats["y"]["span"].astype(np.float32)+stats["y"]["min"].astype(np.float32)<=test["truth_physical"])&(test["truth_physical"]<=upper*stats["y"]["span"].astype(np.float32)+stats["y"]["min"].astype(np.float32))
                metrics=legacy_metrics(coverage(mask),coverage(pmask),names,mode)
                width=upper.astype(float)-lower.astype(float)
                metrics["mean_scaled_interval_width_overall"]=float(width.mean())
                metrics["mean_scaled_interval_score_overall"]=float((width+40*(np.maximum(lower-test["truth"],0)+np.maximum(test["truth"]-upper,0))).mean())
                write_json(method/"coverage_metrics.json",metrics)
                with h5py.File(method/"test_forecasts.h5","w") as f:
                    f.attrs["target_names"]=np.asarray(names,dtype="S")
                    for i,name in enumerate(manifests["test"]):
                        g=f.create_group(name);g.create_dataset("lower_scaled",data=lower[i]);g.create_dataset("upper_scaled",data=upper[i])
        config={"output_dir":str(root/"study"),"ensemble_experiment_dir":str(exp),
                "external_scaled_h5_path":str(scaled_path),"training_scaled_h5_path":str(scaled_path),
                "raw_h5_path":str(raw_path),"reference_scaler_json_path":str(root/"scaler.json"),
                "calibration_forecasts_h5_path":str(root/"cal.h5"),"test_forecasts_h5_path":str(root/"test.h5"),
                "saved_uq_comparison_dir":str(saved),"expected_profile_counts":counts,
                "expected_steps":3,"expected_member_count":10,"plots":False}
        for split,names_ in manifests.items():
            path=root/f"{split}_manifest.json";write_json(path,{f"{split}_profiles":names_})
            config[f"{split}_manifest_path"]=str(path)
        return config,manifests,names

    def test_end_to_end_fixed_inputs_recalibration_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,manifests,names=self.fixture(root)
            before=fingerprint(root/"test.h5")
            out=run(config)
            self.assertTrue(json.loads((out/"scaler_verification.json").read_text())["match"])
            self.assertEqual(before,fingerprint(root/"test.h5"))
            with h5py.File(out/"recalibration.h5","r") as f:
                self.assertEqual(len(f.keys()),8)
                self.assertEqual(f.attrs["quantile_rank"],20)
                self.assertEqual(f["floor_0.001/trajectory_scores"].shape,(20,2))
                self.assertFalse(np.array_equal(f["floor_1e-06/trajectory"][()],f["floor_0.01/trajectory"][()]))
            report=json.loads((out/"saved_run_reconciliation.json").read_text())
            self.assertTrue(all(x["quantiles"]["match"] for x in report.values()))
            self.assertTrue(all(c["match"] for x in report.values() for c in x["saved_coverage_count_comparisons"].values()))
            self.assertTrue(all(c[key]["match"] for x in report.values() for c in x["saved_interval_arrays"]["profiles"] for key in ("lower","upper")))
            self.assertEqual(len(json.loads((out/"floor_fraction_distributions.json").read_text())),28)
            self.assertTrue(json.loads((out/"study_metadata.json").read_text())["completed"])
            metadata=json.loads((out/"study_metadata.json").read_text())
            self.assertTrue(all("/ensemble/model_" in metadata["input_hashes"][f"checkpoint_{i}"]["path"]
                                for i in range(10)))
            with self.assertRaises(FileExistsError):run(config)

    def test_actual_scaled_samples_detect_corruption_with_matching_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,manifests,names=self.fixture(root)
            with h5py.File(root/"scaled.h5","r+") as f:
                key=f"test/files/{manifests['test'][0]}/Y"
                f[key][0,1]+=1e-9
            report,_=verify_scaler(external_path=root/"scaled.h5",training_path=root/"scaled.h5",
                raw_path=root/"raw.h5",reference_json_path=root/"scaler.json",manifests=manifests,target_names=names)
            self.assertFalse(report["match"])
            self.assertTrue(all(x["match"] for x in report["metadata_comparisons"].values()))
            self.assertTrue(any(not x["match"] for x in report["sample_checks"]))

    def test_training_samples_detect_corruption_independently_of_external_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,manifests,names=self.fixture(root)
            training_path=root/"training_scaled.h5"
            shutil.copyfile(root/"scaled.h5",training_path)
            with h5py.File(training_path,"r+") as f:
                key=f"train/files/{manifests['train'][0]}/X"
                f[key][0,-1,-1]+=1e-9
            report,_=verify_scaler(external_path=root/"scaled.h5",training_path=training_path,
                raw_path=root/"raw.h5",reference_json_path=root/"scaler.json",manifests=manifests,target_names=names)
            self.assertFalse(report["match"])
            self.assertTrue(all(x["match"] for x in report["metadata_comparisons"].values()))
            self.assertTrue(all(x["match"] for x in report["sample_checks"] if x["source"]=="external"))
            self.assertTrue(any(not x["match"] for x in report["sample_checks"] if x["source"]=="training"))

    def test_failed_scaler_check_leaves_incomplete_record_and_runs_no_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,manifests,names=self.fixture(root)
            with h5py.File(root/"scaled.h5","r+") as f:
                f[f"test/files/{manifests['test'][0]}/Y"][0,1]+=1e-9
            with self.assertRaisesRegex(ValueError,"Scaler/data checks failed"):
                run(config)
            out=Path(config["output_dir"])
            self.assertFalse(json.loads((out/"study_metadata.json").read_text())["completed"])
            self.assertFalse((out/"recalibration.h5").exists())


if __name__ == "__main__":
    unittest.main()
