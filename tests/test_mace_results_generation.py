"""Small synthetic tests for manuscript Results renderers and LaTeX tables."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _mace_results_figures import (choose_profiles, unsaturated_fraction,
    stochastic_representative, operational_representatives, verify_representative_widths)
from generate_mace_results_tables import _latex_table, make_tables
from generate_mace_paper_figures import selected_figures
from _mace_plot_units import POWER_MW_PER_N, plot_index_and_scale


class ResultsSelectionTests(unittest.TestCase):
    def test_power_is_display_only_6mw_per_n(self):
        names = ["Tf", "n", "x_steam_out"]
        self.assertEqual(POWER_MW_PER_N, 6.0)
        self.assertEqual(plot_index_and_scale("P_MW", names), (1, 6.0))
        self.assertEqual(plot_index_and_scale("n", names), (1, 6.0))
        self.assertEqual(plot_index_and_scale("Tf", names), (0, 1.0))

    def test_one_unsaturated_time_point_is_enough(self):
        a = np.ones((501, 15), dtype=np.float32)
        a[200, 4] = .9
        fraction = unsaturated_fraction(a, 4)
        self.assertAlmostEqual(fraction, 1/501)
        profiles = [("z",fraction),("b",fraction),("d",fraction),("a",0.)]
        result, count = choose_profiles(profiles, count=2, seed=17)
        self.assertEqual(count,3)
        self.assertEqual(len(result),2)
        self.assertEqual(len({p[0] for p in result}),2)
        self.assertEqual(result,choose_profiles(list(reversed(profiles)),2,17)[0])
        with self.assertRaisesRegex(ValueError,"only 3 qualify"):
            choose_profiles(profiles,count=4)
        with self.assertRaisesRegex(ValueError,"positive integer"):
            choose_profiles(profiles,count=0)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            choose_profiles([("z",fraction),("z",fraction)])
        a[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"Invalid"):
            unsaturated_fraction(a,4)

    def test_absolute_band_widths_are_not_suppressed_by_float32_rendering(self):
        names = [f"variable_{j}" for j in range(15)]
        mean = np.full((501,15), .5, dtype=np.float32)
        spread = np.full((501,15), .01, dtype=np.float32)
        tiny = np.float32(1e-8)
        quantiles = {(m,mode):(np.full((501,15),tiny,dtype=np.float32)
                    if mode == "horizon" else np.full(15,tiny,dtype=np.float32))
                    for m in ("mace","absolute") for mode in ("horizon","trajectory")}
        stats = {"y":{"span":np.ones(15),"min":np.zeros(15)}}
        bands, diagnostics = verify_representative_widths(
            mean, spread, quantiles, stats, names)
        low, high = bands[("absolute","horizon")]
        np.testing.assert_allclose(high-low, 2*float(tiny), atol=1e-15, rtol=1e-7)
        row = next(row for row in diagnostics if row["method"]=="absolute"
                   and row["calibration"]=="horizon"
                   and row["target"]=="variable_0")
        self.assertAlmostEqual(row["expected_width_physical_mean"], 2*float(tiny))
        self.assertGreater(row["old_float32_zero_fraction_positive"], 0.9)
        self.assertEqual(len(diagnostics), 4*15)
        self.assertAlmostEqual(row["expected_width_display_mean"],
                               2*float(tiny))

    def test_results_registry_does_not_break_legacy_all(self):
        ids = selected_figures(["results"])
        self.assertEqual(len(ids), 5)
        self.assertIn("stochastic_representative", ids)
        self.assertEqual(ids[0], "operational_controls")
        self.assertTrue(set(ids).isdisjoint(selected_figures(["all"])))

    def test_latex_complete_standard_environment(self):
        body = _latex_table("lc", [r"A & B \\"], "Caption", "tab:sample")
        for token in (r"\begin{table}[t]",r"\caption{Caption}",r"\label{tab:sample}",
                      r"\begin{tabular}{lc}",r"\end{table}"):
            self.assertIn(token,body)


    def test_multiple_stochastic_4x4_figures_and_numbered_captions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ["TN2", "Tm", "Thp", "Tf", "Tsg", "T_steam_out",
                     "x_steam_out", *[f"c[{i}]" for i in range(1,7)],
                     "n", "rho_dollars"]
            ids = ["stoc_a", "stoc_b", "stoc_c"]
            forecasts, quantiles = root/"test.h5", root/"quant.h5"
            with h5py.File(forecasts, "w") as f:
                f.attrs["completed"] = True
                f.attrs["target_names"] = np.asarray(names, dtype="S")
                for i, pid in enumerate(ids):
                    g = f.create_group(pid)
                    truth = np.ones((501, 15), dtype=np.float32)
                    if i < 2:
                        truth[100+i, names.index("x_steam_out")] = .995
                    g["y_true_physical"] = truth
                    g["ensemble_mean_scaled"] = np.ones_like(truth)
                    g["ensemble_std_scaled"] = np.full_like(truth, .02)
                    g["control"] = np.full(501, 45., dtype=np.float32)
            with h5py.File(quantiles, "w") as f:
                f.attrs["target_names"] = np.asarray(names, dtype="S")
                f.attrs["alpha"] = .05
                for key in ("absolute", "floor_0.001"):
                    f.create_dataset(key+"/horizon", data=np.full((501, 15), .05))
                    f.create_dataset(key+"/trajectory", data=np.full(15, .05))
            d = {"names": names, "ids": ids, "stats": {"y": {"min": np.zeros(15),
                 "span": np.ones(15)}}, "files": {"test": forecasts, "quant": quantiles},
                 "inputs": [], "representative_count": 2, "representative_seed": 23}
            with mock.patch("_mace_results_figures._stochastic", return_value=d):
                result = stochastic_representative({}, root, lambda p: p,
                                                    {"formats": ["png"], "dpi": 72})
            self.assertEqual(result["generated_count"], 2)
            self.assertEqual(result["eligible_profiles"], 2)
            self.assertEqual(len(list(root.glob("fig07_*.png"))), 2)
            self.assertEqual(len(list(root.glob("fig07_*.caption.md"))), 2)


    def test_two_operational_six_panel_composites_using_all_four_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ["TN2", "Tm", "Thp", "Tf", "Tsg", "T_steam_out",
                     "x_steam_out", *[f"c[{i}]" for i in range(1, 7)],
                     "n", "rho_dollars"]
            ids = [f"{fam}_{i}" for fam in "BCDF" for i in range(2)]
            forecast, intervals = root/"op.h5", root/"intervals.h5"
            with h5py.File(forecast, "w") as f:
                f.attrs["completed"] = True
                f.attrs["target_names"] = np.asarray(names, dtype="S")
                f.attrs["manifest_identity"] = "fixture"
                for pid in ids:
                    fam, i = pid.split("_")
                    g = f.create_group(pid)
                    g.attrs["family"], g.attrs["split"] = fam, "test"
                    truth = np.ones((501, 15), dtype=np.float32)
                    if fam != "F":
                        truth[23+int(i), names.index("x_steam_out")] = .98
                    g["truth_physical"] = truth
                    g["truth"] = np.zeros((501, 15), dtype=np.float32)
                    g["mean"] = np.zeros((501, 15), dtype=np.float32)
                    g["control_physical"] = np.full(501, 45.)
            with h5py.File(intervals, "w") as f:
                f.attrs["test_profile_names"] = np.asarray(ids, dtype="S")
                for key, value in (("lower",-1.), ("upper",1.), ("mask",True)):
                    f.create_dataset("intervals/horizon/operational/mace/"+key,
                        data=np.full((len(ids),501,15),value))
            d = {"names":names, "ids":ids,
                 "stats":{"y":{"span":np.ones(15),"min":np.zeros(15)}},
                 "files":{"test":forecast,"intervals":intervals},
                 "manifest":{"identity":"fixture", "profiles":[
                     {"id":pid,"split":"test","family":pid[0]} for pid in ids]},
                 "inputs":[],"representative_count":2,"representative_seed":23}
            import _mace_paper_figures as paper_figures
            titles, power_trace = [], []
            def check_plot(fig, output_dir, stem, options):
                self.assertEqual(len(fig.axes), 6)
                titles.append([ax.get_title() for ax in fig.axes])
                power_trace.append(fig.axes[1].lines[0].get_ydata()[0])
                self.assertTrue(all(not text.get_text().startswith(("B:", "C:", "D:", "F:"))
                                    for text in fig.legends[0].texts))
                return paper_figures.save_figure(fig, output_dir, stem, options)
            with mock.patch("_mace_results_figures._operational",return_value=d), \
                 mock.patch("_mace_results_figures._save",side_effect=check_plot):
                result = operational_representatives({},root,lambda p:p,
                                                      {"formats":["png"],"dpi":72})
            self.assertEqual(result["generated_count"],2)
            self.assertEqual(result["eligible_profile_counts"],
                             {fam:2 for fam in "BCDF"})
            self.assertEqual(len(list(root.glob("fig09_*.png"))),2)
            self.assertEqual(len(titles), 2)
            self.assertTrue(all("$P$ (MW)" in row[1] and
                                "$T_" in row[4] and "x_" in row[5] for row in titles))
            self.assertTrue(all(abs(v - 6.0) < 1e-8 for v in power_trace))
            for record in result["selected_sets"]:
                self.assertEqual(set(record["profiles"]),set("BCDF"))
                self.assertEqual(record["profiles"]["F"]["unsaturated_fraction"], 0.)
                self.assertFalse(record["selection_require_unsaturated_by_family"]["F"])


class SyntheticResultsTablesTests(unittest.TestCase):
    def test_saved_data_consumed_without_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            testfile=root/"test.h5"
            ids=["profile_a","profile_b"]
            with h5py.File(testfile,"w") as f:
                for index,pid in enumerate(ids):
                    g=f.create_group(pid)
                    truth=np.zeros((501,15),dtype="f4")
                    mean=np.full_like(truth,.1+index*.1)
                    g.create_dataset("ensemble_mean_physical",data=mean)
                    g.create_dataset("y_true_physical",data=truth)
            op=root/"operational"/"evaluation"
            op.mkdir(parents=True)
            def csv_file(name,rows):
                with (op/name).open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
                    writer.writeheader();writer.writerows(rows)
            common={"phase":"all","family":"pooled","scope":"aggregate"}
            csv_file("accuracy_metrics.csv",[
                dict(common,metric="mae",estimate=".123",ci_lower=".11",ci_upper=".14"),
                dict(common,metric="rmse",estimate=".234",ci_lower=".21",ci_upper=".25"),
            ])
            rows=[]
            for source in ("original","operational"):
                for method in ("mace","absolute"):
                    for metric,val in (("objective_coverage",.91 if source=="original" else .95),
                                       ("mean_full_width",.45 if source=="original" else .55),
                                       ("mean_interval_score",.60 if source=="original" else .57)):
                        rows.append(dict(common,mode="horizon",calibration=source,method=method,
                                         metric=metric,estimate=val if method=="mace" else val*1.2,
                                         ci_lower=val-.01,ci_upper=val+.01))
            csv_file("uq_method_metrics.csv",rows)
            differences={"objective_coverage_difference":.04,
                         "mean_full_width_difference":.10,
                         "mean_interval_score_difference":-.03}
            csv_file("paired_comparisons.csv",[
                dict(common,mode="horizon",comparison="mace:operational_minus_original",
                     metric=m,estimate=v,ci_lower=v-.01,ci_upper=v+.01)
                for m,v in differences.items()
            ])
            dummy={"paths":{"operational_dir":root/"operational"},"ids":ids,"inputs":[]}
            sto={"ids":ids,"files":{"test":testfile},"stats":{"y":{"span":np.ones(15)}},
                 "inputs":[]}
            with mock.patch("generate_mace_results_tables._stochastic",return_value=sto), \
                 mock.patch("generate_mace_results_tables._operational",return_value=dummy):
                contents,inputs,summary=make_tables({})
            self.assertEqual(len(contents),2)
            self.assertIn("Stochastic & 2 & 0.1500",contents["table05_forecast_accuracy.tex"])
            self.assertIn("Operation-informed & 2 & 0.1230",contents["table05_forecast_accuracy.tex"])
            self.assertIn("0.9500",contents["table06_operational_uq.tex"])
            self.assertIn("Difference (operational",contents["table06_operational_uq.tex"])
            self.assertAlmostEqual(summary["stochastic_mae"],.15)
            self.assertAlmostEqual(summary["stochastic_rmse"],np.sqrt((.1**2+.2**2)/2))


if __name__ == "__main__":
    unittest.main()
