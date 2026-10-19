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
    stochastic_representative, operational_representatives, verify_representative_widths,
    profile_mean_absolute_error, spread_error_relationship)
from generate_mace_results_tables import _latex_table, make_tables
from generate_mace_paper_figures import selected_figures
from _mace_plot_units import POWER_MW_PER_N, plot_index_and_scale


class ResultsSelectionTests(unittest.TestCase):
    def test_mean_absolute_error_one_value_per_test_trajectory(self):
        truth = np.zeros((501, 15), dtype=np.float32)
        forecast = np.full((501, 15), .024, dtype=np.float32)
        self.assertAlmostEqual(profile_mean_absolute_error(truth, forecast), .024)
        with self.assertRaisesRegex(ValueError, "501x15"):
            profile_mean_absolute_error(np.zeros((501, 14)), forecast)

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
            import _mace_paper_figures as paper
            figures = []
            def inspect_plot(fig, out, stem, opts):
                figures.append(fig)
                reference = fig.axes[1].lines
                self.assertLessEqual(reference[-2].get_linewidth(), .55)
                self.assertLessEqual(reference[-1].get_linewidth(), .55)
                self.assertAlmostEqual(reference[0].get_linewidth(), .36)
                self.assertAlmostEqual(reference[1].get_linewidth(), .36)
                return paper.save_figure(fig, out, stem, opts)
            with mock.patch("_mace_results_figures._stochastic", return_value=d), \
                 mock.patch("_mace_results_figures._save", side_effect=inspect_plot):
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


    def test_stochastic_uq_coverage_limits_and_text_are_readable(self):
        from _mace_results_figures import stochastic_uq_performance
        import matplotlib.pyplot as plt

        names = [f"c[{i}]" for i in range(1, 7)] + [
            "TN2", "Tm", "Thp", "Tf", "Tsg", "T_steam_out",
            "x_steam_out", "n", "rho_dollars"]
        def metric_row(method, mode, **fields):
            return {"method": method, "mode": mode,
                    "floor": "0.001" if method == "mace" else "None", **fields}
        aggregate = [
            metric_row(method, mode,
                       mean_scaled_full_width=str(.01 + .002*k),
                       mean_scaled_interval_score=str(.02 + .003*k))
            for k, (method, mode) in enumerate(
                (("mace", "horizon"), ("mace", "trajectory"),
                 ("absolute", "horizon"), ("absolute", "trajectory")))
        ]
        target = [
            metric_row(method, "trajectory", target=name,
                       objective_coverage=str(.945 + .001*(j % 5)))
            for method in ("mace", "absolute") for j, name in enumerate(names)
        ]
        horizon = [
            metric_row(method, "horizon", target=name,
                       horizon_index=str(t), coverage="0.950")
            for method in ("mace", "absolute") for name in names
            for t in range(501)
        ]
        d = {"paths": {"stochastic_floor_dir": Path(".")},
             "names": names, "inputs": []}
        fig_out = []
        def inspect(fig, _out, _stem, _opts):
            fig_out.append(fig)
            return ["mock_figure.png"]
        with mock.patch("_mace_results_figures._stochastic", return_value=d), \
             mock.patch("_mace_results_figures._verified", return_value={}), \
             mock.patch("_mace_results_figures._read_rows",
                        side_effect=[aggregate, target, horizon]), \
             mock.patch("_mace_results_figures._save", side_effect=inspect):
            result = stochastic_uq_performance({}, Path("."), lambda x: x, {})
        self.assertEqual(result["files"], ["mock_figure.png"])
        self.assertEqual(len(fig_out), 1)
        fig = fig_out[0]
        try:
            panel_a, panel_b, panel_c, panel_d = fig.axes
            np.testing.assert_allclose(panel_a.get_ylim(), (.93, 1.0))
            np.testing.assert_allclose(panel_b.get_ylim(), (.92, .98))
            for ax in (panel_a, panel_b):
                self.assertTrue(all(t.get_fontsize() >= 8.5
                                    for t in ax.get_legend().get_texts()))
            for ax in (panel_c, panel_d):
                labels = [item for item in ax.texts if item.get_text()]
                self.assertEqual(len(labels), 4)
                self.assertTrue(all(t.get_fontsize() >= 8 for t in labels))
                self.assertTrue(all(len(t.get_text().split(".")[-1]) == 3
                                    for t in labels))
        finally:
            plt.close(fig)

    def test_spread_error_has_original_and_three_panel_backup(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            names = [f"y{i}" for i in range(15)]
            ids = [f"profile_{i:02d}" for i in range(12)]
            archive, calibration = out/"test.h5", out/"quant.h5"
            with h5py.File(archive, "w") as f:
                for i, pid in enumerate(ids):
                    g = f.create_group(pid)
                    mean = np.full((501,15), .05+.001*i, dtype=np.float32)
                    truth = np.full((501,15), .06+.002*i, dtype=np.float32)
                    spread = (.001+.0005*i + .0001*np.arange(501)[:,None]
                              + .00007*np.arange(15)[None,:]).astype(np.float32)
                    g["ensemble_mean_scaled"] = mean
                    g["y_true_scaled"] = truth
                    g["ensemble_std_scaled"] = spread
            with h5py.File(calibration, "w") as f:
                f["floor_0.001/horizon"] = np.full((501,15), 1.)
                f["absolute/horizon"] = np.full((501,15), .025)
            d = {"ids":ids, "files":{"test":archive,"quant":calibration}, "inputs":[]}
            import _mace_paper_figures as paper
            sizes = []
            def verify_and_save(fig, outdir, stem, settings):
                sizes.append((stem, len(fig.axes)))
                return paper.save_figure(fig, outdir, stem, settings)
            with mock.patch("_mace_results_figures._stochastic", return_value=d), \
                 mock.patch("_mace_results_figures._save", side_effect=verify_and_save):
                result = spread_error_relationship(
                    {}, out, lambda p:p, {"formats":["png"], "dpi":72})
            self.assertEqual(sizes, [
                ("fig10_spread_error_relationship", 2),
                ("fig10_spread_error_relationship_three_panel", 3)])
            self.assertEqual(result["histogram_sample_count"], len(ids))
            self.assertEqual(len(list(out.glob("fig10_*.png"))), 2)
            self.assertTrue((out/"fig10_trajectory_scaled_mae.csv").exists())
            values = np.loadtxt(out/"fig10_trajectory_scaled_mae.csv",
                                delimiter=",", skiprows=1, usecols=1)
            self.assertEqual(len(values), len(ids))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((out/"spread_error_bins.csv").exists())


class SyntheticResultsTablesTests(unittest.TestCase):
    def test_saved_data_consumed_without_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ["TN2", "Tm", "Thp", "Tf", "Tsg",
                     "T_steam_out", "x_steam_out",
                     *[f"c[{j}]" for j in range(1, 7)], "n", "rho_dollars"]
            span = np.arange(1, 16, dtype=np.float64)
            testfile = root/"test.h5"
            ids = ["profile_a", "profile_b"]
            with h5py.File(testfile, "w") as f:
                for index, pid in enumerate(ids):
                    group = f.create_group(pid)
                    truth = np.zeros((501, 15), dtype=np.float64)
                    mean = np.broadcast_to((.1 + index*.1)*span, truth.shape)
                    group["ensemble_mean_physical"] = mean
                    group["y_true_physical"] = truth
            op = root/"operational"/"evaluation"
            op.mkdir(parents=True)
            stochastic = root/"stochastic"
            validation = root/"validation"
            stochastic.mkdir()
            validation.mkdir()
            def save_csv(dest, rows):
                if not rows:
                    raise ValueError("Empty synthetic table archive")
                with dest.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            common = {"phase": "all", "family": "pooled"}
            acc_rows = [
                dict(common, scope="aggregate", metric=metric,
                     estimate=val, ci_lower=val-.01, ci_upper=val+.01)
                for metric, val in (("mae",.123), ("rmse",.234))
            ]
            for j, name in enumerate(names):
                for metric, val in (("mae",.123), ("rmse",.234)):
                    acc_rows.append(dict(common, scope=name, metric=metric,
                        estimate=val*span[j],
                        ci_lower=(val-.01)*span[j],
                        ci_upper=(val+.01)*span[j]))
            save_csv(op/"accuracy_metrics.csv", acc_rows)

            op_rows = []
            target_rows = []
            aggregate_rows = []
            for mode, operational_mode in (("horizon","horizon"),
                                            ("trajectory","target_trajectory")):
                for kind in ("mace", "absolute"):
                    stoch_cover = .947 if mode == "horizon" else .951
                    scaled_width = .1 if kind == "mace" else .15
                    scaled_score = .2 if kind == "mace" else .25
                    floor = "0.001" if kind == "mace" else "None"
                    aggregate_rows.append(dict(method=kind,mode=mode,floor=floor,
                        objective_coverage=stoch_cover,
                        mean_scaled_full_width=scaled_width,
                        mean_scaled_interval_score=scaled_score))
                    for j,name in enumerate(names):
                        target_rows.append(dict(
                            method=kind,mode=mode,floor=floor,target=name,
                            objective_coverage=stoch_cover,
                            mean_physical_full_width=scaled_width*span[j],
                            mean_physical_interval_score=scaled_score*span[j]))
                    for source in ("original", "operational"):
                        coverage = .94 if source == "original" else .96
                        for scope in ["aggregate"] + names:
                            index=names.index(scope) if scope != "aggregate" else None
                            factor=1.0 if index is None else span[index]
                            for metric,val in (
                                ("objective_coverage",coverage),
                                ("mean_full_width",scaled_width*factor),
                                ("mean_interval_score",scaled_score*factor)):
                                op_rows.append(dict(common,scope=scope,
                                    mode=operational_mode,
                                    calibration=source,method=kind,metric=metric,
                                    estimate=val,ci_lower=val-.01*factor,
                                    ci_upper=val+.01*factor))
            save_csv(op/"uq_method_metrics.csv",op_rows)
            save_csv(op/"paired_comparisons.csv", [
                dict(common,mode="horizon",scope="aggregate",
                     comparison="mace:operational_minus_original",
                     metric="objective_coverage_difference",
                     estimate=.02,ci_lower=.01,ci_upper=.03)
            ])
            save_csv(stochastic/"target_metrics.csv",target_rows)
            save_csv(stochastic/"aggregate_metrics.csv",aggregate_rows)
            joint_rows = [
                dict(mode="joint",method=kind,scope="aggregate",metric=metric,
                     estimate=val,ci_lower=val-.005,ci_upper=val+.005)
                for kind in ("mace", "absolute")
                for metric,val in (("objective_coverage",.95),
                                   ("mean_full_width",.5),
                                   ("mean_interval_score",.6))
            ]
            save_csv(validation/"bootstrap_method_metrics.csv",joint_rows)
            sto = dict(ids=ids,names=names,files={"test":testfile},
                       stats={"y":{"span":span}},inputs=[],
                       paths={"stochastic_floor_dir":stochastic,
                              "stochastic_validation_dir":validation})
            operational = dict(ids=ids,names=names,inputs=[],
                               paths={"operational_dir":root/"operational"})
            with mock.patch("generate_mace_results_tables._stochastic",return_value=sto), \
                 mock.patch("generate_mace_results_tables._operational",return_value=operational):
                contents, inputs, summary = make_tables({})
            self.assertEqual(set(contents),{
                "table05_forecast_accuracy.tex","table06_operational_uq.tex",
                "table_metrics_detail.csv"})
            table5=contents["table05_forecast_accuracy.tex"]
            table6=contents["table06_operational_uq.tex"]
            self.assertIn("Stoch.",table5)
            self.assertIn("0.1500",table5)
            self.assertIn("0.1230",table5)
            self.assertIn("0.9470/0.1000/0.2000",table5)
            self.assertIn("MACE-Joint",table5)
            self.assertIn("Abs-Joint",table5)
            self.assertIn("0.9500",table5)
            self.assertIn("Stochastic calibration",table6)
            self.assertIn("Independent operation-informed calibration",table6)
            self.assertIn("0.9400/0.1000/0.2000",table6)
            self.assertIn("0.9600/0.1000/0.2000",table6)
            self.assertEqual(table5.count(r"\begin{tabular}"), 3)
            self.assertEqual(table6.count(r"\begin{tabular}"), 2)
            self.assertNotIn("phase breakdown",table6)
            detail=list(csv.DictReader(contents["table_metrics_detail.csv"].splitlines()))
            per_target=[r for r in detail
                        if r["dataset"]=="operational" and
                           r["calibration"]=="operation-informed" and
                           r["output"]=="T_steam_out" and
                           r["method"]=="mace" and
                           r["mode"]=="trajectory" and
                           r["metric"]=="mean_full_width"]
            self.assertEqual(len(per_target),3)
            self.assertAlmostEqual(float(next(x["value"] for x in per_target
                              if x["bound"]=="estimate")),.1)
            self.assertEqual(summary["per_output_count"],15)
            self.assertAlmostEqual(summary["stochastic_mae"],.15)
            self.assertAlmostEqual(summary["stochastic_rmse"],
                                   np.sqrt((.1**2+.2**2)/2))
            self.assertAlmostEqual(summary["operational_mae"],.123)


if __name__ == "__main__":
    unittest.main()
