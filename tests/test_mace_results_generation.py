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

from _mace_results_figures import choose_profile, unsaturated_fraction
from generate_mace_results_tables import _latex_table, make_tables


class ResultsSelectionTests(unittest.TestCase):
    def test_deterministic_unsaturated_truth_selection(self):
        a = np.ones((501,15))
        a[:102,4] = .9
        self.assertAlmostEqual(unsaturated_fraction(a,4),102/501)
        self.assertEqual(choose_profile([("z",.3),("b",.25),("a",.04),("d",.2)],.1),
                         (("d",.2),3))
        with self.assertRaisesRegex(ValueError, "No reference profile"):
            choose_profile([("a",.05)],.1)
        a[0,0]=np.nan
        with self.assertRaisesRegex(ValueError, "Invalid"):
            unsaturated_fraction(a,4)

    def test_latex_complete_standard_environment(self):
        body = _latex_table("lc", [r"A & B \\"], "Caption", "tab:sample")
        for token in (r"\begin{table}[t]",r"\caption{Caption}",r"\label{tab:sample}",
                      r"\begin{tabular}{lc}",r"\end{table}"):
            self.assertIn(token,body)


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
