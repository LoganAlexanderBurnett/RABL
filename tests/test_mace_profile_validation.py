"""Tests of paired resampling, joint calibration and fixed-input validation."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from _mace_floor_study import evaluate, fingerprint
from _mace_profile_bootstrap import (FIELDS, profile_statistics, resample_statistics,
                                    safe_reduction, scoped, interval_row)
from _mace_horizon_diagnostics import contiguous_regions
from run_mace_profile_validation import joint_calibration, run
import test_mace_floor_sensitivity as floor_test
from run_lstm_floor_sensitivity import run as floor_run


def fake_statistics(width):
    width=np.asarray(width,dtype=float)
    return {"width_scaled":width,"score_scaled":2*width,
            "pointwise_coverage":(width>2).astype(float),
            "trajectory_coverage":(width>3).astype(float),
            "joint_coverage":(width>3).all(axis=1).astype(float)[:,None]}


class BootstrapTests(unittest.TestCase):
    def test_count_weighted_means_equal_direct_profile_resampling(self):
        st=fake_statistics([[1,2],[3,4],[5,6]])
        samples,digest=resample_statistics({"horizon/mace":st,"horizon/absolute":st},replicates=9,seed=123,batch_size=4)
        indices=np.random.Generator(np.random.PCG64(123)).integers(0,3,size=(9,3),dtype=np.int64)
        self.assertEqual(digest,hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
        for field in FIELDS:
            expected=np.asarray([st[field][row].mean(axis=0) for row in indices])
            np.testing.assert_allclose(samples['horizon/mace'][field],expected,atol=1e-15,rtol=0)
            np.testing.assert_array_equal(samples['horizon/mace'][field],samples['horizon/absolute'][field])

    def test_draws_and_statistics_are_batch_size_invariant(self):
        st=fake_statistics(np.arange(24).reshape(8,3)+1)
        a,ah=resample_statistics({'mace':st},replicates=21,seed=91,batch_size=1)
        b,bh=resample_statistics({'mace':st},replicates=21,seed=91,batch_size=8)
        self.assertEqual(ah,bh)
        for field in FIELDS:np.testing.assert_array_equal(a['mace'][field],b['mace'][field])

    def test_profile_reduction_preserves_time_and_target_events(self):
        truth=np.zeros((3,501,2),np.float32)
        truth[1,-1,0]=2;truth[2,10,1]=2
        result=evaluate({'truth':truth,'mean':np.zeros_like(truth),'spread':np.zeros_like(truth)},
                        np.ones(2,np.float32),floor=None,alpha=.05,span=np.array([2.,3.]))
        st=profile_statistics(result)
        np.testing.assert_array_equal(st['trajectory_coverage'],[[1,1],[0,1],[1,0]])
        np.testing.assert_array_equal(st['joint_coverage'],[[1],[0],[0]])
        self.assertEqual(st['pointwise_coverage'].shape,(3,2))
        self.assertEqual(st['joint_coverage'].mean(),1/3)

    def test_ratio_of_means_and_physical_units(self):
        a=fake_statistics([[1],[9]]);b=fake_statistics([[1],[3]])
        point={field:v.mean(axis=0) for field,v in a.items()}
        value=scoped(point,index=0,span=np.array([10.]),mode='trajectory')
        self.assertEqual(value['mean_full_width'],50.)
        self.assertEqual(safe_reduction(a['width_scaled'].mean(),b['width_scaled'].mean()),-1.5)
        self.assertNotEqual(-1.5,float(safe_reduction(a['width_scaled'],b['width_scaled']).mean()))

    def test_undefined_ratios_are_reported(self):
        draws=safe_reduction(np.array([1.,2.]),np.array([0.,4.]))
        report=interval_row(.5,draws)
        self.assertEqual(report['undefined_replicates'],1)
        self.assertEqual(report['valid_replicates'],1)
        self.assertIsNone(interval_row(np.nan,np.array([np.nan]))['ci_lower'])

    def test_joint_quantile_uses_profile_joint_max_not_max_of_target_quantiles(self):
        maxima=np.array([[8.,0.],[0.,8.],[0.,0.]],dtype=np.float32)
        q,scores=joint_calibration(maxima,rank=2)
        np.testing.assert_array_equal(scores,[8,8,0])
        self.assertEqual(q,8)
        self.assertEqual(np.sort(maxima,axis=0)[1].max(),0)
        with self.assertRaises(ValueError):joint_calibration(maxima,rank=4)

    def test_contiguous_regions_include_boundaries_and_empty_case(self):
        self.assertEqual(contiguous_regions([True,True,False,True,False,True]),[(0,1),(3,3),(5,5)])
        self.assertEqual(contiguous_regions([False,False]),[])


class WorkflowTests(unittest.TestCase):
    def make_study(self,root):
        floor_config,profiles,names=floor_test.IntegrationTests().fixture(root)
        study=floor_run(floor_config)
        config={'floor_study_dir':str(study),'output_dir':str(root/'validation'),
                'bootstrap_replicates':60,'bootstrap_seed':43,'bootstrap_batch_size':7,
                'plots':False,'include_joint':True,'worst_cells_per_target':2}
        return config,profiles,names

    def test_workflow_joint_pairing_and_immutable_scientific_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,profiles,names=self.make_study(root)
            before=fingerprint(root/'test.h5')
            out=run(config)
            self.assertEqual(before,fingerprint(root/'test.h5'))
            meta=json.loads((out/'validation_metadata.json').read_text())
            self.assertTrue(meta['completed']);self.assertTrue(meta['calibration_fixed_during_bootstrap'])
            with h5py.File(out/'joint_calibration.h5','r') as f:
                self.assertEqual(f.attrs['rank'],20)
                for method in ('mace','absolute'):
                    scores=f[f'{method}/profile_scores'][()]
                    self.assertEqual(f[f'{method}/q_joint'][()],np.sort(scores)[19])
            with h5py.File(out/'profile_statistics.h5','r') as f:
                self.assertEqual(set(f.keys()),{'horizon','trajectory','joint'})
                for method in ('mace','absolute'):
                    joint=f[f'joint/{method}/joint_coverage'][()]
                    targetwise=f[f'trajectory/{method}/joint_coverage'][()]
                    self.assertTrue(np.all(joint>=targetwise))
            with h5py.File(out/'bootstrap_distributions.h5','r') as f:
                self.assertEqual(f['comparisons/trajectory/aggregate/mean_full_width_relative_reduction'].shape,(60,))
            comparisons=list(csv.DictReader((out/'bootstrap_comparisons.csv').read_text().splitlines()))
            self.assertTrue(any(r['mode']=='joint' and r['scope']=='aggregate' and r['metric']=='objective_coverage_difference' for r in comparisons))
            cells=list(csv.DictReader((out/'horizon_diagnostics.csv').read_text().splitlines()))
            self.assertEqual(len(cells),2*3*2)
            regimes=list(csv.DictReader((out/'control_regime_metrics.csv').read_text().splitlines()))
            for method in ('absolute','mace'):
                for target in names:
                    self.assertEqual(sum(int(r['point_count']) for r in regimes if r['method']==method and r['target']==target),15)
            with self.assertRaises(FileExistsError):run(config)

    def test_joint_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,_,_=self.make_study(root);config['include_joint']=False
            out=run(config)
            self.assertFalse((out/'joint_calibration.h5').exists())
            with h5py.File(out/'profile_statistics.h5','r') as f:
                self.assertEqual(set(f.keys()),{'horizon','trajectory'})

    def test_changed_forecasts_are_rejected_before_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,profiles,_=self.make_study(root)
            with h5py.File(root/'test.h5','r+') as f:f[f"{profiles['test'][0]}/y_true_scaled"][0,0]+=1e-3
            with self.assertRaisesRegex(ValueError,'Input differs from the completed floor study'):
                run(config)
            self.assertFalse((Path(config['output_dir'])/'bootstrap_distributions.h5').exists())

    def test_changed_saved_quantiles_are_rejected_with_incomplete_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,_,_=self.make_study(root)
            with h5py.File(root/'study/recalibration.h5','r+') as f:f['floor_0.001/trajectory'][0]+=1
            with self.assertRaisesRegex(ValueError,'Frozen calibration mismatch'):
                run(config)
            metadata=json.loads((Path(config['output_dir'])/'validation_metadata.json').read_text())
            self.assertFalse(metadata['completed'])


if __name__=='__main__':unittest.main()
