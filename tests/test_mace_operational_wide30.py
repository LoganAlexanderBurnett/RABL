"""Broad recipes, retained calibration provenance and new-only orchestration."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _mace_floor_study import fingerprint,write_json
from _mace_operational_profiles import (RECIPE,WIDE_RECIPE,TIME,control,generate,load_manifest,phase_masks)
from _mace_original_calibration_cache import validate_cache
from run_mace_operational_validation import configuration,run,source_hashes
from test_mace_operational_plots import fixture


class BroadTests(unittest.TestCase):
    def test_independent_broad_profiles_cover_both_sides_with_valid_constant_speeds(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            m=generate(root/'wide',calibration_count=1000,test_count=2000,seed=20261012,settling_seconds=20,recipe=WIDE_RECIPE)
            self.assertEqual(load_manifest(root/'wide')['identity'],m['identity'])
            self.assertEqual(m['allowed_angle_degrees'],[15,75])
            self.assertEqual(m['profiles'][0]['id'],'results_drum_profile_30001')
            self.assertEqual(len({json.dumps(p['moves'],sort_keys=True) for p in m['profiles']}),3000)
            for split,n in [('cal',1000),('test',2000)]:
                selected=[p for p in m['profiles'] if p['split']==split]
                self.assertEqual(len(selected),n)
                for family in 'BCDF':
                    ps=[p for p in selected if p['family']==family]
                    angles=np.concatenate([control(p['moves'])[0] for p in ps])
                    self.assertGreaterEqual(angles.min(),15)
                    self.assertLessEqual(angles.max(),75)
                    self.assertLess(angles.min(),17)
                    self.assertGreater(angles.max(),73)
            for p in m['profiles']:
                angle,velocity=control(p['moves'])
                self.assertEqual(angle[0],45)
                self.assertLessEqual(abs(velocity).max(),1.5)
                moves=p['moves']
                self.assertTrue(all(m['start_s']<m['end_s']<=200 for m in moves))
                self.assertTrue(all(a['end_s']<=b['start_s'] for a,b in zip(moves,moves[1:])))
                self.assertTrue(all(.01<=abs(m['speed_deg_s'])<=1.5 for m in moves))
                masks=phase_masks(moves)
                np.testing.assert_array_equal(sum(masks[k].astype(int) for k in ['motion','early_settling','late_hold','initial_hold']),np.ones(501))
                if p['family'] in 'DF':self.assertEqual(angle[-1],45)
            again=generate(root/'again',calibration_count=1000,test_count=2000,seed=20261012,settling_seconds=20,recipe=WIDE_RECIPE)
            self.assertEqual([p['moves'] for p in m['profiles']],[p['moves'] for p in again['profiles']])

    def test_local_recipe_remains_the_same_and_output_protection(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'local'
            m=generate(root,calibration_count=20,test_count=8,seed=20261010,settling_seconds=20)
            p=m['profiles'][0]
            self.assertEqual(m['recipe'],RECIPE)
            self.assertEqual(p['id'],'results_drum_profile_10001')
            self.assertEqual(p['moves'][0]['to_deg'],47.711898)
            self.assertEqual(p['moves'][0]['start_s'],21.6)
            self.assertEqual(p['moves'][0]['end_s'],70.4)
            with self.assertRaises(FileExistsError):generate(root,calibration_count=20,test_count=8,seed=1,settling_seconds=20,recipe=WIDE_RECIPE)

    def test_new_only_preflights_then_runs_only_the_new_output_stages(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'wide';origin=root/'retained'
            payload=dict(output_dir=str(out),floor_study_dir=str(root/'floor'),profile_recipe='wide30',reuse_original_calibration_dir=str(origin),operational_calibration_count=20,operational_test_count=8,plots=False)
            stages=[]
            with mock.patch('run_mace_operational_validation.frozen_inputs',return_value=({}, {}, {})),mock.patch('run_mace_scaler_use_audit.runtime_settings',return_value=(None,{})),mock.patch('_mace_original_calibration_cache.validate_cache') as cache,mock.patch('run_mace_operational_validation.simulate_stage',side_effect=lambda c,m:stages.append(('simulate',c['output_dir']))),mock.patch('run_mace_operational_validation.forecast_stage',side_effect=lambda c,m,root:stages.append(('forecast',c['output_dir']))),mock.patch('run_mace_operational_validation.evaluate_stage',side_effect=lambda c,m:stages.append(('evaluate',c['output_dir']))):
                run(payload,'new-only')
            cache.assert_called_once()
            self.assertEqual(stages,[('simulate',str(out)),('forecast',str(out)),('evaluate',str(out))])
            self.assertFalse(origin.exists())
            self.assertEqual(load_manifest(out)['recipe'],WIDE_RECIPE)
            with self.assertRaises(FileExistsError):run(payload,'new-only')
            with self.assertRaises(ValueError):run({**payload,'reuse_original_calibration_dir':None},'new-only')
            with self.assertRaises(ValueError):configuration({**payload,'output_dir':str(origin)})

    def test_cache_rejects_changed_inputs_runtime_source_and_legacy_feedback(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);origin=root/'old';origin.mkdir()
            manifest,stats,names,ids=fixture(origin)
            shutil.copyfile(origin/'forecasts/operational_test.h5',origin/'forecasts/original_cal.h5')
            fp=origin/'forecasts/forecast_metadata.json';ep=origin/'evaluation/evaluation_metadata.json'
            fm=json.loads(fp.read_text());em=json.loads(ep.read_text())
            fm.update(inputs_unchanged=True,runtime=dict(device='cpu'),original_calibration_profile_names=ids)
            fm['output_hashes']['original_cal.h5']=fingerprint(origin/'forecasts/original_cal.h5')
            em['forecast_hashes']=fm['output_hashes'];em['source_hashes']=source_hashes()
            write_json(fp,fm);write_json(ep,em)
            config=dict(output_dir=str(root/'new'),reuse_original_calibration_dir=str(origin))
            prior=dict(target_names=names,profile_names=dict(cal=ids));hashes=fm['input_hashes']
            valid=validate_cache(config,prior,hashes,root=ROOT,runtime=dict(device='cpu'))
            self.assertEqual(valid['original_manifest_identity'],manifest['identity'])
            with self.assertRaisesRegex(ValueError,'runtime'):validate_cache(config,prior,hashes,root=ROOT,runtime=dict(device='cuda'))
            changed={k:{**v,'sha256':'changed'} for k,v in hashes.items()}
            with self.assertRaisesRegex(ValueError,'frozen'):validate_cache(config,prior,changed,root=ROOT)
            bad={**em,'source_hashes':{**em['source_hashes'],'scripts/_mace_operational_forecasts.py':'unreviewed'}};write_json(ep,bad)
            with self.assertRaisesRegex(ValueError,'recurrence'):validate_cache(config,prior,hashes,root=ROOT)
            write_json(ep,em);write_json(fp,{**fm,'feedback':'legacy-direct-feedback'})
            with self.assertRaisesRegex(ValueError,'corrected'):validate_cache(config,prior,hashes,root=ROOT)


if __name__=='__main__':unittest.main()
