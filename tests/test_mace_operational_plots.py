"""Plot selection, native endpoints, units, ordering, and read-only rendering."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from _mace_floor_study import bounds, fingerprint, write_json
from plot_mace_operational_forecasts import (FEEDBACK, ORDER, TIME, load_profile,
                                             render_grid, run, select_profiles)


def fixture(root):
    for name in ('forecasts', 'evaluation'):
        (root/name).mkdir()
    profiles = [dict(id=f'results_drum_profile_{11000+i:05d}', split='test', family=f,
                     moves=[]) for i, f in enumerate('BBBCCCDDDFFF')][::-1]
    profiles.append(dict(id='results_drum_profile_00001', split='cal', family='B', moves=[]))
    identity = hashlib.sha256(json.dumps(profiles, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = dict(profiles=profiles, identity=identity)
    write_json(root/'operational_manifest.json', manifest)
    names = list(ORDER[3:]+ORDER[:3])  # force a non-display order
    ids = [p['id'] for p in profiles if p['split'] == 'test']
    stats = dict(x=dict(min=np.zeros(16), span=np.ones(16)),
                 y=dict(min=np.arange(15, dtype=float)+300, span=np.arange(15, dtype=float)+10))
    scaler = root/'scaler.h5'
    with h5py.File(scaler, 'w') as f:
        f.attrs.update(state_feature_names=np.asarray(names,dtype='S'), control_feature_name=np.asarray(['drum'],dtype='S'))
        for axis, items in stats.items():
            for k, v in items.items():f.create_dataset(f'scaling/{axis}_{k}', data=v)
    with h5py.File(root/'forecasts/operational_test.h5','w') as f, h5py.File(root/'evaluation/intervals_and_statistics.h5','w') as b, h5py.File(root/'evaluation/calibration_quantiles.h5','w') as q:
        f.attrs.update(completed=True, feedback=FEEDBACK, manifest_identity=identity,
                       member_count=10, ensemble_std_ddof=0,target_names=np.asarray(names,dtype='S'))
        b.attrs.update(target_names=np.asarray(names,dtype='S'),test_profile_names=np.asarray(ids,dtype='S'),coverage_convention='Inclusive float32 scaled endpoints/truth for every coverage metric')
        q.attrs.update(target_names=np.asarray(names,dtype='S'),alpha=.05,common_floor=.001)
        lower = {s:{m:[] for m in ('horizon','target_trajectory','joint_trajectory')} for s in ('original','operational')}
        upper = {s:{m:[] for m in lower[s]} for s in lower}
        masks = {s:{m:[] for m in lower[s]} for s in lower}
        for i, p in enumerate(profiles[:-1]):
            mean = np.broadcast_to(np.arange(15,dtype=np.float32)*.01+.3+i*.001,(501,15)).copy()
            members = np.stack([mean+np.float32((j-4)*.002) for j in range(10)])
            mean, spread = members.mean(axis=0), members.std(axis=0,ddof=0)
            physical = mean.astype(float)*stats['y']['span']+stats['y']['min']+.004
            truth = ((physical-stats['y']['min'])/stats['y']['span']).astype(np.float32)
            g=f.create_group(p['id']);g.attrs.update(family=p['family'],split='test')
            for k,v in dict(mean=mean,spread=spread,truth=truth,truth_physical=physical,
                            member_predictions_scaled=members,time_seconds=TIME,
                            control_physical=45+TIME*.01).items():g.create_dataset(k,data=v)
            for s in lower:
                for m in lower[s]:
                    shape = (501,15) if m=='horizon' else (15,) if m=='target_trajectory' else ()
                    val = np.full(shape,.4 if s=='original' else .8,dtype=np.float32)
                    if i==0:q.create_dataset(f'{s}/mace/{m}',data=val)
                    lo,hi=bounds(dict(mean=mean,spread=spread),val,.001)
                    lower[s][m].append(lo);upper[s][m].append(hi);masks[s][m].append((lo<=truth)&(truth<=hi))
        for s in lower:
            for m in lower[s]:
                for field,values in [('lower',lower[s][m]),('upper',upper[s][m]),('mask',masks[s][m])]:
                    b.create_dataset(f'intervals/{m}/{s}/mace/{field}',data=np.stack(values))
    sf, ff=fingerprint(scaler), fingerprint(root/'forecasts/operational_test.h5')
    fm=dict(completed=True,feedback=FEEDBACK,manifest_identity=identity,target_names=names,
            input_hashes={'external_scaled_h5_path':sf},output_hashes={'operational_test.h5':ff})
    em=dict(completed=True,manifest_identity=identity,target_names=names,test_profile_names=ids,
            coverage_convention='Inclusive float32 scaled endpoints/truth for every coverage metric',alpha=.05,common_floor=.001,
            forecast_hashes=fm['output_hashes'],input_hashes=fm['input_hashes'])
    write_json(root/'forecasts/forecast_metadata.json',fm);write_json(root/'evaluation/evaluation_metadata.json',em)
    return manifest, stats, names, ids


class OperationalPlotTests(unittest.TestCase):
    def test_selection_sorted_test_only_and_requires_all_families(self):
        with tempfile.TemporaryDirectory() as t:
            manifest,*_=fixture(Path(t))
            selected=select_profiles(manifest)
            self.assertEqual(len(selected),12)
            self.assertEqual([p['family'] for p in selected],list('BBBCCCDDDFFF'))
            self.assertTrue(all(p['split']=='test' for p in selected))
            self.assertEqual(selected[0]['id'],'results_drum_profile_11000')
            with self.assertRaisesRegex(ValueError,'fewer'):select_profiles(manifest,4)

    def test_physical_endpoints_native_truth_display_order_and_16_panels(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);manifest,stats,names,ids=fixture(root);p=select_profiles(manifest)[0];i=ids.index(p['id'])
            with h5py.File(root/'forecasts/operational_test.h5') as f,h5py.File(root/'evaluation/intervals_and_statistics.h5') as b,h5py.File(root/'evaluation/calibration_quantiles.h5') as q:
                data=load_profile(f,b,q,profile=p,index=i,names=names,stats=stats,sources=('original','operational'),modes=('target_trajectory','joint_trajectory'),floor=.001)
                np.testing.assert_array_equal(data['truth'],f[p['id']+'/truth_physical'][()])
                saved=b['intervals/target_trajectory/operational/mace/lower'][i]
                np.testing.assert_array_equal(data['bands']['operational']['target_trajectory'][0],saved.astype(float)*stats['y']['span']+stats['y']['min'])
                self.assertTrue(all(c['match'] for c in data['checks']))
            fig=render_grid(data,profile=p,source='operational',mode='joint_trajectory',alpha=.05)
            try:
                self.assertEqual(len(fig.axes),16)
                for ax,name in zip(fig.axes[1:],ORDER):
                    j=names.index(name)
                    factor = 6.0 if name == 'n' else 1.0
                    np.testing.assert_array_equal(ax.lines[-2].get_ydata(),
                                                  data['truth'][:,j]*factor)
                    np.testing.assert_array_equal(ax.lines[-1].get_ydata(),
                                                  data['mean'][:,j]*factor)
                    if name == 'n':
                        self.assertIn('P', ax.get_title())
                        self.assertIn('MW', ax.get_title())
                self.assertIn('MACE joint trajectory',[x.get_text() for x in fig.legends[0].get_texts()])
                self.assertNotIn('MACE target trajectory',[x.get_text() for x in fig.legends[0].get_texts()])
            finally:plt.close(fig)

    def test_one_ulp_endpoint_corruption_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root)
            with h5py.File(root/'evaluation/intervals_and_statistics.h5','r+') as f:
                d=f['intervals/horizon/original/mace/lower'];d[0,0,0]=np.nextafter(d[0,0,0],np.float32(np.inf))
            with self.assertRaisesRegex(ValueError,'Saved array mismatch'):
                run(root,output_dir=root/'plots')
            self.assertFalse((root/'plots').exists())

    def test_profile_order_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root)
            with h5py.File(root/'evaluation/intervals_and_statistics.h5','r+') as f:
                f.attrs['test_profile_names']=f.attrs['test_profile_names'][::-1]
            with self.assertRaisesRegex(ValueError,'profile order mismatch'):run(root)

    def test_render_four_families_preserves_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root)
            before={str(p):fingerprint(p)['sha256'] for p in root.rglob('*') if p.is_file()}
            report=run(root,profiles_per_family=1,calibration='operational',trajectory_mode='target_trajectory',dpi=40,formats=('png',))
            out=root/'forecast_report_4x4'
            self.assertTrue(report['completed']);self.assertEqual(report['grid_count'],4)
            self.assertEqual(len(list(out.rglob('*.png'))),4)
            self.assertEqual(len(list(out.rglob('*.pdf'))),1)
            self.assertEqual(report['pdf_pages'],9)
            self.assertEqual(len([p for p in report['pdf_index'] if p['kind']=='grid']),4)
            self.assertEqual({str(p):fingerprint(p)['sha256'] for p in map(Path,before)},before)
            self.assertEqual([p['family'] for p in report['selected_profiles']],list('BCDF'))
            with self.assertRaises(FileExistsError):run(root)


if __name__=='__main__':unittest.main()
