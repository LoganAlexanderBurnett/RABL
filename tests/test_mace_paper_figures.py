"""Scientific covariance, data selection, retained scope, and runner integrity."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from _mace_paper_figures import ASSETS, DISPLAY, csv_columns, sampling_covariance
from generate_mace_paper_figures import run, selected_figures


def make_fixture(root):
    config = json.loads((ROOT/'scripts/mace_paper_figures_config.json').read_text())
    config.update(output_dir='plots', assets_dir='assets', formats=['png', 'pdf'], dpi=72)
    config['sampling'].update(profiles_dir='controls', count=3, intervals=16, mode='stored')
    config['dataset'].update(simulation_dir='sim', expected_profiles=3,
                             existing_image='assets/timeseries_batch_0003_ALL_PROFILES.png')
    config['operational'].update(results_dir='operational', scaler_file=None, profiles_per_family=1)
    for folder in ['assets', 'controls', 'sim', 'operational']:
        (root/folder).mkdir()
    for name in (*ASSETS, 'timeseries_batch_0003_ALL_PROFILES.png'):
        Image.new('RGB', (100, 60), color='white').save(root/'assets'/name)
    gen, t, cov, *_ = sampling_covariance(config['sampling'])
    b = gen.build_integration_matrix(t)
    for i in range(3):
        v = np.linspace(0, .01*(i+1), len(t)-1)
        angle = 45+b@v
        np.savetxt(root/'controls'/f'drum_profile_{i:05d}.csv',
                   np.column_stack([t, angle, np.r_[v, np.nan]]), delimiter=',',
                   header='Time(s),Drum_Angle(deg),Drum_Velocity(deg/s)', comments='')
        y = np.column_stack([t, *[np.full(len(t), j+i) for j in range(16)]])
        np.savetxt(root/'sim'/f'results_drum_profile_{i:05d}.csv', y, delimiter=',',
                   header=','.join(('t', *DISPLAY)), comments='')
    profiles=[]
    for family in 'BCDF':
        for i in [3, 1, 2]:
            profiles.append(dict(id=f'{family}_{i}', family=family, split='test',
                                 moves=[dict(start_s=30, end_s=50, from_deg=45.,
                                             to_deg=46., speed_deg_s=.05)]))
    identity=hashlib.sha256(json.dumps(profiles, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (root/'operational'/'operational_manifest.json').write_text(json.dumps(
        dict(recipe='BCDF-independent-parameter-draws-v1', profiles=profiles, identity=identity)))
    file=root/'config.json';file.write_text(json.dumps(config))
    return config, file


class PaperFigureTests(unittest.TestCase):
    def test_conditioned_dispersion_matches_actual_dense_linear_transform(self):
        config=json.loads((ROOT/'scripts/mace_paper_figures_config.json').read_text())
        config['sampling']['intervals']=16
        gen,t,k,u,c=sampling_covariance(config['sampling'])
        transform=np.eye(len(k))-np.outer(k[:,0]/(k[0,0]+gen.cond_jitter), np.eye(len(k))[0])
        transform[0]=0
        b=gen.build_integration_matrix(t)
        expected=b@transform@k@transform.T@b.T
        np.testing.assert_allclose(c*c, np.diag(expected), rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(u*u, np.diag(b@k@b.T), rtol=3e-14, atol=3e-14)
        self.assertEqual(c[1],0)
        self.assertLess(c[-1],u[-1])
        config['sampling']['duration_seconds']=100
        with self.assertRaisesRegex(ValueError,'0–200'):
            sampling_covariance(config['sampling'])

    def test_control_legend_only_in_second_panel(self):
        from unittest import mock
        import _mace_paper_figures as paper
        original_save = paper.save_figure
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,file=make_fixture(root)
            inspection = []
            def spy_save(fig, directory, stem, options):
                if stem == 'fig06_operational_controls':
                    inspection.append(
                        [ax.get_legend() is not None for ax in fig.axes])
                    legend = fig.axes[1].get_legend()
                    self.assertEqual(
                        [x.get_text() for x in legend.get_texts()],
                        ['Trajectory 1'])
                    self.assertEqual(legend._ncols, 1)
                return original_save(fig, directory, stem, options)
            with mock.patch.object(paper, 'save_figure', side_effect=spy_save):
                status=run(file, figures=['operational_controls'], repo_root=root)
            self.assertTrue(status['completed'])
            self.assertEqual(inspection, [[False, True, False, False]])

    def test_selective_run_units_order_data_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,file=make_fixture(root)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            result=run(file, figures=['sampling,dataset', 'operational_controls', 'existing_assets'], repo_root=root)
            self.assertTrue(result['completed'])
            self.assertFalse((root/'plots'/'operational_forecasts').exists())
            for p,digest in before.items():self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),digest)
            sm=json.loads((root/'plots'/'sampling'/'figure_metadata.json').read_text())
            self.assertEqual(sm['selected_ids'],[f'drum_profile_{i:05d}' for i in range(3)])
            cm=json.loads((root/'plots'/'operational_controls'/'figure_metadata.json').read_text())
            self.assertEqual(cm['selected_ids'],['B_1','C_1','D_1','F_1'])
            self.assertTrue((root/'plots'/'operational_controls'/'fig06_operational_controls.png').exists())
            self.assertEqual([entry['label'] for entry in cm['trajectory_labels_and_extrema']],
                             ['Trajectory 1']*4)
            dm=json.loads((root/'plots'/'dataset'/'figure_metadata.json').read_text())
            self.assertEqual(dm['display_order'],list(DISPLAY))
            self.assertEqual(dm['profile_count'],3)
            asset=root/'plots'/'existing_assets'/'hpmr_bop_viz.png'
            self.assertEqual(asset.read_bytes(),(root/'assets'/'hpmr_bop_viz.png').read_bytes())
            with self.assertRaises(FileExistsError):run(file, figures=['sampling'], repo_root=root)
            run(file, figures=['sampling'], repo_root=root, overwrite=True)
            index=json.loads((root/'plots'/'figure_index.json').read_text())
            self.assertIn('dataset',index['figures'])

    def test_incomplete_dataset_and_bad_sampling_grid_fail_without_replacing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,file=make_fixture(root)
            run(file,figures=['sampling'],repo_root=root)
            prior=(root/'plots'/'sampling'/'sampling_data.npz').read_bytes()
            control=root/'controls'/'drum_profile_00000.csv'
            a=csv_columns(control,('Time(s)','Drum_Angle(deg)','Drum_Velocity(deg/s)'))
            a[1,0]=np.nextafter(a[1,0],np.inf)
            np.savetxt(control,a,delimiter=',',header='Time(s),Drum_Angle(deg),Drum_Velocity(deg/s)',comments='')
            (root/'sim'/'results_drum_profile_00000.csv').unlink()
            result=run(file,figures=['sampling','dataset','existing_assets'],overwrite=True,repo_root=root)
            self.assertFalse(result['completed'])
            self.assertIn('differs exactly',result['statuses']['sampling']['error'])
            self.assertIn('Expected 3',result['statuses']['dataset']['error'])
            self.assertTrue(result['statuses']['existing_assets']['completed'])
            self.assertEqual(prior,(root/'plots'/'sampling'/'sampling_data.npz').read_bytes())
            self.assertFalse((root/'plots'/'dataset').exists())

    def test_image_fallback_preserves_pixels_and_illustrations_are_labeled(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,file=make_fixture(root)
            config['sampling'].update(mode='illustrative',illustration_seed=10)
            config['dataset']['mode']='stored_image';file.write_text(json.dumps(config))
            result=run(file,figures=['sampling','dataset'],repo_root=root)
            self.assertTrue(result['completed'])
            sm=json.loads((root/'plots'/'sampling'/'figure_metadata.json').read_text())
            self.assertEqual(sm['sample_origin'],'illustrative')
            dm=json.loads((root/'plots'/'dataset'/'figure_metadata.json').read_text())
            self.assertFalse(dm['count_verified_from_raw'])
            self.assertEqual((root/'plots'/'dataset'/'fig05_generated_transient_dataset.png').read_bytes(),
                             (root/'assets'/'timeseries_batch_0003_ALL_PROFILES.png').read_bytes())

    def test_all_includes_forecast_report_and_rejects_broader_study(self):
        # Reuse the existing saved-forecast fixture and its exact native endpoint checks.
        from test_mace_operational_plots import fixture
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,file=make_fixture(root)
            op=root/'operational'
            (op/'operational_manifest.json').unlink()
            manifest,*_=fixture(op)
            manifest['recipe']='BCDF-independent-parameter-draws-v1'
            (op/'operational_manifest.json').write_text(json.dumps(manifest))
            result=run(file,figures=['all'],repo_root=root)
            self.assertTrue(result['completed'])
            report=root/'plots'/'operational_forecasts'/'forecast_report'
            self.assertEqual(len(list(report.rglob('*.pdf'))),1)
            self.assertEqual(len(list(report.rglob('*.png'))),16)
            self.assertEqual(set(result['statuses']),set(selected_figures(['all'])))
            manifest['recipe']='BCDF-bidirectional-plus-minus30-duration-draws-v1'
            (op/'operational_manifest.json').write_text(json.dumps(manifest))
            result=run(file,figures=['operational_controls','operational_forecasts'],overwrite=True,repo_root=root)
            self.assertFalse(result['completed'])
            self.assertTrue(all('excluded' in v['error'] for v in result['statuses'].values()))

    def test_output_must_not_replace_source_and_unknown_id_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,file=make_fixture(root)
            with self.assertRaises(ValueError):run(file,figures=['all'],output_dir=root,repo_root=root)
            with self.assertRaises(ValueError):run(file,figures=['oops'],repo_root=root)
            with self.assertRaises(ValueError):selected_figures(['all','sampling'])


if __name__=='__main__':unittest.main()
