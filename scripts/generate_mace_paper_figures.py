"""Generate selected MACE paper data figures and export supplied source graphics."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

# Stable descriptive IDs: manuscript numbering can change without changing CLI IDs.
# Add a renderer to _mace_paper_figures.py and one entry here for future sections.
REGISTRY = {
    'sampling': ('sampling_figure', 'Figure 1: velocity correlation and integrated control trajectories'),
    'dataset': ('dataset_figure', 'Figure 5: existing or regenerated 4×4 simulation dataset'),
    'operational_controls': ('operational_controls', 'Figure 6: representative operation-informed control maneuvers'),
    'existing_assets': ('existing_assets', 'Preserve supplied graphics; manual corrections stay manual'),
    'operational_forecasts': ('operational_forecasts', 'Saved 4×4 forecasts: PNGs + one consolidated PDF'),
    'representative_forecast': ('representative_forecast', 'Single saved operational test forecast: 4×4 MACE UQ comparison'),
    'stochastic_representative': ('stochastic_representative', 'Figure 7: four calibrated methods on stochastic test profiles'),
    'stochastic_uq_performance': ('stochastic_uq_performance', 'Figure 8: matched stochastic Horizon/Trajectory comparisons'),
    'operational_representatives': ('operational_representatives', 'Figure 9: four operational trajectory groups'),
    'spread_error_relationship': ('spread_error_relationship', 'Figure 10: ensemble spread versus error and interval width'),
}


RESULT_FIGURES = ('operational_controls', 'stochastic_representative',
                  'stochastic_uq_performance', 'operational_representatives',
                  'spread_error_relationship')


def selected_figures(value):
    requested = [part for item in value for part in item.split(',') if part]
    if requested == ['all']:
        # Keep the historical Methodology/report default; Results figures are opt-in.
        return [name for name in REGISTRY if name not in RESULT_FIGURES]
    if requested == ['results']:
        return list(RESULT_FIGURES)
    if (not requested or 'all' in requested or 'results' in requested
            or set(requested)-set(REGISTRY)):
        raise ValueError(f'Choose all, results or specific IDs: {", ".join(REGISTRY)}')
    return list(dict.fromkeys(requested))


def run(config_file, *, figures=('all',), output_dir=None, overwrite=False,
        formats=None, dpi=None, sampling_mode=None, dataset_mode=None,
        profile_id=None, calibration_source=None, trajectory_mode=None,
        representative_count=None, representative_seed=None, repo_root=None):
    root = Path(repo_root or ROOT).resolve()
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute(): cfg_path = root/cfg_path
    config = json.loads(cfg_path.read_text(encoding='utf-8'))
    selected = selected_figures(figures)
    if sampling_mode: config['sampling']['mode'] = sampling_mode
    if dataset_mode: config['dataset']['mode'] = dataset_mode
    if profile_id: config['operational']['representative_profile_id'] = profile_id
    if calibration_source: config['operational']['representative_calibration'] = calibration_source
    if trajectory_mode: config['operational']['representative_trajectory_mode'] = trajectory_mode
    if representative_count is not None:
        if representative_count < 1: raise ValueError('--representative-count must be positive')
        config['results']['representative_count'] = representative_count
    if representative_seed is not None:
        if representative_seed < 0: raise ValueError('--representative-seed must be nonnegative')
        config['results']['representative_seed'] = representative_seed
    options = dict(formats=formats or config['formats'], dpi=dpi or config['dpi'])
    if not options['formats'] or set(options['formats'])-{'png', 'pdf', 'svg'} or options['dpi'] < 72:
        raise ValueError('Formats must be PNG/PDF/SVG; dpi must be >=72')
    output_root = Path(os.environ.get('RABL_OUTPUT_ROOT') or root/'outputs').expanduser().resolve()
    def path(raw):
        p = Path(raw).expanduser()
        if p.is_absolute(): return p.resolve()
        # Honor the repository's external output-root convention for outputs/... paths.
        if p.parts and p.parts[0] == 'outputs': return output_root.joinpath(*p.parts[1:])
        return (root/p).resolve()
    destination = path(output_dir or config['output_dir'])
    # Never replace source inputs through an ill-chosen output destination.
    sources = [path(config['assets_dir']), path(config['sampling']['profiles_dir']),
               path(config['dataset']['simulation_dir']), path(config['operational']['results_dir'])]
    if 'results' in config:
        sources += [path(config['results'][key]) for key in (
            'stochastic_forecasts_dir', 'stochastic_floor_dir',
            'stochastic_validation_dir', 'operational_dir', 'scaler_file')]
        sources.append(path(config['results']['results_tables_output_dir']))
    if destination == root or any(destination == s or destination in s.parents or s in destination.parents for s in sources):
        raise ValueError('Paper output directory must be separate from all source directories')
    for key in selected:
        target = destination/key
        if target.exists() and not overwrite:
            raise FileExistsError(f'{target} exists; use --overwrite to replace only selected figure folders')
    import _mace_paper_figures as renderers
    destination.mkdir(parents=True, exist_ok=True)
    index_file = destination/'figure_index.json'
    index = json.loads(index_file.read_text()) if index_file.exists() else {'schema_version': 1, 'figures': {}}
    statuses = {}
    for key in selected:
        target = destination/key
        print(f'Paper figure: {key}', flush=True)
        with tempfile.TemporaryDirectory(prefix=f'.{key}-', dir=destination) as temp:
            stage = Path(temp)
            try:
                result = getattr(renderers, REGISTRY[key][0])(config, stage, path, options)
                # Exact input hashes, including saved forecast sources, must remain unchanged.
                for record in result.get('inputs', []):
                    if renderers.fingerprint(record['path'])['sha256'] != record['sha256']:
                        raise ValueError(f"Input changed while rendering: {record['path']}")
                (stage/'caption.md').write_text(result['caption']+'\n', encoding='utf-8')
                result['files'].append('caption.md')
                source_files = [Path(__file__), Path(renderers.__file__)]
                if key in RESULT_FIGURES and key != 'operational_controls':
                    import _mace_results_figures
                    source_files.append(Path(_mace_results_figures.__file__))
                result.update(completed=True, figure_id=key, options=options,
                              effective_config=config, configuration_sha256=renderers.fingerprint(cfg_path)['sha256'],
                              source_sha256={p.name:renderers.fingerprint(p)['sha256'] for p in source_files},
                              inputs_unchanged=True)
                result['output_hashes'] = [dict(file=f, sha256=renderers.fingerprint(stage/f)['sha256'])
                                           for f in result['files']]
                renderers.write_json(stage/'figure_metadata.json', result)
                # Replace only after successful validation; failed retries retain old output.
                if target.exists(): shutil.rmtree(target)
                shutil.copytree(stage, target)
                statuses[key] = {'completed': True, 'directory': str(target), 'files': result['files']}
                index['figures'][key] = statuses[key]
                print(f'Completed {key}: {target}', flush=True)
            except (OSError, ValueError, ImportError, KeyError, TypeError) as error:
                statuses[key] = {'completed': False, 'error': str(error)}
                print(f'FAILED {key}: {error}', file=sys.stderr, flush=True)
    index['last_run'] = {'requested': selected, 'statuses': statuses,
                         'completed': all(s['completed'] for s in statuses.values())}
    renderers.write_json(index_file, index)
    return index['last_run']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('scripts/mace_paper_figures_config.json'))
    p.add_argument('--figures', nargs='+', default=['all'], help='all (legacy), results, space-separated IDs, or comma-separated IDs')
    p.add_argument('--list', action='store_true', help='List available IDs without loading data or plotting dependencies')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--overwrite', action='store_true', help='Replace successful selected figure folders only')
    p.add_argument('--formats', nargs='+', choices=('png', 'pdf', 'svg'))
    p.add_argument('--dpi', type=int)
    p.add_argument('--sampling-mode', choices=('auto', 'stored', 'illustrative'))
    p.add_argument('--dataset-mode', choices=('auto', 'raw', 'stored_image'))
    p.add_argument('--profile-id', help='Saved operation-informed test ID for representative_forecast')
    p.add_argument('--representative-count', type=int, help='Number of stochastic profiles and four-group operational figure sets (default: 10)')
    p.add_argument('--representative-seed', type=int, help='Seed for reproducible representative sampling')
    p.add_argument('--calibration-source', choices=('original', 'operational'),
                   help='Calibration population for the representative test forecast')
    p.add_argument('--trajectory-mode', choices=('target_trajectory', 'joint_trajectory'),
                   help='Trajectory conformal comparison for representative_forecast')
    args = p.parse_args()
    if args.list:
        for key, (_, description) in REGISTRY.items(): print(f'{key}: {description}')
        return
    result = run(args.config, figures=args.figures, output_dir=args.output_dir,
                 overwrite=args.overwrite, formats=args.formats, dpi=args.dpi,
                 sampling_mode=args.sampling_mode, dataset_mode=args.dataset_mode,
                 profile_id=args.profile_id, calibration_source=args.calibration_source,
                 trajectory_mode=args.trajectory_mode,
                 representative_count=args.representative_count,
                 representative_seed=args.representative_seed)
    if not result['completed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
