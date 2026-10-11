"""Read-only 4x4 operational forecast grids from completed saved results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from _mace_floor_study import bounds, fingerprint, write_json
from _mace_scaler_audit import compare_arrays, decode_names, read_scaler

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('Tf', 'Tm', 'Thp', 'TN2', 'Tsg', 'T_steam_out', 'x_steam_out',
         'c[1]', 'c[2]', 'c[3]', 'c[4]', 'c[5]', 'c[6]', 'n', 'rho_dollars')
LABELS = (r'$T_f$ [K]', r'$T_m$ [K]', r'$T_{hp}$ [K]', r'$T_{N_2}$ [K]',
          r'$T_{sg}$ [K]', r'$T_{\mathrm{steam,out}}$ [K]', r'$x_{\mathrm{steam,out}}$ [-]',
          *[rf'$c_{i}$' for i in range(1, 7)], r'$n$', r'$\rho$ [dollars]')
TIME = np.round(np.linspace(0, 200, 501), 8)
FEEDBACK = 'native-Y-to-X-affine-float32-v1'


def resolve(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def select_profiles(manifest, count=3):
    """First sorted test IDs per family; no residual/coverage-based selection."""
    if count < 1:
        raise ValueError('profiles-per-family must be positive')
    selected = []
    for family in 'BCDF':
        candidates = sorted((p for p in manifest['profiles']
                             if p['split'] == 'test' and p['family'] == family), key=lambda p: p['id'])
        if len(candidates) < count:
            raise ValueError(f'Family {family} has fewer than {count} test profiles')
        selected.extend(candidates[:count])
    return selected


def exact(actual, reference, label):
    report = compare_arrays(actual, reference)
    if not report['match']:
        raise ValueError(f'Saved array mismatch: {label}: {report}')
    return dict(field=label, **report)


def load_profile(forecasts, intervals, quantiles, *, profile, index, names, stats,
                 sources, modes, floor):
    g = forecasts[profile['id']]
    if g.attrs.get('family') != profile['family'] or g.attrs.get('split') != 'test':
        raise ValueError('Forecast family/split disagrees with manifest')
    mean, spread, truth = (g[k][()] for k in ('mean', 'spread', 'truth'))
    members = g['member_predictions_scaled'][()]
    if (members.shape != (10, 501, 15) or members.dtype != np.float32
            or any(a.shape != (501, 15) or a.dtype != np.float32 for a in (mean, spread, truth))):
        raise ValueError('Expected native float32 ten-member, 501-step, 15-target forecasts')
    checks = [exact(mean, members.mean(axis=0), 'mean'),
              exact(spread, members.std(axis=0, ddof=0), 'spread'),
              exact(g['time_seconds'][()], TIME, 'time_seconds')]
    physical_truth = g['truth_physical'][()]
    if physical_truth.shape != mean.shape or not np.all(np.isfinite(physical_truth)):
        raise ValueError('Invalid native physical truth')
    checks.append(exact(truth, (physical_truth-stats['y']['min'])/stats['y']['span'], 'scaled_truth'))
    angle = g['control_physical'][()]
    if angle.shape != (501,) or not np.all(np.isfinite(angle)):
        raise ValueError('Invalid physical drum-angle input')
    if not np.all(np.isfinite(members)) or np.any(spread < 0):
        raise ValueError('Invalid ensemble values')
    minimum, span = stats['y']['min'].astype(float), stats['y']['span'].astype(float)
    physical = lambda a: a.astype(float)*span+minimum
    bands = {}
    for source in sources:
        bands[source] = {}
        for mode in ('horizon', *modes):
            path = f'intervals/{mode}/{source}/mace'
            lower, upper, mask = (intervals[path+'/'+k][index] for k in ('lower', 'upper', 'mask'))
            if lower.dtype != np.float32 or upper.dtype != np.float32 or mask.dtype != np.bool_:
                raise ValueError('Expected saved float32 endpoints and boolean coverage mask')
            q = quantiles[f'{source}/mace/{mode}'][()]
            expected = bounds(dict(mean=mean, spread=spread), q, floor)
            checks.extend([exact(lower, expected[0], source+'/'+mode+'/lower'),
                           exact(upper, expected[1], source+'/'+mode+'/upper'),
                           exact(mask, (lower <= truth) & (truth <= upper), source+'/'+mode+'/mask')])
            if lower.shape != mean.shape or upper.shape != mean.shape or np.any(lower > upper):
                raise ValueError('Invalid interval shape/order')
            # Plot saved endpoints, not reconstructed bounds or physically clipped bands.
            bands[source][mode] = (physical(lower), physical(upper))
    return dict(time=TIME, angle=angle, truth=physical_truth, mean=physical(mean),
                raw=(physical(mean-2*spread), physical(mean+2*spread)), bands=bands,
                names=names, checks=checks)


def render_grid(data, *, profile, source, mode, alpha):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    indices = [data['names'].index(name) for name in ORDER]
    label = 'MACE target trajectory' if mode == 'target_trajectory' else 'MACE joint trajectory'
    colors = ('#9b72cf', '#2b9a39', '#df2828')
    with plt.rc_context({'font.family': 'DejaVu Serif', 'font.size': 12, 'mathtext.fontset': 'dejavuserif'}):
        fig, axes = plt.subplots(4, 4, figsize=(18, 11), sharex=True)
        time = data['time']
        axes.flat[0].plot(time, data['angle'], color='black', lw=1.8)
        axes.flat[0].set_ylabel(r'$\theta$ [deg]')
        for ax, j, ylabel in zip(list(axes.flat)[1:], indices, LABELS):
            for band, color in zip((data['raw'], data['bands'][source]['horizon'],
                                    data['bands'][source][mode]), colors):
                ax.fill_between(time, band[0][:, j], band[1][:, j], color=color, alpha=.10, lw=0)
                ax.plot(time, band[0][:, j], color=color, lw=.85)
                ax.plot(time, band[1][:, j], color=color, lw=.85)
            ax.plot(time, data['truth'][:, j], color='black', lw=1.8)
            ax.plot(time, data['mean'][:, j], color='#075cbb', lw=1.8)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        for ax in axes.flat:
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 100, 200])
            ax.grid(alpha=.22)
        handles = [Line2D([], [], color='black', lw=2, label='Truth'),
                   Line2D([], [], color='#075cbb', lw=2, label='Ensemble mean'),
                   Patch(facecolor=colors[0], alpha=.18, label=r'Raw ensemble $\pm2\sigma$'),
                   Patch(facecolor=colors[1], alpha=.18, label='MACE horizon'),
                   Patch(facecolor=colors[2], alpha=.18, label=label)]
        source_label = 'original calibration (transfer diagnostic)' if source == 'original' else 'operational calibration'
        fig.suptitle(f"Family {profile['family']} | {profile['id']} | {source_label} | "
                     f'{100*(1-alpha):g}% nominal objective', fontsize=15, y=.995)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .968), ncol=5, frameon=False)
        fig.supxlabel('Forecast time [s]', y=.015)
        fig.tight_layout(rect=(0, .04, 1, .95))
    return fig


def run(results_dir, *, output_dir=None, scaler_file=None, profiles_per_family=3,
        calibration='both', trajectory_mode='both', dpi=200, formats=('png', 'pdf')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if dpi < 30 or not formats or set(formats)-{'png', 'pdf'}:
        raise ValueError('Require dpi >=30 and PNG/PDF formats')
    if calibration not in ('both', 'original', 'operational') or trajectory_mode not in ('both', 'target_trajectory', 'joint_trajectory'):
        raise ValueError('Unknown calibration or trajectory mode')
    folder = resolve(results_dir)
    out = resolve(output_dir) if output_dir else folder/'forecast_plots_4x4'
    if out.exists():
        raise FileExistsError(f'Plot directory exists; use --output-dir for a new attempt: {out}')
    files = dict(manifest=folder/'operational_manifest.json', forecast_metadata=folder/'forecasts/forecast_metadata.json',
                 evaluation_metadata=folder/'evaluation/evaluation_metadata.json',
                 forecasts=folder/'forecasts/operational_test.h5',
                 intervals=folder/'evaluation/intervals_and_statistics.h5',
                 quantiles=folder/'evaluation/calibration_quantiles.h5')
    manifest = json.loads(files['manifest'].read_text())
    fm, em = (json.loads(files[k].read_text()) for k in ('forecast_metadata', 'evaluation_metadata'))
    identity = hashlib.sha256(json.dumps(manifest['profiles'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not fm.get('completed') or not em.get('completed') or fm.get('feedback') != FEEDBACK:
        raise ValueError('Requires completed corrected forecasts and evaluation')
    if any(m.get('manifest_identity') != identity for m in (fm, em)) or manifest['identity'] != identity:
        raise ValueError('Manifest identity mismatch')
    names = fm['target_names']
    if len(names) != 15 or set(names) != set(ORDER) or names != em['target_names']:
        raise ValueError('Expected the 15 reactor targets in matching stored order')
    selected = select_profiles(manifest, profiles_per_family)
    ids = [p['id'] for p in manifest['profiles'] if p['split'] == 'test']
    if len(set(ids)) != len(ids) or ids != em['test_profile_names']:
        raise ValueError('Test profile identity/order mismatch')
    scaler_record = fm['input_hashes']['external_scaled_h5_path']
    files['scaler'] = resolve(scaler_file or scaler_record['path'])
    print('Verifying saved result/scaler hashes (large files may take a minute)', flush=True)
    hashes = {k: fingerprint(p) for k, p in files.items()}
    for record in (fm['output_hashes']['operational_test.h5'], em['forecast_hashes']['operational_test.h5']):
        if hashes['forecasts']['sha256'] != record['sha256']:
            raise ValueError('Operational test forecast changed since evaluation')
    if hashes['scaler']['sha256'] != scaler_record['sha256'] or hashes['scaler']['sha256'] != em['input_hashes']['external_scaled_h5_path']['sha256']:
        raise ValueError('Scaler file differs from the evaluated frozen scaler')
    stats, sm = read_scaler(files['scaler'])
    if sm['target_names'] != names or len(stats['y']['span']) != 15:
        raise ValueError('Scaler target order/shape mismatch')
    sources = ('original', 'operational') if calibration == 'both' else (calibration,)
    modes = ('target_trajectory', 'joint_trajectory') if trajectory_mode == 'both' else (trajectory_mode,)
    with h5py.File(files['forecasts']) as f, h5py.File(files['intervals']) as b, h5py.File(files['quantiles']) as q:
        if (not f.attrs.get('completed') or f.attrs.get('feedback') != FEEDBACK
                or f.attrs.get('manifest_identity') != identity or int(f.attrs.get('member_count', 0)) != 10
                or int(f.attrs.get('ensemble_std_ddof', -1)) != 0 or set(f.keys()) != set(ids)):
            raise ValueError('Saved forecast provenance mismatch')
        if any(decode_names(g.attrs.get('target_names', [])) != names for g in (f, b, q)):
            raise ValueError('Saved target order mismatch')
        if decode_names(b.attrs.get('test_profile_names', [])) != ids:
            raise ValueError('Saved interval profile order mismatch')
        if b.attrs.get('coverage_convention') != em['coverage_convention']:
            raise ValueError('Coverage convention mismatch')
        if float(q.attrs['alpha']) != em['alpha'] or float(q.attrs['common_floor']) != em['common_floor']:
            raise ValueError('Saved quantile objective/floor mismatch')
        # Validate every selected array before creating any plot output.
        loaded = [(p, load_profile(f, b, q, profile=p, index=ids.index(p['id']), names=names,
                                  stats=stats, sources=sources, modes=modes, floor=em['common_floor'])) for p in selected]
    out.mkdir(parents=True)
    report = dict(completed=False, selection='First sorted test profile IDs per family; independent of errors/coverage',
                  selected_profiles=selected, calibration_sources=sources, trajectory_modes=modes,
                  input_hashes=hashes, plotting_source_sha256=fingerprint(Path(__file__))['sha256'],
                  alpha=em['alpha'], common_floor=em['common_floor'], target_order=ORDER,
                  array_checks=[dict(profile=p['id'], **c) for p, d in loaded for c in d['checks']], figures=[])
    write_json(out/'plot_metadata.json', report)
    for profile, data in loaded:
        for source in sources:
            for mode in modes:
                dest = out/source/mode/profile['family']
                dest.mkdir(parents=True, exist_ok=True)
                fig = render_grid(data, profile=profile, source=source, mode=mode, alpha=em['alpha'])
                try:
                    for fmt in formats:
                        path = dest/(profile['id']+'.'+fmt)
                        fig.savefig(path, dpi=dpi, bbox_inches='tight')
                        report['figures'].append(str(path.relative_to(out)))
                finally:
                    plt.close(fig)
                print(f"Rendered {source}/{mode}/{profile['family']}/{profile['id']}", flush=True)
    for k, path in files.items():
        if fingerprint(path)['sha256'] != hashes[k]['sha256']:
            raise ValueError(f'Input changed during plotting: {path}')
    report.update(completed=True, input_files_unchanged=True,
                  grid_count=len(selected)*len(sources)*len(modes))
    write_json(out/'plot_metadata.json', report)
    (out/'README.md').write_text(
        '# Operational forecast grids\n\n'
        'Each 4x4 grid has drum angle and 15 targets in physical units. Black: native simulator truth; '
        'blue: descaled ensemble mean; purple: raw ensemble +/-2 standard deviations (not calibrated); '
        'green: saved MACE horizon bounds; red: saved MACE target-trajectory or joint-trajectory bounds, '
        'as labeled. Original and operational calibration are in separate folders, as are the two trajectory objectives. '
        'The same test profiles and mean forecasts are used in every folder. No bounds are clipped. '
        'Endpoints and coverage masks are checked exactly at stored precision against the saved quantiles.\n\n'
        'First sorted test IDs per family are selected without consulting errors or coverage. These examples '
        'illustrate forecasts, not aggregate performance. Original calibration is a nonexchangeable transfer '
        'diagnostic. Nominal objectives refer to the pooled mixture; target trajectory covers each target over '
        'time, while joint trajectory covers all targets over time. Raw ensemble spread has no nominal guarantee. '
        'The separate target and joint bands must not be confused. No simulations, inference, fitting or calibration '
        'are performed. Inspect completed in plot_metadata.json before sharing.\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path('outputs/ml_results/mace_operational_validation_v1'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--scaler-file', type=Path, help='Relocated frozen external scaled HDF5; bytes must match recorded SHA-256')
    p.add_argument('--profiles-per-family', type=int, default=3)
    p.add_argument('--calibration', choices=('both', 'original', 'operational'), default='both')
    p.add_argument('--trajectory-mode', choices=('both', 'target_trajectory', 'joint_trajectory'), default='both')
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--formats', nargs='+', choices=('png', 'pdf'), default=['png', 'pdf'])
    args = p.parse_args()
    report = run(**vars(args))
    print(f"Completed {report['grid_count']} forecast grids", flush=True)


if __name__ == '__main__':
    main()
