"""Scientific panels and preserved assets for the MACE manuscript.

No simulation, model inference, scaler fitting, or conformal calibration occurs.
Add future data figures as renderers returning output and provenance records.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
STYLE = {'font.family': 'DejaVu Serif', 'font.size': 8,
         'mathtext.fontset': 'dejavuserif', 'axes.labelsize': 8,
         'legend.fontsize': 7, 'axes.titlesize': 9,
         'pdf.fonttype': 42, 'svg.fonttype': 'none'}
DISPLAY = ('drumAngleDeg', 'TN2', 'Tm', 'Thp', 'Tf', 'Tsg', 'T_steam_out',
           'x_steam_out', 'c[1]', 'c[2]', 'c[3]', 'c[4]', 'c[5]', 'c[6]',
           'P_MW', 'rho_dollars')
LABELS = (r'$\theta$ (degree)', r'$T_{N2}$ (K)', r'$T_m$ (K)', r'$T_{hp}$ (K)',
          r'$T_f$ (K)', r'$T_{sg}$ (K)', r'$T_{\mathrm{steam,out}}$ (K)',
          r'$x_{\mathrm{steam,out}}$ (–)',
          *[rf'$c_{i}$ (normalized)' for i in range(1, 7)],
          r'$P$ (MW)', r'$\rho_{\$}$ (dollars)')
ASSETS = ('hpmr_bop_viz.png', 'lstm_architecture.png',
          'neural_autoregression.png', 'rolling_forecast.png')


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': h.hexdigest()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_figure(fig, out, stem, options):
    """Keep exact panel dimensions; captions are separate Markdown files."""
    files = []
    try:
        for suffix in options['formats']:
            dest = out / f'{stem}.{suffix}'
            fig.savefig(dest, dpi=options['dpi'], facecolor='white')
            files.append(dest.name)
    finally:
        plt.close(fig)
    return files


def panel(ax, letter, title=None):
    ax.set_title(f'({letter})' + (f' {title}' if title else ''), loc='left')
    ax.grid(alpha=.18)


def csv_columns(path, names):
    """Read the saved decimal values without accepting missing/invalid data."""
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [s.strip() for s in (reader.fieldnames or [])]
        missing = set(names) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = [[float(row[n]) if row[n].strip() else np.nan for n in names]
                for row in reader]
    a = np.asarray(rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        raise ValueError(f'{path}: fewer than two rows')
    return a


def sampling_covariance(settings):
    """Use the repository GP covariance; compute variance in O(L^2), not BKB^T."""
    from rabl.variography.DrumVariography import DrumProfileGenerator
    intervals = int(settings['intervals'])
    if intervals < 2 or settings['kernel'] != 'matern52':
        raise ValueError('Sampling figure requires Matérn-5/2 and >=2 intervals')
    if settings['duration_seconds'] != 200:
        raise ValueError('Paper controls must retain the 0–200 s time domain')
    for key in ('duration_seconds', 'ell_seconds', 'design_terminal_sd_degrees'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'Invalid sampling {key}')
    time = np.linspace(0, settings['duration_seconds'], intervals + 1)
    gen = DrumProfileGenerator(kernel=settings['kernel'], ell=settings['ell_seconds'],
                               nugget_v_deg2_s2=settings['nugget_velocity'],
                               jitter_frac=settings['jitter_fraction'],
                               cond_jitter=settings['conditioning_jitter'])
    gen.solve_params_for_sigma_theta(time, settings['design_terminal_sd_degrees'],
                                    ell=settings['ell_seconds'],
                                    nugget=settings['nugget_velocity'], update_instance=True)
    cov = gen.build_velocity_cov(time[:-1])
    dt = time[1] - time[0]
    unconditioned = np.r_[0., np.cumsum(np.cumsum(cov, axis=0), axis=1).diagonal()] * dt**2
    # Actual generator transform: v-a*v[0], then force v[0]=0 exactly.
    tail = cov[1:, 1:]
    sums = np.cumsum(np.cumsum(tail, axis=0), axis=1).diagonal()
    cross = np.cumsum(cov[1:, 0])
    a = cross / (cov[0, 0] + gen.cond_jitter)
    conditioned = np.r_[0., 0., sums - 2*a*cross + a*a*cov[0, 0]] * dt**2
    if np.any(conditioned < -1e-12):
        raise ValueError('Negative conditioned variance')
    return gen, time, cov, np.sqrt(unconditioned), np.sqrt(np.maximum(conditioned, 0))


def illustrative_samples(gen, time, cov, settings):
    """Explicitly illustrative draws; Cholesky avoids repeated large SVDs."""
    from scipy.linalg import cholesky
    rng = np.random.default_rng(settings['illustration_seed'])
    factor = cholesky(cov, lower=True, check_finite=True)
    samples = []
    for _ in range(max(100, settings['count'] * 100)):
        v = factor @ rng.standard_normal(len(cov))
        v -= cov[:, 0] / (cov[0, 0] + gen.cond_jitter) * v[0]
        v[0] = 0.
        angle = np.r_[settings['baseline_degrees'],
                      settings['baseline_degrees'] + np.cumsum(v) * (time[1] - time[0])]
        if np.all((angle >= 0) & (angle <= 180)):
            samples.append((f'illustration_{len(samples)+1:02d}', time, angle, v))
        if len(samples) == settings['count']:
            return samples
    raise ValueError('Unable to draw enough admissible illustrative controls')


def sampling_figure(config, out, path, options):
    settings = config['sampling']
    if not isinstance(settings['count'], int) or settings['count'] < 1:
        raise ValueError('sampling.count must be positive')
    directory = path(settings['profiles_dir'])
    candidates = sorted(directory.glob('drum_profile_*.csv'))
    mode = settings['mode']
    if mode not in ('auto', 'stored', 'illustrative'):
        raise ValueError('sampling.mode must be auto/stored/illustrative')
    if mode == 'stored' and not candidates:
        raise FileNotFoundError(f'No saved control tables in {directory}')
    gen, time, cov, uncond, cond = sampling_covariance(settings)
    inputs, samples = [fingerprint(inspect.getfile(type(gen)))], []
    stored = mode != 'illustrative' and bool(candidates)
    if stored:
        if len(candidates) < settings['count']:
            raise ValueError('Fewer stored control trajectories than requested')
        for file in candidates[:settings['count']]:
            a = csv_columns(file, ('Time(s)', 'Drum_Angle(deg)', 'Drum_Velocity(deg/s)'))
            if not np.array_equal(a[:, 0], time):
                delta = float(np.max(np.abs(a[:, 0] - time))) if len(a) == len(time) else None
                raise ValueError(f'{file}: stored float64 control time grid differs exactly; max difference={delta}')
            if not np.isfinite(a[:, :2]).all() or not np.isfinite(a[:-1, 2]).all():
                raise ValueError(f'{file}: invalid angle/interval velocity')
            if a[0, 2] != 0 or a[0, 1] != settings['baseline_degrees']:
                raise ValueError(f'{file}: initial angle/velocity differs')
            increments = a[:-1, 2] * np.diff(a[:, 0])
            reconstructed = np.r_[a[0, 1], a[0, 1] + np.cumsum(increments)]
            difference = float(np.max(np.abs(reconstructed - a[:, 1])))
            # This compares different arithmetic (BLAS dot vs cumulative sum),
            # not two stored arrays. Use a summation error bound, not a broad tolerance.
            eps = np.finfo(np.float64).eps
            operations = 2*len(increments)+2
            gamma = operations*eps/(1-operations*eps)
            error_bound = gamma*(abs(a[0, 1])+float(np.sum(np.abs(increments))))
            if difference > error_bound:
                raise ValueError(f'{file}: integration differs by {difference}, exceeding floating summation bound {error_bound}')
            rec = fingerprint(file)
            rec.update(max_time_grid_difference=float(np.max(np.abs(a[:, 0]-time))),
                       max_integration_reconstruction_difference=difference,
                       integration_floating_summation_error_bound=error_bound)
            inputs.append(rec)
            samples.append((file.stem, a[:, 0], a[:, 1], a[:-1, 2]))
    else:
        print('sampling: using explicitly labeled illustrative GP draws; not training samples', flush=True)
        samples = illustrative_samples(gen, time, cov, settings)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(178*MM, 125*MM), layout='constrained')
        grid = fig.add_gridspec(2, 2)
        ax_corr = fig.add_subplot(grid[0, 0])
        ax_velocity = fig.add_subplot(grid[0, 1])
        ax_angle = fig.add_subplot(grid[1, :])
        lag = np.linspace(0, 60, 601)
        for ell, color, style in [(5, '.55', '--'), (10, 'black', '-'), (20, '.35', ':')]:
            r = np.sqrt(5)*lag/ell
            ax_corr.plot(lag, (1+r+r*r/3)*np.exp(-r), color=color,
                         ls=style, label=rf'$\ell={ell}$ s')
        ax_corr.set(xlim=(0, 60), ylim=(0, 1.03), xlabel='Lag (s)', ylabel='Velocity correlation')
        ax_corr.legend(frameon=False)
        colors = plt.get_cmap('tab10').colors
        for i, (name, t, angle, velocity) in enumerate(samples):
            ax_velocity.plot(t[:-1], velocity, lw=.65, color=colors[i % 10])
            ax_angle.plot(t, angle, lw=.65, color=colors[i % 10])
        vmax = max(float(np.max(np.abs(v))) for _, _, _, v in samples)
        ax_velocity.set(xlim=(0, 200), ylim=(-max(vmax*1.1, .001), max(vmax*1.1, .001)),
                        xlabel='Time (s)', ylabel='Velocity (degree/s)')
        ax_velocity.axhline(0, color='.6', lw=.5)
        ax_angle.set(xlim=(0, 200), xlabel='Time (s)', ylabel='Control angle (degree)')
        ax_angle.axhline(settings['baseline_degrees'], color='.6', ls='--', lw=.6)
        titles = ('Velocity correlation', 'Sampled velocity trajectories',
                  'Integrated Control Trajectories')
        for ax, letter, title in zip((ax_corr, ax_velocity, ax_angle), 'abc', titles):
            panel(ax, letter, title)
        files = save_figure(fig, out, 'fig01_control_trajectory_sampling', options)
    np.savez_compressed(out/'sampling_data.npz', time=time, unconditioned_sd=uncond,
                        conditioned_sd=cond, angle=np.stack([s[2] for s in samples]),
                        velocity=np.stack([s[3] for s in samples]))
    files.append('sampling_data.npz')
    caption = ('Construction of control trajectories from Matérn-5/2 interval velocity. '
               f"Panels (b,c) show {'stored batch controls' if stored else 'illustrative draws, not training samples'}. "
               'The first velocity is conditioned to zero. Matching colors pair velocities and '
               'integrated control angles. The sidecar records selected IDs and sampling settings.')
    return dict(files=files, inputs=inputs, caption=caption, sample_origin='stored' if stored else 'illustrative',
                selected_ids=[s[0] for s in samples], settings=settings,
                conditioned_terminal_sd_degrees=float(cond[-1]),
                unconditioned_terminal_sd_degrees=float(uncond[-1]),
                covariance_source='rabl.variography.DrumVariography.DrumProfileGenerator')


def image_export(source, out, stem, options):
    """Preserve the original PNG bytes; vector wrappers do not add resolution."""
    from PIL import Image
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    dest = out/(stem+source.suffix.lower())
    shutil.copyfile(source, dest)
    files = [dest.name]
    with Image.open(source) as im:
        size = im.size
        rgba = np.asarray(im.convert('RGBA'))
    if set(options['formats']) & {'pdf', 'svg'}:
        fig = plt.figure(figsize=(178*MM, 178*MM*size[1]/size[0]))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgba)
        ax.axis('off')
        vector = dict(options, formats=[f for f in options['formats'] if f != 'png'])
        files.extend(save_figure(fig, out, stem, vector))
    return files, size


def dataset_figure(config, out, path, options):
    settings = config['dataset']
    directory = path(settings['simulation_dir'])
    candidates = sorted(directory.glob('results_drum_profile_*.csv'))
    mode = settings['mode']
    if mode not in ('auto', 'stored_image', 'raw'):
        raise ValueError('dataset.mode must be auto/stored_image/raw')
    if mode == 'raw' and not candidates:
        raise FileNotFoundError(f'No simulation CSVs in {directory}')
    if mode == 'stored_image' or (mode == 'auto' and not candidates):
        source = path(settings['existing_image'])
        files, size = image_export(source, out, 'fig05_generated_transient_dataset', options)
        return dict(files=files, inputs=[fingerprint(source)], image_pixels=size,
                    source_kind='preserved supplied image', count_verified_from_raw=False,
                    caption='Preserved supplied 4×4 generated-transient dataset figure (batch 0003). '
                    'Control angle plus 15 outputs; P in MW is displayed instead of forecast output n. '
                    'This is a raster original, not a new simulation or a vector reconstruction. '
                    'The original trace count has not been independently verified from this image.')
    count = settings.get('expected_profiles')
    if count is not None and len(candidates) != count:
        raise ValueError(f'Expected {count} batch simulation CSVs, found {len(candidates)}; refusing to label an incomplete batch as all profiles')
    inputs = []
    from matplotlib.collections import LineCollection
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 4, figsize=(178*MM, 135*MM), sharex=True, layout='constrained')
        try:
            # Stream finite chunks; rasterize dense paths within otherwise vector output.
            for start in range(0, len(candidates), 100):
                chunks = [[] for _ in DISPLAY]
                for file in candidates[start:start+100]:
                    a = csv_columns(file, ('t', *DISPLAY))
                    if not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) <= 0):
                        raise ValueError(f'{file}: invalid simulator trajectory')
                    if a[0, 0] != 0 or a[-1, 0] != 200:
                        raise ValueError(f'{file}: expected 0–200 s')
                    for j in range(len(DISPLAY)):
                        chunks[j].append(a[:, [0, j+1]])
                    inputs.append(fingerprint(file))
                for j, ax in enumerate(axes.flat):
                    color = 'black' if j in (0, 15) else '#e377c2' if j == 14 else '#006400' if 8 <= j < 14 else '#d62728'
                    lines = LineCollection(chunks[j], colors=color, linewidths=.25, alpha=.12, rasterized=True)
                    ax.add_collection(lines, autolim=True)
                    ax.autoscale_view()
                print(f'dataset: read {min(start+100, len(candidates))}/{len(candidates)} trajectories', flush=True)
            for ax, label in zip(axes.flat, LABELS):
                ax.set_title(label, fontsize=8)
                ax.set_xlim(0, 200)
                ax.set_xticks([0, 100, 200])
                ax.grid(alpha=.15)
                ax.tick_params(labelsize=6)
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            for ax in axes[-1]:
                ax.set_xlabel('Time (s)')
            files = save_figure(fig, out, 'fig05_generated_transient_dataset', options)
        finally:
            plt.close(fig)
    return dict(files=files, inputs=inputs, count_verified_from_raw=True,
                source_kind='raw simulator CSVs', profile_count=len(candidates), display_order=DISPLAY,
                caption=f'Generated multivariate transients: all {len(candidates)} trajectories in {directory.name}. '
                'The first panel is prescribed control, not an output variable. Physical power P (MW) is displayed '
                'instead of normalized forecast output n; display order differs from native model storage. '
                'All lines are simulator responses, not forecasts or uncertainty bounds.')


def operational_controls(config, out, path, options):
    settings = config['operational']
    file = path(settings['results_dir'])/'operational_manifest.json'
    manifest = json.loads(file.read_text(encoding='utf-8'))
    if manifest.get('recipe') != 'BCDF-independent-parameter-draws-v1':
        raise ValueError('Only the retained B/C/D/F recipe is allowed; broader-angle study is excluded')
    identity = hashlib.sha256(json.dumps(manifest['profiles'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if manifest.get('identity') != identity:
        raise ValueError('Operational manifest identity differs from its profile list')
    from _mace_operational_profiles import TIME, control
    count = settings['profiles_per_family']
    if not isinstance(count, int) or not 1 <= count <= 3:
        raise ValueError('Control panels support 1–3 profiles per family')
    selected = []
    for family in 'BCDF':
        group = sorted((p for p in manifest['profiles'] if p['split'] == 'test' and p['family'] == family), key=lambda p:p['id'])
        if len(group) < count:
            raise ValueError(f'Family {family} has fewer than {count} evaluation profiles')
        selected.extend(group[:count])
    labels = ('Single maneuver with hold', 'Successive maneuvers with holds',
              'Maneuver with return', 'Bidirectional maneuvers')
    traces = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(178*MM, 110*MM), sharex=True, sharey=True, layout='constrained')
        for ax, family, label, letter in zip(axes.flat, 'BCDF', labels, 'bcde'):
            for p, color, ls in zip([p for p in selected if p['family'] == family],
                                    ['black', '#0072B2', '#D55E00'], ['-', '--', ':']):
                angle, velocity = control(p['moves'])
                if not np.isfinite(angle).all() or angle.min() < 44.25 or angle.max() > 48.75:
                    raise ValueError(f"{p['id']}: outside retained recipe envelope")
                ax.plot(TIME, angle, color=color, ls=ls, lw=1, label=p['id'].rsplit('_', 1)[-1])
                traces.append(angle)
            ax.axhline(45, color='.6', ls='--', lw=.5)
            ax.set(xlim=(0, 200), ylim=(44, 49), xticks=[0, 50, 100, 150, 200], yticks=[44, 45, 46, 47, 48, 49])
            panel(ax, letter, label)
            ax.legend(frameon=False, ncol=count, loc='upper left')
        for ax in axes[-1]: ax.set_xlabel('Time (s)')
        for ax in axes[:, 0]: ax.set_ylabel('Control angle (degree)')
        files = save_figure(fig, out, 'fig06_bcde_operational_controls', options)
    np.savez_compressed(out/'operational_controls_data.npz', time=TIME, angle=np.stack(traces))
    files.append('operational_controls_data.npz')
    return dict(files=files, inputs=[fingerprint(file), fingerprint(inspect.getfile(control))], selected_ids=[p['id'] for p in selected],
                selected_profiles=selected, manifest_identity=identity, recipe=manifest['recipe'],
                caption='Representative operation-informed control trajectories for the calibration study. '
                'The three lowest evaluation IDs for each maneuver type are selected independently of errors '
                'and coverage. Curves reconstruct the saved prescribed moves, not noisy applied-angle samples. '
                'Time and angle axes are shared. Add the protocol diagram (a) separately; no flowchart is generated here.')


def existing_assets(config, out, path, options):
    directory = path(config['assets_dir'])
    inputs, files, images = [], [], []
    for name in ASSETS:
        source = directory/name
        exported, size = image_export(source, out, source.stem, options)
        files.extend(exported)
        inputs.append(fingerprint(source))
        images.append(dict(name=name, pixels=size, unchanged=True))
    return dict(files=files, inputs=inputs, images=images, manual_revision_required=True,
                caption='Supplied plant, architecture, neural-autoregression, and rolling-forecast graphics, '
                'preserved unchanged. These are source assets, not corrected publication figures. Apply the '
                'outline corrections manually (13-row window, 15 outputs, member feedback, Y-to-X mapping, '
                'current plant labels). Combine architecture/autoregression for Figure 2 and move corrected '
                'Rolling Forecast to Appendix A. PDF/SVG wrappers contain the original raster pixels.')


def operational_forecasts(config, out, path, options):
    settings = config['operational']
    # Gate retained recipe before calling the existing, rigorously checked renderer.
    manifest = json.loads((path(settings['results_dir'])/'operational_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('recipe') != 'BCDF-independent-parameter-draws-v1':
        raise ValueError('Broader-angle study is excluded from manuscript figures')
    from plot_mace_operational_forecasts import run
    scaler = settings.get('scaler_file')
    report = run(path(settings['results_dir']), output_dir=out/'forecast_report',
                 scaler_file=path(scaler) if scaler else None,
                 profiles_per_family=settings['profiles_per_family'],
                 calibration=settings['calibration'], trajectory_mode=settings['trajectory_mode'],
                 dpi=options['dpi'], formats=('png',))
    files = [str(p.relative_to(out)) for p in sorted((out/'forecast_report').rglob('*')) if p.is_file()]
    return dict(files=files, inputs=list(report['input_hashes'].values()),
                grid_count=report['grid_count'], selected_profiles=report['selected_profiles'],
                caption='Saved operational forecasts with mean and native MACE endpoints, validated by the '
                'existing renderer. Individual grids are PNGs; one consolidated operational_forecasts.pdf '
                'contains sections and provenance. Calibration arms and per-output/joint trajectory views remain separate.')
