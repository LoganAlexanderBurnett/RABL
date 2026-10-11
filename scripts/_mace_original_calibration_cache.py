"""Validate and copy completed corrected original-calibration forecasts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py

from _mace_floor_study import fingerprint
from _mace_operational_profiles import manifest_identity
from _mace_scaler_audit import decode_names

FEEDBACK = 'native-Y-to-X-affine-float32-v1'
# Reviewed pre-extension helper, LF and Git-for-Windows CRLF. The corrected
# recurrence/aggregation functions remain unchanged in this extension.
REVIEWED_FORECAST_SOURCES = {
    'fa2108e7831ea45354466a191d72a67a40480a2362cf475dc509b356343bb77d',
    '35f85d29136483c01b7ce34463a4b6692751b4cd6e65d0b76c371f0f2d0f1a8f'}
ENGINE_SOURCES = ('src/rabl/machine_learning/lstm_pipeline.py',
                  'src/rabl/machine_learning/bagging_ensemble.py',
                  'scripts/_mace_scaler_use.py', 'scripts/_mace_scaler_audit.py')


def validate_cache(config, prior, hashes, *, root, runtime=None):
    source = Path(config['reuse_original_calibration_dir'])
    destination = Path(config['output_dir'])
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Reuse source and new output must be separate sibling studies')
    paths = dict(forecasts=source/'forecasts/original_cal.h5',
                 forecast_metadata=source/'forecasts/forecast_metadata.json',
                 evaluation_metadata=source/'evaluation/evaluation_metadata.json',
                 manifest=source/'operational_manifest.json')
    records = {k: fingerprint(p) for k, p in paths.items()}
    fm, em, manifest = [json.loads(paths[k].read_text()) for k in ('forecast_metadata', 'evaluation_metadata', 'manifest')]
    if not fm.get('completed') or not fm.get('inputs_unchanged') or not em.get('completed') or fm.get('feedback') != FEEDBACK:
        raise ValueError('Reuse requires completed corrected forecasts/evaluation with unchanged inputs')
    identity = manifest_identity(manifest['profiles'])
    if manifest['identity'] != identity or any(m.get('manifest_identity') != identity for m in (fm, em)):
        raise ValueError('Cache origin manifest identity mismatch')
    if fm['target_names'] != prior['target_names'] or em['target_names'] != prior['target_names'] or fm['original_calibration_profile_names'] != prior['profile_names']['cal']:
        raise ValueError('Cache original calibration identities/target order mismatch')
    expected = {k: v['sha256'] for k, v in hashes.items()}
    if any({k: v['sha256'] for k, v in m['input_hashes'].items()} != expected for m in (fm, em)):
        raise ValueError('Cache frozen datasets/scaler/checkpoints differ')
    recorded_sources = {k.replace('\\', '/'): v for k, v in em['source_hashes'].items()}
    for name in ENGINE_SOURCES:
        if recorded_sources.get(name) != fingerprint(root/name)['sha256']:
            raise ValueError(f'Cache prediction/scaling source changed: {name}')
    forecast_source = recorded_sources.get('scripts/_mace_operational_forecasts.py')
    if forecast_source not in REVIEWED_FORECAST_SOURCES | {fingerprint(root/'scripts/_mace_operational_forecasts.py')['sha256']}:
        raise ValueError('Cache recurrence source is not a reviewed compatible version')
    if runtime is not None and fm['runtime'] != runtime:
        raise ValueError('Cache runtime/precision settings differ; use reuse_original_calibration_dir=null to recompute original calibration')
    for m, key in ((fm, 'output_hashes'), (em, 'forecast_hashes')):
        if records['forecasts']['sha256'] != m[key]['original_cal.h5']['sha256']:
            raise ValueError('Cached original calibration forecast changed')
    original_identity = fm.get('original_calibration_manifest_identity', identity)
    with h5py.File(paths['forecasts']) as f:
        if (not f.attrs.get('completed') or f.attrs.get('feedback') != FEEDBACK
                or f.attrs.get('manifest_identity') != original_identity
                or int(f.attrs.get('member_count', 0)) != 10 or int(f.attrs.get('ensemble_std_ddof', -1)) != 0
                or decode_names(f.attrs.get('target_names', [])) != prior['target_names']
                or set(f.keys()) != set(prior['profile_names']['cal'])):
            raise ValueError('Cached forecast native provenance mismatch')
    return dict(source_directory=str(source), input_records=records,
                original_manifest_identity=original_identity, runtime=fm['runtime'])


def copy_cache(record, destination):
    source = Path(record['input_records']['forecasts']['path'])
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)
    if fingerprint(destination)['sha256'] != record['input_records']['forecasts']['sha256']:
        raise ValueError('Original calibration cache copy is not byte-identical')
    for value in record['input_records'].values():
        if fingerprint(Path(value['path']))['sha256'] != value['sha256']:
            raise ValueError('Cache origin changed during reuse')
    return {**record, 'copied_byte_identical': True, 'original_inference_runs': 0}
