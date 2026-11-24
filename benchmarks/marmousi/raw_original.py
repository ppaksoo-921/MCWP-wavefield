"""Direct processing of the user-designated full original Marmousi file."""
from pathlib import Path
import numpy as np
from common import (FD, GROUP, REGIONS, SHOTS, array_sha, digest, implementation_hashes,
                    jread, jwrite, load_prepared, public_snapshot, save_npz, sha,
                    vendor_hashes, verify_public)
from prepare import affine, load_legacy


def load_raw_original(c):
    path = Path(c['original_binary'])
    if not path.is_file():
        raise FileNotFoundError('FULL original Marmousi required: %s. No fallback to main, hard, or GenWP public data.' % path)
    public = (Path(c['study_root']) / 'GenWP-pub/dataset/test/Marmousi').resolve()
    if path.resolve().is_relative_to(public):
        raise ValueError('GenWP public Marmousi cannot be the full original input.')
    layout = c.get('raw_binary_layout', 'xz_C')
    if path.suffix.lower() == '.bin':
        flat = np.fromfile(path, dtype='<f4')
        if flat.size != 751 * 2301:
            raise ValueError('Expected little-endian float32, 751*2301 samples: ' + str(path))
        if layout == 'xz_C':
            raw = flat.reshape(2301, 751).T
        elif layout == 'zx_C':
            raw = flat.reshape(751, 2301)
        else:
            raise ValueError('raw_binary_layout must be xz_C or zx_C.')
    elif path.suffix.lower() == '.npy':
        raw = np.load(path, allow_pickle=False)
        layout = 'npy_explicit_zx'
    else:
        raise ValueError('Use the full original .bin or full-resolution .npy in z,x order.')
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    if raw.shape != (751, 2301) or not np.isfinite(raw).all():
        raise ValueError('Expected finite full 751x2301 original data, not a 128x128 evaluation case.')
    if abs(float(raw.min()) - 1500) > .1 or abs(float(raw.max()) - 5500) > .1:
        raise ValueError('The original file differs from the declared 1500..5500 m/s model. Check source/units.')
    return raw, dict(path=str(path), file_sha256=sha(path), array_sha256=array_sha(raw),
        layout=layout, shape_zx=[751, 2301], min_mps=float(raw.min()), max_mps=float(raw.max()),
        top_row_min_mps=float(raw[0].min()), top_row_max_mps=float(raw[0].max()),
        actual_velocity_input=True, identity_check='User-designated full file; shape, dtype, units/range checks')


def prepare(c):
    from scipy.ndimage import zoom
    work = Path(c['work_dir'])
    target = work / 'preparation.json'
    if target.is_file():
        m = load_prepared(c)
        if m.get('source_mode') != 'direct_full_original_file':
            raise ValueError('Use a NEW work_dir for direct-original processing; preserve old inverse results.')
        print('Direct-original preparation reused after hash checks:', work, flush=True)
        return m
    meta = jread(c['metadata'])
    coordinates = {'left': [0, 767], 'middle': [767, 1534], 'right': [1534, 2301]}
    if meta.get('original_shape_zx') != [751, 2301] or meta.get('regions') != coordinates:
        raise ValueError('Expected full 751x2301 Marmousi and the recorded three raw crop windows.')
    full, raw_source = load_raw_original(c)
    protected = public_snapshot(c)
    models, cases, geometry_agreement = [], [], []
    for region in REGIONS:
        x0, x1 = coordinates[region.lower()]
        original_128 = zoom(full[:, x0:x1], (128 / 751, 128 / 767),
                            order=1, prefilter=False).astype(np.float32)
        if original_128.shape != (128, 128):
            raise ValueError('Unexpected resampled raw-model shape.')
        new = affine(original_128, 1500, 5500, 1500, 5000)
        if new.min() < 1499.99 or new.max() > 5000.01:
            raise ValueError('New velocity escaped 1500..5000 m/s.')
        source = dict(method='direct_crop_and_linear_resize_of_full_original_file',
            raw_source_file=raw_source['path'], raw_source_sha256=raw_source['file_sha256'],
            crop_zx_half_open=[0, 751, x0, x1], original_crop_shape=[751, 767],
            resampling='scipy.ndimage.zoom(order=1,prefilter=False), 751x767 -> 128x128',
            legacy_main_used_as_velocity_input=False, hard_used_as_velocity_input=False)
        region_cases, old = [], None
        for shot, expected_x in enumerate(SHOTS, 1):
            legacy_name = 'MarmousiMain%s_vel1_shot%d' % (region, shot)
            path = Path(c['legacy_cases_dir']) / (legacy_name + '.npz')
            historical_vp, loc = load_legacy(path)
            if loc != expected_x:
                raise ValueError('Historical source location differs: ' + str(path))
            if old is not None and not np.array_equal(old, historical_vp):
                raise ValueError('Historical shots do not share a velocity model: ' + region)
            old = historical_vp
            new_name = 'MarmousiOriginalV1500_5000_%s_shot%d' % (region, shot)
            region_cases.append(dict(case=new_name, group=GROUP, region=region, shot=shot,
                loc_x=expected_x, legacy_case=legacy_name, legacy_file=str(path),
                legacy_file_sha256=sha(path), old_velocity_sha256=array_sha(historical_vp),
                genwp_noise_seed=(6 + len(cases) + shot - 1) * 100,
                file='fd_cases/' + new_name + '.npz'))
        # Audit only: historical main velocities never influence new input values.
        old_low, old_high = map(float, meta['main_ood_target_range'])
        expected_old = affine(original_128, 1500, 5500, old_low, old_high)
        diff = np.abs(expected_old.astype(np.float64) - old.astype(np.float64))
        matches = bool(diff.max() <= .05)
        geometry_agreement.append(matches)
        source['comparison_with_historical_main'] = dict(max_abs_mps=float(diff.max()),
            mean_abs_mps=float(diff.mean()), matches_recorded_affine_and_geometry=matches,
            influence_on_new_velocity='none')
        relative = 'velocity_models/MarmousiOriginalV1500_5000_%s.npz' % region
        save_npz(work / relative, vp=new, original_vp=original_128, previous_main_vp=old,
                 velocity_unit=np.array('m/s'), region=np.array(region))
        models.append(dict(region=region, file=relative, file_sha256=sha(work / relative),
            new_velocity_sha256=array_sha(new), min_mps=float(new.min()), max_mps=float(new.max()),
            mean_mps=float(new.mean()), source=source))
        for item in region_cases:
            item['velocity_sha256'] = array_sha(new)
            item['velocity_file'] = relative
            cases.append(item)
    identity = dict(group=GROUP, source_mode='direct_full_original_file',
        source_kind='user_designated_full_original_Marmousi_not_GenWP_public',
        source_global_range_mps=[1500.0, 5500.0], target_global_range_mps=[1500.0, 5000.0],
        formula='v_new = 1500 + (v_original - 1500) * 3500 / 4000',
        per_crop_minmax_rescaling=False, clipping=False, grid=[128, 128], fd=FD,
        raw_binary_audit=raw_source, same_geometry_as_legacy_after_recorded_affine=all(geometry_agreement),
        network_velocity_normalization_range_mps=[1400.0, 5000.0],
        initial_condition='FD frame 0 plus four older zeros; predict 1..100 without FD reinjection',
        raw_metadata_file=c['metadata'], raw_metadata_sha256=sha(c['metadata']),
        legacy_cases_dir=c['legacy_cases_dir'], protected_genwp_public_files=protected,
        vendor_hashes=vendor_hashes(), implementation_hashes=implementation_hashes(), models=models, cases=cases)
    identity['experiment_fingerprint'] = digest(identity)
    jwrite(target, identity)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(full, cmap='viridis', vmin=1500, vmax=5500, aspect='auto')
    for x in (767, 1534):ax.axvline(x, color='white', linewidth=1)
    ax.set(title='Actual full original-file input, before velocity mapping', xlabel='x index', ylabel='z index')
    fig.colorbar(im, ax=ax, label='P-wave velocity (m/s)')
    fig.tight_layout(); fig.savefig(work / 'raw_original_preview.png', dpi=150); plt.close(fig)
    verify_public(protected)
    print('FULL ORIGINAL FILE USED:', raw_source['path'], flush=True)
    print('Layout:', raw_source['layout'], '; SHA256:', raw_source['file_sha256'], flush=True)
    print('Prepared raw-file crops / 15 source cases:', work, flush=True)
    print('Matches historical preparation:', all(geometry_agreement), '(audit only)', flush=True)
    return identity
