#!/usr/bin/env python3
"""Convert the supplied Pluto IEEE SEG-Y and prepare fixed geological crops.

Velocity arrays use (z, x) ordering. This reader intentionally supports the
actual attached fixed-length, big-endian, IEEE float32 file, not arbitrary SEG-Y.
"""
from pathlib import Path
import argparse, hashlib, json, struct
import numpy as np

FT_TO_M = 0.3048
SHOTS = [25, 38, 62, 105, 126]
# Fixed before any FD/model prediction. All intervals are Python half-open.
WINDOWS = [
    ('Pluto_01_Left', 0, 1201, 1450, 2651, 'left salt flank and underlying sediments'),
    ('Pluto_02_Central', 0, 1201, 3300, 4501, 'central salt body, basal boundary and right flank'),
    ('Pluto_03_Right', 0, 1201, 4500, 5701, 'right salt body and surrounding sediments'),
]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_pluto_segy(path):
    raw = bytearray(Path(path).read_bytes())
    if len(raw) < 3840:
        raise ValueError('Not a complete SEG-Y file')
    ns = struct.unpack_from('>H', raw, 3220)[0]
    fmt = struct.unpack_from('>H', raw, 3224)[0]
    units = struct.unpack_from('>H', raw, 3254)[0]
    rev = struct.unpack_from('>H', raw, 3500)[0]
    ext = struct.unpack_from('>h', raw, 3504)[0] if rev >= 256 else 0
    if fmt != 5 or ns != 1201 or units != 2 or ext != 0:
        raise ValueError(f'Expected attached Pluto: IEEE32 BE, ns=1201, English distance units, no extended headers; got {fmt=}, {ns=}, {units=}, {ext=}')
    stride = 240 + ns * 4
    count, remainder = divmod(len(raw)-3600, stride)
    if remainder or count != 6960:
        raise ValueError(f'Unexpected trace inventory: {count=}, {remainder=}')
    records = np.ndarray((count,), dtype=np.dtype([
        ('header', 'u1', (240,)), ('values', '>f4', (ns,))]), buffer=raw, offset=3600)
    headers = records['header'].copy()
    trace_ns = (headers[:,114].astype(np.uint16) << 8) + headers[:,115]
    if not np.all(trace_ns == ns):
        raise ValueError('Variable trace length is not supported')
    text = bytes(raw[:3200]).decode('cp500')
    if 'Pluto 1.5 Interval Velocity in Depth' not in text or '25 ft' not in text:
        raise ValueError('Attached Pluto header identity/grid interval does not match')
    vp_ft = records['values'].astype(np.float32).T.copy()
    if not np.isfinite(vp_ft).all() or not (4000 < vp_ft.min() < 6000 and 14000 < vp_ft.max() < 16000):
        raise ValueError('Unexpected velocity range; do not guess units')
    return raw, records, headers, vp_ft, text

def make_package(source, output, make_figures=True):
    from scipy.ndimage import zoom
    source = Path(source).resolve(); output = Path(output).resolve()
    (output/'full').mkdir(parents=True, exist_ok=True)
    (output/'velocity_models').mkdir(exist_ok=True)
    raw, records, headers, vp_ft, text = read_pluto_segy(source)
    # International foot, exact conversion in float64, then FP32 storage.
    exact = vp_ft.astype(np.float64) * FT_TO_M
    vp = exact.astype(np.float32)
    conversion_error = float(np.max(np.abs(vp.astype(np.float64)-exact)))
    np.save(output/'full/Pluto_full_vp_mps.npy', vp, allow_pickle=False)
    # Preserve original distance coordinates/header units (feet). Only velocities
    # change; no SEG-Y velocity-unit field is assumed to exist.
    records['values'][:] = vp.T
    lines = [text[i:i+80] for i in range(0,3200,80)]
    notes = {
        31:'VELOCITY SAMPLES: m/s; original ft/s multiplied by 0.3048.',
        32:'SPATIAL GRID: original 25 ft = 7.62 m; distance headers preserved.',
        33:'SAMPLE FORMAT: IEEE float32, big-endian. No amplitude normalization.',
        34:'OUTPUT ARRAY CONVENTION: depth z rows, horizontal x columns.',
        35:'Original SMAART JV property/attribution and other headers preserved.',
    }
    for i,note in notes.items():
        lines[i] = f'C{i+1:2d} {note}'[:80].ljust(80)
    raw[:3200] = ''.join(lines).encode('cp500')
    converted = output/'full/Pluto_full_vp_mps.sgy'
    converted.write_bytes(raw)
    # Independent byte-level reread proves sample orientation and conversion.
    out_records = np.ndarray((6960,), dtype=records.dtype, buffer=converted.read_bytes(), offset=3600)
    assert np.array_equal(out_records['values'].astype(np.float32).T, vp)
    assert np.array_equal(out_records['header'], headers)
    source_hash = sha256(source)
    report = {
        'dataset':'Pluto 1.5', 'source_filename':source.name,
        'source_sha256':source_hash, 'shape_zx':list(vp.shape),
        'original_velocity_unit':'ft/s', 'output_velocity_unit':'m/s',
        'multiplier':FT_TO_M, 'vp_min_mps':float(vp.min()), 'vp_max_mps':float(vp.max()),
        'dtype':'float32', 'conversion_max_rounding_error_mps':conversion_error,
        'original_dx_dz_ft':[25,25], 'original_dx_dz_m':[7.62,7.62],
        'origin':'Grid indices used; absolute survey origin is not inferred from zero coordinate headers.',
        'segy_coordinate_note':'Velocity values are m/s; distance headers and the depth sampling description remain in feet.',
        'no_clipping':True, 'no_range_matching':True, 'no_velocity_normalization':True,
        'source_attribution':'SMAART JV, Pluto 1.5; original source usage terms continue to apply.',
    }
    (output/'full/Pluto_full_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    models=[]
    for model_id,(name,z0,z1,x0,x1,purpose) in enumerate(WINDOWS,1):
        crop=vp[z0:z1,x0:x1].copy()
        assert crop.shape==(1201,1201)
        intermediate=zoom(crop,(256/1201,256/1201),order=1,prefilter=False)
        resized=zoom(intermediate,(.5,.5),order=1,prefilter=False).astype(np.float32)
        assert intermediate.shape==(256,256) and resized.shape==(128,128)
        assert np.isfinite(resized).all() and resized.min()>=1450 and resized.max()<=5000
        target=output/'velocity_models'/f'{name}_128.npz'
        np.savez_compressed(target, vp=resized, velocity_unit=np.asarray('m/s'),
            axis_order=np.asarray('z,x'), fd_dx_m=np.float32(10),fd_dz_m=np.float32(10),
            source_window_zx=np.asarray([z0,z1,x0,x1],dtype=np.int32),
            source_sha256=np.asarray(source_hash))
        np.save(output/'velocity_models'/f'{name}_128.npy',resized,allow_pickle=False)
        models.append({'model_id':model_id,'name':name,'path':str(target.relative_to(output)),
            'sha256':sha256(target),'source_window_zx_half_open':[z0,z1,x0,x1],
            'purpose':purpose,'vp_min_mps':float(resized.min()),'vp_max_mps':float(resized.max()),
            'original_crop_shape':[1201,1201],'original_crop_node_span_m':9144.0,
            'resampling':'bilinear 1201x1201 -> 256x256 -> 128x128; no Gaussian filter',
            'shape':[128,128],'fd_dx_m':10.0,'fd_dz_m':10.0,'shots_x':SHOTS})
    manifest={'dataset':'pluto','source':report,'models':models,'model_count':3,'shot_count':15,
        'forecast_frames':100,'initial_reference_frames':1,'snapshot_interval_s':.01,
        'fd':{'dx_m':10,'dz_m':10,'dt_s':.001,'nt':1001,'ricker_hz':15,'pml_cells':50,'time_downsample':10,'density':1},
        'case_order':'model_id ascending then shot index ascending',
        'case_index_start':51,'genwp_seed_start':5100,
        'evaluation_note':'New geology-family evaluation, velocity values inside the original training range. Three crops share one geological parent.',
        'geometric_note':'Regional square crops preserve relative x:z geometry. The 128-grid FD benchmark uses 10m spacing, so this is a rescaled benchmark, not propagation over the original 9.144km crop at its original physical scale.',
        'selection_note':'Coordinates fixed using only the input geological structure before any FD or learned-model result.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if make_figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        (output/'figures').mkdir(exist_ok=True)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
        fig,ax=plt.subplots(figsize=(15,4.4))
        im=ax.imshow(vp,cmap='turbo',vmin=1450,vmax=5000,origin='upper',aspect='auto')
        for j,(name,z0,z1,x0,x1,_) in enumerate(WINDOWS,1):
            ax.add_patch(Rectangle((x0-.5,z0-.5),x1-x0,z1-z0,fill=False,edgecolor='white',linewidth=1.8))
            ax.text((x0+x1)/2,75,f'{j:02d}',ha='center',color='white',weight='bold',fontsize=14)
        ax.set_xlabel('Original x grid index');ax.set_ylabel('Original z grid index')
        ax.set_title('Pluto 1.5 · m/s · fixed evaluation windows',loc='left',weight='bold')
        fig.colorbar(im,ax=ax,label='P-wave velocity (m/s)');fig.tight_layout()
        fig.savefig(output/'figures/Pluto_full_windows.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(12,4.6))
        for ax,item in zip(axs,models):
            with np.load(output/item['path'],allow_pickle=False) as d:arr=d['vp']
            im=ax.imshow(arr,cmap='turbo',vmin=1450,vmax=5000,origin='upper',interpolation='nearest')
            ax.set_title(item['name'].replace('Pluto_','').replace('_',' · '),weight='bold')
            ax.set_xticks([0,64,127]);ax.set_yticks([0,64,127]);ax.set_xlabel('x grid index')
        axs[0].set_ylabel('z grid index')
        fig.subplots_adjust(left=.055,right=.91,bottom=.18,top=.84,wspace=.17)
        cax=fig.add_axes([.93,.25,.014,.50]);fig.colorbar(im,cax=cax,label='P-wave velocity (m/s)')
        fig.suptitle('Pluto 1.5 · 128 x 128 OOD velocity inputs',x=.055,ha='left',weight='bold',fontsize=16)
        fig.text(.055,.06,'Same speed units and FD grid as the existing study · no clipping or per-model normalization',fontsize=10,color='#526170')
        fig.savefig(output/'figures/Pluto_128_velocity_models.png',dpi=170);plt.close(fig)
    return manifest

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--no-figures',action='store_true')
    args=ap.parse_args()
    r=make_package(args.source,args.output,not args.no_figures)
    print(json.dumps({'shape':r['source']['shape_zx'],'min_mps':r['source']['vp_min_mps'],
        'max_mps':r['source']['vp_max_mps'],'models':3,'future_cases':15},indent=2))
if __name__=='__main__':main()
