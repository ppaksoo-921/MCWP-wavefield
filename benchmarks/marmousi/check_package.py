"""CPU checks for actual-full-file preparation. GPU inference requires server smoke."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from scipy.ndimage import zoom
from common import HERE, SHOTS, config, jread, load_prepared, read_csv, sha
from prepare import affine, prepare
from raw_original import load_raw_original
from report import historical_update


@contextmanager
def fixture(mismatched_legacy=False):
    with tempfile.TemporaryDirectory(dir=HERE) as t:
        root=Path(t); (root/'Marmousi_OOD').mkdir()
        shutil.copy2(HERE/'source_metadata_reference.json',root/'Marmousi_OOD/metadata.json')
        z=np.linspace(0,1,751,dtype=np.float32)[:,None]
        x=np.linspace(.8,1,2301,dtype=np.float32)[None,:]
        full=(1500+4000*z*x).astype(np.float32)
        full.T.astype('<f4').tofile(root/'vp.bin')
        legacy=root/'GenWP-pub/dataset/marmousi_ood_main_outputs';legacy.mkdir(parents=True)
        public=root/'GenWP-pub/dataset/test/Marmousi';public.mkdir(parents=True)
        for s in (1,2):(public/f'shot{s}.mat').write_bytes(f'public unchanged {s}'.encode())
        blank=np.zeros((101,128,128),np.float32)
        for reg,x0,x1 in [('Left',0,767),('Middle',767,1534),('Right',1534,2301)]:
            raw=zoom(full[:,x0:x1],(128/751,128/767),order=1,prefilter=False).astype(np.float32)
            old=affine(raw,1500,5500,1924.1829833984375,4800)
            if mismatched_legacy:old=np.flipud(old).copy()
            for s,loc in enumerate(SHOTS,1):
                np.savez_compressed(legacy/f'MarmousiMain{reg}_vel1_shot{s}.npz',vp=old,snaps=blank,loc_x=loc)
        settings=jread(HERE/'config.json')
        settings.update(study_root=str(root),work_dir=str(root/'new_raw_outputs'),raw_binary_layout='xz_C')
        file=root/'config.json';file.write_text(json.dumps(settings));c=config(file)
        yield root,c,full


class RawOriginalChecks(unittest.TestCase):
    def test_minimum_and_global_endpoint_mapping(self):
        np.testing.assert_array_equal(affine(np.array([1500,3500,5500]),1500,5500,1500,5000),[1500,3250,5000])

    def test_binary_layout_and_actual_input(self):
        with fixture() as (root,c,full):
            got,source=load_raw_original(c)
            np.testing.assert_array_equal(got,full)
            self.assertTrue(source['actual_velocity_input'])
            self.assertEqual(source['file_sha256'],sha(root/'vp.bin'))

    def test_missing_raw_never_falls_back_to_existing_evaluation(self):
        with fixture() as (root,c,full):
            (root/'vp.bin').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'No fallback'):load_raw_original(c)

    def test_wrong_public_input_is_rejected(self):
        with fixture() as (root,c,full):
            c['original_binary']=str(root/'GenWP-pub/dataset/test/Marmousi/shot1.mat')
            with self.assertRaisesRegex(ValueError,'GenWP public'):load_raw_original(c)

    def test_raw_preparation_is_independent_of_wrong_legacy_values(self):
        with fixture(mismatched_legacy=True) as (root,c,full):
            public=root/'GenWP-pub/dataset/test/Marmousi'
            before={p.name:sha(p) for p in public.iterdir()}
            m=prepare(c)
            self.assertEqual(m['source_mode'],'direct_full_original_file')
            self.assertFalse(m['same_geometry_as_legacy_after_recorded_affine'])
            self.assertEqual(len(m['cases']),15)
            for info,(x0,x1) in zip(m['models'],[(0,767),(767,1534),(1534,2301)]):
                resized=zoom(full[:,x0:x1],(128/751,128/767),order=1,prefilter=False)
                expected=affine(resized,1500,5500,1500,5000)
                with np.load(root/'new_raw_outputs'/info['file']) as d:np.testing.assert_array_equal(d['vp'],expected)
                self.assertFalse(info['source']['legacy_main_used_as_velocity_input'])
                self.assertFalse(info['source']['hard_used_as_velocity_input'])
            self.assertLess(m['models'][0]['max_mps'],5000)
            self.assertEqual(before,{p.name:sha(p) for p in public.iterdir()})
            self.assertTrue((root/'new_raw_outputs/raw_original_preview.png').is_file())
            self.assertEqual(load_prepared(c)['experiment_fingerprint'],m['experiment_fingerprint'])
            with (root/'vp.bin').open('r+b') as f:f.write(b'bad!')
            with self.assertRaisesRegex(RuntimeError,'Full original input file changed'):load_prepared(c)

    def test_metrics_exclude_initial_reference(self):
        tree=ast.parse((HERE/'inference.py').read_text())
        f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='metrics')
        ns={'np':np};exec(compile(ast.Module(body=[f],type_ignores=[]),'metrics_test','exec'),ns)
        ref=np.full((101,2,2),2,np.float32);pred=ref.copy();pred[0]=9999
        metrics,frames=ns['metrics'](pred,ref);self.assertEqual(metrics['mae'],0)
        pred[1:]=3;metrics,frames=ns['metrics'](pred,ref)
        self.assertEqual(metrics['mae'],1);self.assertEqual(metrics['mse'],1);self.assertEqual(metrics['relative_l2'],.5)

    def test_macro_replacement_preserves_public_marmousi(self):
        ref=jread(HERE/'previous_reference_provenance.json');old=read_csv(HERE/'previous_66_case_accuracy.csv')
        models=list(dict.fromkeys(r['model'] for r in old))
        records=[dict(model=r['model'],label=r['label'],region='Test',case='NEW_'+r['case'],legacy_case=r['case'],
            mae=float(r['mae'])*.8,relative_l2=float(r['relative_l2'])*.8) for r in old if r['paper_group']=='marmousi_new']
        m=dict(same_geometry_as_legacy_after_recorded_affine=False,cases=[dict(legacy_case=k,legacy_file_sha256=v)
            for k,v in ref['old_marmousi_case_file_sha256'].items()])
        with tempfile.TemporaryDirectory(dir=HERE) as t:
            status,macro=historical_update({'models':models},m,records,Path(t))
            self.assertEqual(status['comparison_type'],'new_direct_raw_preparation_not_an_isolated_range_change')
            updated=read_csv(Path(t)/'updated_66_per_case.csv');self.assertEqual(len(updated),462)
            for model in models:
                prior=[r for r in old if r['model']==model]
                average=np.mean([np.mean([float(r['mae']) for r in prior if r['paper_group']==g]) for g in ref['group_counts']])
                marm=np.mean([float(r['mae']) for r in prior if r['paper_group']=='marmousi_new'])
                got=next(r for r in macro if r['model']==model)
                self.assertAlmostEqual(got['mae_equal_group'],average-.2*marm/6,places=13)
                a={r['case']:r['mae'] for r in prior if r['paper_group']=='marmousi_official'}
                b={r['case']:r['mae'] for r in updated if r['model']==model and r['paper_group']=='marmousi_official'}
                self.assertEqual(a,b)

    def test_vendor_files_match_supplied_study_code(self):
        self.assertEqual(sha(HERE/'vendor/fd_torch.py'),'39572b2eacbe83024deb48482fc68f34c704f3b0e8a7ad2376e528d14f08a6dd')
        self.assertEqual(sha(HERE/'vendor/fno_wrapper.py'),'1a77ab1fe2955068266650b85df9168b94d9a91360173954e3e7f8eaa697ddf3')


if __name__=='__main__':unittest.main(verbosity=2)
