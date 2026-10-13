"""Check training, resume, inference, and PNG output using synthetic data."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import nibabel as nib
import numpy as np
import yaml


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/rsam.yaml').read_text())
    config['model'].update(base_channels=8, hidden_dim=8, image_size=32, n_residual_blocks=3)
    cfg=config['training']
    cfg.update(epochs=2, decay_start_epoch=1, global_batch_size=3, slices_per_case=1,
               num_workers=0, cpu_threads=1, save_every_epochs=1, horizontal_flip=False, region_balanced_sampling=False)
    cfg['loss_weights']['perceptual']=0.
    (output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    grid=np.mgrid[-1:1:32j,-1:1:32j,-1:1:5j]
    x,y,z=grid
    body=x*x/.8**2+y*y/.7**2<1
    mr=np.where(body, 110+45*x+20*y+7*z, 0).astype('float32')
    ct=np.where(body, 30+65*x+25*z, -1000).astype('float32')
    ct[((x/.8)**2+(y/.7)**2>.8)&body]=800
    affine=np.diag([1.2,1.2,2.5,1.])
    rows=[]
    for split in ['train','val']:
        for region in config['region_order']:
            case=f'{region}_{split}';pid=f'subject_{case}'
            paired=region!='abdomen'
            for name,array in [('mr',mr),('ct',ct)]:
                nib.save(nib.Nifti1Image(array,affine),output/f'{case}_{name}.nii.gz')
            rows.append(dict(case_id=case,patient_id=pid,ct_patient_id=pid if paired else pid+'_ct',
                region=region,split=split,paired=str(paired).lower(),mr=f'{case}_mr.nii.gz',ct=f'{case}_ct.nii.gz'))
    with (output/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    out=prepare(args.output)
    if args.prepare_only:return
    cli=Path(sys.executable).parent/'rsam'
    if not cli.is_file():raise RuntimeError("Install this package first: python -m pip install -e .")
    def run(*command):subprocess.run([cli,*map(str,command)],check=True)
    train=['train','--config',out/'config.yaml','--manifest',out/'manifest.csv',
           '--output',out/'run','--device',args.device]
    run(*train,'--max-steps',1)
    import torch
    first=torch.load(out/'run/training_last.pt',map_location='cpu',weights_only=True)
    assert first['steps']==1 and first['next_epoch']==1
    run(*train,'--resume',out/'run/training_last.pt')
    state=torch.load(out/'run/training_last.pt',map_location='cpu',weights_only=True)
    assert state['steps']==2 and state['next_epoch']==2
    assert any(k.startswith('projection_heads.') for k in state['generators'])
    run('infer','--checkpoint',out/'run/generator_mr_to_ct.pt','--manifest',out/'manifest.csv',
        '--split','val','--output',out/'predictions','--device',args.device)
    for region in ['brain','pelvis','abdomen']:
        ref=nib.load(out/f'{region}_val_mr.nii.gz');pred=nib.load(out/f'predictions/{region}_val_sct.nii.gz')
        assert ref.shape==pred.shape and np.array_equal(ref.affine,pred.affine)
        assert np.isfinite(pred.get_fdata()).all()
    run('visualize','--mr',out/'brain_val_mr.nii.gz','--prediction',out/'predictions/brain_val_sct.nii.gz',
        '--ct',out/'brain_val_ct.nii.gz','--slice',2,'--output',out/'comparison.png')
    assert (out/'comparison.png').is_file()
    print(json.dumps({'status':'PASS','completed':['train','resume','infer','visualize'],
                      'output':str(out),'quality_benchmark':False}))

if __name__=='__main__':main()
