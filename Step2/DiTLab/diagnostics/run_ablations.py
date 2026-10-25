"""One-factor training probes; original experiment files are never overwritten."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='+', default=['baseline','noise_scale100','more_high_noise','lr5e4'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--resume-epoch1', action='store_true')
    args = parser.parse_args()
    base = json.loads((ROOT/'DiTLab/outputs/edm_cfg/config.json').read_text(encoding='utf-8'))
    diagnostic = ROOT/'DiTLab/outputs/edm_diagnosis'
    variants = {'baseline': {}, 'noise_scale100': {'noise_condition_scale':100.},
                'more_high_noise': {'p_mean':0.}, 'lr5e4': {}}
    for name in args.names:
        config = copy.deepcopy(base)
        config['algorithm']['options'].update(variants[name])
        if name == 'noise_scale100':
            config['sampler']['options']['noise_condition_scale'] = 100.
        if name == 'lr5e4':
            config['training']['learning_rate'] = 5e-4
        directory = diagnostic/f'{name}_{args.epochs}ep'
        if directory.exists():
            raise FileExistsError(directory)
        directory.mkdir(parents=True)
        config['training'].update(epochs=args.epochs,output_dir=str(directory),
                                  preview_every=938,save_every=938,log_every=200)
        config['dataset']['options']['root'] = str(ROOT.parent/'data')
        path = directory/'requested_config.json'
        path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        started = time.perf_counter()
        command = [sys.executable,'-u','-m','DiTLab.train','--config',str(path)]
        if args.resume_epoch1:
            command = [sys.executable,'-u','-m','DiTLab.train','--resume',
                       str(diagnostic/f'{name}_1ep/weights/latest.pt'),'--epochs',str(args.epochs),
                       '--output-dir',str(directory)]
        print('[EDM diagnostic]', name, flush=True)
        with (directory/'train.log').open('w',encoding='utf-8') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line,end='',flush=True)
            code = process.wait()
        (directory/'runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-started,
            'exit_code':code,'command':command},indent=2),encoding='utf-8')
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}')
        subprocess.run([sys.executable,'-u',str(ROOT/'DiTLab/diagnostics/probe_edm.py'),
                        '--checkpoint',str(directory/'weights/latest.pt'),
                        '--output',str(directory/'probes'),'--quick'],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
