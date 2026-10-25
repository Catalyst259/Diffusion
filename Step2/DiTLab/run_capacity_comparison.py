"""Run the matched 57.66M-parameter MNIST Drift/EDM experiment sequentially.

python -m DiTLab.run_capacity_comparison --stage preflight
python -m DiTLab.run_capacity_comparison --stage train
Completed runs are retained; use --resume to continue an interrupted experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

from .artifacts import load_checkpoint, save_json
from .config import ExperimentConfig, ROOT
from .engine import Trainer
from .registry import build_algorithm, build_dataset, build_model, build_sampler


PARAMETERS = 57656848
MODEL_OPTIONS = {'patch_size':4, 'hidden_size':512, 'depth':12, 'num_heads':8}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def tensor_digest(model):
    digest = hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(directory):
    manifest_path = directory/'experiment.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding='utf-8'))
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Existing nonempty directory without experiment manifest: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'configs').mkdir()
    base = json.loads((ROOT/'outputs/default/config.json').read_text(encoding='utf-8'))
    edm = json.loads((ROOT/'configs/mnist_edm_cfg.json').read_text(encoding='utf-8'))
    configs = {}
    for method in ('drift','edm'):
        values = json.loads(json.dumps(base if method == 'drift' else edm))
        values['model'] = {'name':'dit','options':MODEL_OPTIONS}
        values['training'].update(epochs=5,batch_size=64,learning_rate=1e-4,weight_decay=0.,
            max_grad_norm=2.,seed=42,device='cuda',preview_every=500,save_every=500,
            preview_count=10,log_every=50,num_workers=0,output_dir=str(directory/method))
        values['dataset'] = {'name':'mnist','options':{'root':str(ROOT.parent.parent/'data'),'download':False}}
        config = ExperimentConfig.from_dict(values)
        configs[method] = config.to_dict()
        save_json(config.to_dict(),directory/'configs'/f'{method}.json')
    manifest = {'created_at':timestamp(),'status':'configured','parameters':PARAMETERS,
        'unet_parameters':56960545,'model':MODEL_OPTIONS,'configs':configs,
        'training_order':['drift','edm'],'initialization_seed':42,
        'precision':'float32','gradient_accumulation':False,'activation_checkpointing':False,
        'notes':['Equal architecture, initial weights, real data batches and update budget.',
                 'Drift generates 8 particles per present class (up to 80); EDM denoises the 64 real examples.',
                 'Each method uses its own objective; loss magnitudes and compute are not identical.']}
    save_json(manifest,manifest_path)
    return manifest


def create_components(directory,method):
    config = ExperimentConfig.load(directory/'configs'/f'{method}.json')
    torch.manual_seed(config.training.seed)
    data = build_dataset(config)
    model = build_model(config,data.spec)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    return config,data,model,build_algorithm(config),build_sampler(config)


def probe(directory,method):
    torch.set_num_threads(4)
    config,data,model,algorithm,sampler = create_components(directory,method)
    digest = tensor_digest(model)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.training.learning_rate,weight_decay=0.)
    order = torch.randperm(len(data.train_dataset),generator=torch.Generator().manual_seed(42))[:64]
    images = torch.stack([data.train_dataset[int(i)][0] for i in order]).cuda()
    labels = torch.tensor([data.train_dataset[int(i)][1] for i in order],device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    durations,losses = [],[]
    for step in range(8):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = algorithm.loss(model,images,labels,generator=generator,training=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Preflight loss is nonfinite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-start)
        losses.append(float(loss.detach()))
    train_peak = torch.cuda.max_memory_allocated()/1024**2
    noise = torch.randn(10,1,28,28,device='cuda',generator=generator)
    generated = sampler.sample(model,noise,labels[:10])
    assert generated.shape == noise.shape and torch.isfinite(generated).all()
    result = {'method':method,'passed':True,'initial_weights_sha256':digest,
        'parameters':PARAMETERS,'real_batch_size':64,'present_classes':int(labels.unique().numel()),
        'precision':'float32','warmup_steps':2,'measured_steps':6,'seconds_per_update':sum(durations[2:])/6,
        'train_peak_allocated_MiB':train_peak,'overall_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2,'losses':losses,
        'gpu':torch.cuda.get_device_name(),'torch_version':str(torch.__version__),
        'gpu_total_memory_MiB':torch.cuda.get_device_properties(0).total_memory/1024**2,
        'sampling_verified':True,'preflight_weights_discarded':True}
    save_json(result,directory/f'preflight_{method}.json')
    print(json.dumps(result,indent=2),flush=True)


def train_method(directory,method,resume):
    torch.set_num_threads(4)
    config,data,model,algorithm,sampler = create_components(directory,method)
    digest = tensor_digest(model)
    probe_record = json.loads((directory/f'preflight_{method}.json').read_text())
    assert digest == probe_record['initial_weights_sha256']
    latest = directory/method/'weights/latest.pt'
    if latest.exists() and not resume:
        raise FileExistsError(f'Use --resume for {latest}')
    checkpoint = load_checkpoint(latest) if latest.exists() else None
    trainer = Trainer(config,data,model,algorithm,sampler,checkpoint=checkpoint)
    torch.cuda.reset_peak_memory_stats()
    started_at,start = timestamp(),time.perf_counter()
    trainer.run()
    torch.cuda.synchronize()
    result = {'method':method,'status':'completed','started_at':started_at,'finished_at':timestamp(),
        'wall_seconds':time.perf_counter()-start,'epochs':trainer.epoch,'global_step':trainer.global_step,
        'parameters':PARAMETERS,'initial_weights_sha256':digest,'resumed':checkpoint is not None,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2,'precision':'float32',
        'gpu':torch.cuda.get_device_name(),'torch_version':str(torch.__version__)}
    save_json(result,directory/method/'runtime.json')
    print(json.dumps(result,indent=2),flush=True)


def subprocess_logged(command,log_path):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8') as log:
        process = subprocess.Popen(command,cwd=ROOT.parent,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line,end='',flush=True)
        result = process.wait()
    if result:
        raise RuntimeError(f'Child process failed ({result}); see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/dit57m_compare')
    parser.add_argument('--stage',choices=('preflight','train'),default='preflight')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--child',choices=('drift','edm'))
    args = parser.parse_args()
    directory = args.root.resolve()
    if args.child:
        (probe(directory,args.child) if args.stage == 'preflight'
         else train_method(directory,args.child,args.resume))
        return
    manifest = prepare(directory)
    if args.stage == 'train':
        probes = [json.loads((directory/f'preflight_{method}.json').read_text()) for method in ('drift','edm')]
        assert all(p['passed'] for p in probes)
        assert probes[0]['initial_weights_sha256'] == probes[1]['initial_weights_sha256']
    manifest['status'] = args.stage+'_running'
    save_json(manifest,directory/'experiment.json')
    try:
        for method in ('drift','edm'):
            if args.stage == 'train' and args.resume and (directory/method/'runtime.json').exists():
                previous = json.loads((directory/method/'runtime.json').read_text())
                if previous.get('status') == 'completed' and previous.get('global_step') == 4690:
                    print('Already completed:',method,flush=True)
                    continue
            command = [sys.executable,'-u','-m','DiTLab.run_capacity_comparison',
                       '--root',str(directory),'--stage',args.stage,'--child',method]
            if args.resume:
                command.append('--resume')
            print(f'{args.stage}: {method}',flush=True)
            log = (directory/f'preflight_{method}.log' if args.stage == 'preflight'
                   else directory/method/'train.log')
            subprocess_logged(command,log)
        manifest['status'] = args.stage+'_completed'
        manifest[args.stage+'_finished_at'] = timestamp()
    except Exception:
        manifest['status'] = args.stage+'_failed'
        raise
    finally:
        save_json(manifest,directory/'experiment.json')


if __name__ == '__main__':
    main()
