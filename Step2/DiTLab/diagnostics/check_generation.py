"""End-to-end checkpoint regression on fresh noise and a frozen classifier.

This is a diagnostic label-agreement check, not a perceptual quality benchmark.
Run with the original checkpoint to reproduce failure and the balanced-noise
checkpoint to verify recovery. Its seed was not used in the ablation probes.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from DiTLab.sample import load_generator
from DiTLab.diagnostics.mnist_evaluator import load_evaluator, predict_accuracy


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=500)
    parser.add_argument('--seed',type=int,default=19873)
    parser.add_argument('--min-accuracy',type=float,default=.5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model,sampler,config,spec = load_generator(args.checkpoint)
    device = next(model.parameters()).device
    labels = torch.arange(args.count,device=device)%10
    # A single fixed tensor means batching and model loading cannot alter noise.
    noise = torch.randn(args.count,1,28,28,device=device,
                        generator=torch.Generator(device=device).manual_seed(args.seed))
    generated = torch.cat([sampler.sample(model,noise[i:i+50],labels[i:i+50]).cpu()
                           for i in range(0,args.count,50)])
    evaluator = load_evaluator()
    metrics = predict_accuracy(evaluator,generated,labels.cpu())
    destination = Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    torch.save({'samples':generated,'labels':labels.cpu(),'seed':args.seed},destination/'samples.pt')
    report = {'checkpoint':str(Path(args.checkpoint).resolve()),'noise_seed':args.seed,
              'sample_count':args.count,'metrics':metrics,'threshold':args.min_accuracy,
              'passed':metrics['accuracy']>=args.min_accuracy,
              'evaluator_validation_accuracy':evaluator.evaluator_metadata['final_validation']['accuracy'],
              'limitations':'Classifier label agreement on generated images; not FID or proof of visual validity.'}
    (destination/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('sample_count','noise_seed','passed')}))
    print('label agreement:',metrics['accuracy'],flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
