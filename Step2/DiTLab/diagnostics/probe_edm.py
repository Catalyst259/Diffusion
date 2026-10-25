"""Fixed-input denoising and sampler probes for the MNIST EDM investigation."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from DiTLab.sample import load_generator
from DiTLab.data import build_mnist
from DiTLab.preview import save_preview
from DiTLab.algorithms.edm import EDMAlgorithm


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='DiTLab/outputs/edm_cfg/weights/latest.pt')
    parser.add_argument('--output', default='DiTLab/outputs/edm_diagnosis/original_probes')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, sampler, config, spec = load_generator(args.checkpoint)
    device = next(model.parameters()).device
    dataset = build_mnist('../data', download=False).train_dataset
    # Fixed examples from the training split, no official-test-based tuning.
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(871))[:256]
    images = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
    labels = torch.tensor([dataset[int(i)][1] for i in indices], device=device)
    noise = torch.randn(images.shape, device=device, generator=torch.Generator(device=device).manual_seed(872))
    probes = []
    for sigma in (.01, .03, .1, .3, 1., 3., 10., 80.):
        noisy = images + sigma * noise
        estimates, shuffled = [], []
        for start in range(0, len(images), 64):
            x, y = noisy[start:start+64], labels[start:start+64]
            estimates.append(sampler._denoise(model, x, y, torch.tensor(sigma, device=device)))
            shuffled.append(sampler._denoise(model, x, (y+1)%10, torch.tensor(sigma, device=device)))
        estimate, wrong = torch.cat(estimates), torch.cat(shuffled)
        mse = (estimate-images).square().mean().item()
        probes.append({'sigma': sigma, 'denoised_mse': mse,
                       'weighted_mse': mse*(sigma*sigma+.25)/(sigma*.5)**2,
                       'wrong_label_mse': (wrong-images).square().mean().item(),
                       'label_effect_rms': (wrong-estimate).square().mean().sqrt().item(),
                       'noisy_mse': (noisy-images).square().mean().item()})
    samples = {}
    sample_labels = torch.arange(10, device=device).repeat(10)
    initial = torch.randn((100,1,28,28), device=device,
                          generator=torch.Generator(device=device).manual_seed(873))
    settings = [('cfg2_steps32', 2., 32, 80.)]
    if not args.quick:
        settings += [('cfg1_steps32', 1.,32,80.), ('cfg2_steps128',2.,128,80.),
                     ('cfg2_max10',2.,32,10.)]
    for name, cfg, steps, maximum in settings:
        options = dict(config.sampler.options)
        options.update(cfg_scale=cfg, num_steps=steps, sigma_max=maximum)
        method = EDMAlgorithm(**options)
        generated = torch.cat([method.sample(model, initial[i:i+25], sample_labels[i:i+25])
                               for i in range(0,100,25)]).cpu()
        samples[name] = generated
        references = torch.stack([dataset[next(i for i in range(len(dataset)) if dataset.targets[i] == label)][0]
                                  for label in range(10)])
        save_preview(references, generated[:10], sample_labels[:10].cpu(), output/f'{name}.png', title=name)
    torch.save({'labels':sample_labels.cpu(),'samples':samples,'noise_seed':873}, output/'samples.pt')
    (output/'probes.json').write_text(json.dumps({'checkpoint':args.checkpoint, 'denoising':probes,
        'sample_count':100, 'sample_noise_seed':873, 'diagnostic_indices':indices.tolist()},indent=2),encoding='utf-8')
    print(json.dumps(probes,indent=2),flush=True)
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
