"""Score completed fixed-input probes with the frozen MNIST evaluator."""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from DiTLab.diagnostics.mnist_evaluator import load_evaluator, predict_accuracy

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]/'outputs/edm_diagnosis'
evaluator = load_evaluator()
scores = {}
for probe in sorted(root.glob('*/probes/probes.json')):
    saved = torch.load(probe.parent/'samples.pt',map_location='cpu',weights_only=True)
    results = {name:predict_accuracy(evaluator,images,saved['labels'])
               for name,images in saved['samples'].items()}
    scores[probe.parents[1].name] = results
    print(probe.parents[1].name, {key:round(value['accuracy'],4) for key,value in results.items()},flush=True)
(root/'ablation_scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
