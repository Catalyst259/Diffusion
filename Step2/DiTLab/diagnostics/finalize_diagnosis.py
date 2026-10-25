"""Publish verified diagnostic artifacts without changing either original run."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from PIL import Image,ImageDraw,ImageFont
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from DiTLab.artifacts import atomic_checkpoint, save_json

torch.set_num_threads(4)
diagnostic = ROOT/'DiTLab/outputs/edm_diagnosis'
source = diagnostic/'more_high_noise_5ep'
destination = ROOT/'DiTLab/outputs/edm_cfg_balanced'
if destination.exists():
    raise FileExistsError(destination)
shutil.copytree(source,destination)
checkpoint = torch.load(destination/'weights/latest.pt',map_location='cpu',weights_only=True)
checkpoint['config']['training']['output_dir'] = str(destination)
atomic_checkpoint(checkpoint,destination/'weights/latest.pt')
save_json(checkpoint['config'],destination/'config.json')
save_json({'source_run':str(source),'source_checkpoint_sha256':
           hashlib.sha256((source/'weights/latest.pt').read_bytes()).hexdigest(),
           'operation':'Copied completed 5-epoch result; changed checkpoint output_dir metadata only.',
           'epoch1_run':str(diagnostic/'more_high_noise_1ep'),
           'training_changes_from_original':{'algorithm.p_mean':[-1.2,0.0]},
           'artifact_cadence':'Diagnostic previews/checkpoints each 938 updates, final 4690; original cadence 500.',
           'rng_note':'Preview/evaluation RNG is isolated; cadence does not affect optimizer updates.'},
          destination/'provenance.json')

# Preserve the exact old preset, then make the diagnosed setting the default
# MNIST EDM recipe. Existing output configs/checkpoints remain authoritative.
preset = ROOT/'DiTLab/configs/mnist_edm_cfg.json'
legacy = preset.with_name('mnist_edm_cfg_legacy.json')
if legacy.exists():
    raise FileExistsError(legacy)
shutil.copy2(preset,legacy)
settings = json.loads(preset.read_text(encoding='utf-8'))
settings['algorithm']['options']['p_mean'] = 0.0
settings['training']['output_dir'] = 'DiTLab/outputs/edm_cfg_balanced'
save_json(settings,preset)

original = json.loads((diagnostic/'regression_original/result.json').read_text())
balanced = json.loads((diagnostic/'regression_balanced/result.json').read_text())
assert not original['passed'] and balanced['passed']
assert original['noise_seed'] == balanced['noise_seed']
original_samples = torch.load(diagnostic/'regression_original/samples.pt',weights_only=True)
balanced_samples = torch.load(diagnostic/'regression_balanced/samples.pt',weights_only=True)
assert torch.equal(original_samples['labels'],balanced_samples['labels'])

font = ImageFont.load_default(size=16)
# Fixed final training preview, no sample selection.
panels = [Image.open(ROOT/'DiTLab/outputs/edm_cfg/previews/step_00004690.png').convert('RGB'),
          Image.open(destination/'previews/step_00004690.png').convert('RGB')]
width,height = panels[0].size
comparison = Image.new('RGB',(width,2*(height+34)),(245,245,245))
draw = ImageDraw.Draw(comparison)
titles = ['Original EDM | p_mean=-1.2 | 5 epochs | same DiT / LR / seed',
          'Balanced-noise EDM | p_mean=0.0 | 5 epochs | same DiT / LR / seed']
for row,(panel,title) in enumerate(zip(panels,titles)):
    y = row*(height+34)
    draw.text((12,y+8),title,fill='black',font=font)
    comparison.paste(panel,(0,y+34))
comparison.save(destination/'graph/before_after.png')

# First 30 images from the fresh 500-image validation seed, unfiltered.
tile,gap,left,header = 112,10,130,54
grid = Image.new('RGB',(left+10*(tile+gap)+12,header+6*(tile+gap)+12),(240,240,240))
draw = ImageDraw.Draw(grid)
draw.text((12,10),'Fresh seed 19873 | first 30 / 500 samples | old 11.2%, balanced 76.4% label agreement',
          fill='black',font=font)
for column in range(10):
    draw.text((left+column*(tile+gap),32),f'Label {column}',fill='black',font=font)
for block in range(3):
    for method,payload in enumerate((original_samples,balanced_samples)):
        row=2*block+method
        y=header+row*(tile+gap)
        draw.text((10,y+42),'Original EDM' if method==0 else 'Balanced EDM',fill='black',font=font)
        for column in range(10):
            tensor=payload['samples'][block*10+column,0]
            pixels=(((tensor+1)/2).clamp(0,1)*255).round().to(torch.uint8).numpy()
            tile_image=Image.fromarray(pixels).convert('RGB').resize((tile,tile),Image.Resampling.NEAREST)
            grid.paste(tile_image,(left+column*(tile+gap),y))
grid.save(destination/'graph/fresh_noise_comparison.png')

report = {
    'finding':'The original log-normal noise allocation undertrained global/class structure in this small DiT at five epochs.',
    'single_training_change':{'p_mean_before':-1.2,'p_mean_after':0.0,'p_std':1.2},
    'fraction_sigma_above_2':{str(mu):.5*math.erfc((math.log(2)-mu)/(1.2*math.sqrt(2))) for mu in (-1.2,0.)},
    'unchanged':{'parameters':2834512,'epochs':5,'updates':4690,'lr':1e-4,'batch_size':64,'seed':42},
    'fresh_noise_regression':{'count':500,'seed':19873,'original_accuracy':original['metrics']['accuracy'],
                             'balanced_accuracy':balanced['metrics']['accuracy']},
    'evaluator_validation_accuracy':balanced['evaluator_validation_accuracy'],
    'original_sampler_ablation_accuracy_100':{'cfg2_32':.07,'cfg1_32':.07,'cfg2_128':.07,'sigma_max10':.08},
    'noise_embedding_scale100_5epochs_accuracy_100':.1,
    'unet_parameters':56960545,
    'tests':{'unit_and_integration_passed':53,'old_generation_check':'failed as expected','balanced_generation_check':'passed'},
    'limitations':['Classifier agreement is not FID and not proof of image quality.',
                   'Single training seed; some balanced samples still have wrong or fragmented digits.',
                   'Changing the sigma distribution changes the averaged loss task; do not directly compare loss magnitudes.',
                   'The result identifies a useful causal configuration change, not a unique or globally optimal recipe.'],
}
save_json(report,destination/'diagnosis.json')
print(json.dumps(report,indent=2))
