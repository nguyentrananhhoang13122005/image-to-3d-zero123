# === Monkey-patch torchmcubes -> mcubes ===
import sys, types
import mcubes as _mcubes
import numpy as _np

_fake = types.ModuleType('torchmcubes')
def _mc(field, threshold):
    import torch
    device = field.device
    v, f = _mcubes.marching_cubes(field.cpu().numpy(), threshold)
    return (torch.from_numpy(v.copy().astype(_np.float32)).to(device),
            torch.from_numpy(f.copy().astype(_np.int64)).to(device))
_fake.marching_cubes = _mc
sys.modules['torchmcubes'] = _fake

# Fix trimesh + numpy 2.5
import trimesh, trimesh.util
def _fixed_allclose(a, b, atol=1e-8):
    diff = a - b
    return float(diff.max() - diff.min()) < atol
trimesh.util.allclose = _fixed_allclose

import torch, torch.nn as nn, gradio as gr, spaces, tempfile, os, rembg, warnings
from PIL import Image
from diffusers import DiffusionPipeline
warnings.filterwarnings('ignore', message='.*dtype=torch.float16.*')

# === KEY REMAPPING: old transformers -> new transformers ===
def remap_checkpoint_keys(sd):
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        if 'image_tokenizer.model.encoder.layer.' in k:
            new_k = k.replace('image_tokenizer.model.encoder.layer.',
                              'image_tokenizer.model.layers.')
            new_k = new_k.replace('.attention.attention.query.', '.attention.q_proj.')
            new_k = new_k.replace('.attention.attention.key.', '.attention.k_proj.')
            new_k = new_k.replace('.attention.attention.value.', '.attention.v_proj.')
            new_k = new_k.replace('.attention.output.dense.', '.attention.o_proj.')
            new_k = new_k.replace('.intermediate.dense.', '.mlp.fc1.')
            new_k = new_k.replace('.output.dense.', '.mlp.fc2.')
        new_sd[new_k] = v
    return new_sd

# === LoRA ===
class LoRALinear(nn.Module):
    def __init__(self, orig, rank=8, alpha=1.0):
        super().__init__()
        self.orig = orig; self.scale = alpha / rank
        self.down = nn.Linear(orig.in_features, rank, bias=False)
        self.up = nn.Linear(rank, orig.out_features, bias=False)
        orig.weight.requires_grad = False
        if orig.bias is not None: orig.bias.requires_grad = False
    def forward(self, x):
        return self.orig(x) + self.up(self.down(x)) * self.scale

def apply_lora(unet, rank=8, alpha=1.0):
    for name, mod in unet.named_modules():
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and any(t in cn for t in ['to_q','to_k','to_v','to_out']):
                setattr(mod, cn, LoRALinear(child, rank, alpha))

def load_lora_weights(unet, sd):
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            if name+'.down.weight' in sd:
                module.down.weight.data = sd[name+'.down.weight'].to(module.down.weight.device, dtype=module.down.weight.dtype)
            if name+'.up.weight' in sd:
                module.up.weight.data = sd[name+'.up.weight'].to(module.up.weight.device, dtype=module.up.weight.dtype)

# === Load Zero123++ ===
print('Loading Zero123++...')
pipe = DiffusionPipeline.from_pretrained(
    'sudo-ai/zero123plus-v1.2',
    custom_pipeline='sudo-ai/zero123plus-pipeline',
    torch_dtype=torch.float16, trust_remote_code=True,
)
ckpt = torch.load('zero123_lora_best.pt', map_location='cpu', weights_only=False)
apply_lora(pipe.unet, rank=ckpt['config']['rank'], alpha=ckpt['config']['alpha'])
load_lora_weights(pipe.unet, ckpt['lora_state'])
pipe.unet = pipe.unet.half()
print('Zero123++ loaded!')

# === Load TripoSR ===
print('Loading TripoSR...')
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

config_path = hf_hub_download('stabilityai/TripoSR', 'config.yaml')
with open(config_path, 'r') as f:
    raw = f.read()
raw = raw.replace('${tokenizer.num_channels}', '1024')
cfg = OmegaConf.create(raw)

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

triposr = TSR(cfg)
weight_path = hf_hub_download('stabilityai/TripoSR', 'model.ckpt')
sd = torch.load(weight_path, map_location='cpu', weights_only=False)

# REMAP old key names to new key names
sd = remap_checkpoint_keys(sd)
missing, unexpected = triposr.load_state_dict(sd, strict=False)
print(f'After remap: {len(missing)} missing, {len(unexpected)} unexpected (should be ~0!)')
if missing:
    print(f'Still missing: {missing[:5]}')

triposr.renderer.set_chunk_size(131072)
rembg_session = rembg.new_session()
print('=== Both models ready! ===')

def fix_mesh_orientation(mesh):
    verts = _np.array(mesh.vertices, dtype=_np.float64)
    new_verts = verts.copy()
    new_verts[:, 1] = verts[:, 2]
    new_verts[:, 2] = -verts[:, 1]
    mesh.vertices = new_verts
    return mesh

@spaces.GPU(duration=90)
def run(image, steps):
    if image is None:
        return None, None

    pipe.to('cuda')
    triposr.to('cuda')

    with torch.no_grad():
        mv = pipe(image.convert('RGB'), num_inference_steps=int(steps)).images[0]

    pipe.to('cpu')
    torch.cuda.empty_cache()

    img_rgba = remove_background(image.convert('RGB'), rembg_session)
    img_rgba = resize_foreground(img_rgba, 0.85)
    img_arr = _np.array(img_rgba).astype(_np.float32) / 255.0
    img_arr = img_arr[:,:,:3] * img_arr[:,:,3:4] + (1 - img_arr[:,:,3:4]) * 0.5
    img_proc = Image.fromarray((img_arr * 255).astype(_np.uint8))

    with torch.no_grad():
        scene = triposr(img_proc, device='cuda')
        mesh = triposr.extract_mesh(scene, resolution=256)[0]
        mesh = fix_mesh_orientation(mesh)

    glb = tempfile.mktemp(suffix='.glb')
    mesh.export(glb)
    triposr.to('cpu')
    torch.cuda.empty_cache()

    return mv, glb

with gr.Blocks() as demo:
    gr.Markdown('# 🪑 Zero123++ Chair: Image to Multi-View to 3D')
    gr.Markdown('Upload a chair photo. Get **6 views** (our LoRA) + **3D model** (TripoSR) in one shot!')
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label='Input', type='pil')
            steps = gr.Slider(10, 50, value=28, step=1, label='Steps')
            btn = gr.Button('Generate', variant='primary', size='lg')
        with gr.Column(scale=1):
            out_img = gr.Image(label='Multi-View (Zero123++ LoRA)')
        with gr.Column(scale=1):
            out_3d = gr.Model3D(label='3D Model (drag to rotate)')
    gr.Markdown('Zero123++ v1.2 + LoRA (600 chairs) | Val Loss: 0.0194 | 3D: TripoSR bundled')
    btn.click(fn=run, inputs=[inp, steps], outputs=[out_img, out_3d])

demo.launch(ssr_mode=False)
