import os, argparse, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from diffusers import DiffusionPipeline

class LoRALinear(nn.Module):
    def __init__(self,orig,rank=8,alpha=1.0):
        super().__init__()
        self.orig=orig; self.scale=alpha/rank
        self.down=nn.Linear(orig.in_features,rank,bias=False)
        self.up=nn.Linear(rank,orig.out_features,bias=False)
        orig.weight.requires_grad=False
        if orig.bias is not None: orig.bias.requires_grad=False
    def forward(self,x): return self.orig(x)+self.up(self.down(x))*self.scale

def apply_lora(unet,rank=8,alpha=1.0):
    for nm,mod in unet.named_modules():
        for cn,child in list(mod.named_children()):
            if isinstance(child,nn.Linear) and any(t in cn for t in ["to_q","to_k","to_v","to_out"]):
                setattr(mod,cn,LoRALinear(child,rank,alpha))

def load_lora(unet,sd):
    for name,m in unet.named_modules():
        if isinstance(m,LoRALinear):
            if name+".down.weight" in sd: m.down.weight.data=sd[name+".down.weight"].to(m.down.weight.device,dtype=m.down.weight.dtype)
            if name+".up.weight" in sd: m.up.weight.data=sd[name+".up.weight"].to(m.up.weight.device,dtype=m.up.weight.dtype)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data_dir",required=True)
    p.add_argument("--output_dir",default="./test_results")
    p.add_argument("--num_samples",type=int,default=10)
    p.add_argument("--steps",type=int,default=28)
    a=p.parse_args()
    os.makedirs(a.output_dir,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    pipe=DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,trust_remote_code=True).to(device)
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    apply_lora(pipe.unet,ckpt["config"]["rank"],ckpt["config"]["alpha"])
    load_lora(pipe.unet,ckpt["lora_state"])
    pipe.unet=pipe.unet.half()
    samples=sorted([d for d in Path(a.data_dir).iterdir() if d.is_dir() and (d/"input.png").exists()])
    for i,d in enumerate(samples[:a.num_samples]):
        with torch.no_grad():
            r=pipe(Image.open(d/"input.png").convert("RGB"),num_inference_steps=a.steps).images[0]
        r.save(f"{a.output_dir}/pred_{d.name}.png")
        print(f"[{i+1}] {d.name} saved")
    print(f"Done! {a.output_dir}/")

if __name__=="__main__": main()