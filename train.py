import os, json, time, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DiffusionPipeline
from pathlib import Path

class LoRALinear(nn.Module):
    def __init__(self, orig, rank=8, alpha=1.0):
        super().__init__()
        self.orig=orig; self.scale=alpha/rank
        self.down=nn.Linear(orig.in_features,rank,bias=False)
        self.up=nn.Linear(rank,orig.out_features,bias=False)
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        orig.weight.requires_grad=False
        if orig.bias is not None: orig.bias.requires_grad=False
    def forward(self,x): return self.orig(x)+self.up(self.down(x))*self.scale

def apply_lora(unet, rank=8, alpha=1.0):
    count=0
    for name,mod in unet.named_modules():
        for cn,child in list(mod.named_children()):
            if isinstance(child,nn.Linear) and any(t in cn for t in ["to_q","to_k","to_v","to_out"]):
                setattr(mod,cn,LoRALinear(child,rank,alpha)); count+=1
    return count

def get_lora_params(unet):
    p=[]
    for m in unet.modules():
        if isinstance(m,LoRALinear): p+=list(m.down.parameters())+list(m.up.parameters())
    return p

def save_lora(unet,cfg,path):
    sd={}
    for n,m in unet.named_modules():
        if isinstance(m,LoRALinear):
            sd[n+".down.weight"]=m.down.weight.data.cpu()
            sd[n+".up.weight"]=m.up.weight.data.cpu()
    torch.save({"config":cfg,"lora_state":sd},path)

class ChairDataset(Dataset):
    def __init__(self,root):
        self.root=Path(root)
        self.samples=sorted([d for d in self.root.iterdir()
            if d.is_dir() and (d/"input.png").exists() and (d/"target.png").exists()])
        self.ct=transforms.Compose([transforms.Resize((320,320)),transforms.ToTensor()])
        self.tt=transforms.ToTensor()
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        d=self.samples[i]
        return self.ct(Image.open(d/"input.png").convert("RGB")),self.tt(Image.open(d/"target.png").convert("RGB"))

@torch.no_grad()
def evaluate(pipe,loader,device):
    pipe.unet.eval(); s,n=0,0
    for cond,tgt in loader:
        cond,tgt=cond.to(device),tgt.to(device)
        lat=pipe.vae.encode(tgt).latent_dist.sample()*pipe.vae.config.scaling_factor
        t=torch.randint(0,pipe.scheduler.config.num_train_timesteps,(lat.shape[0],),device=device).long()
        noise=torch.randn_like(lat)
        clat=pipe.vae.encode(cond).latent_dist.sample()*pipe.vae.config.scaling_factor
        pred=pipe.unet(torch.cat([pipe.scheduler.add_noise(lat,noise,t),clat],1),t).sample
        s+=F.mse_loss(pred,noise).item(); n+=1
    return s/n

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",default="./training_data")
    p.add_argument("--output_dir",default="./checkpoints")
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--batch_size",type=int,default=2)
    p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--lora_rank",type=int,default=8)
    p.add_argument("--lora_alpha",type=float,default=1.0)
    a=p.parse_args()
    torch.manual_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.output_dir,exist_ok=True)
    pipe=DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,trust_remote_code=True).to(device)
    cfg={"rank":a.lora_rank,"alpha":a.lora_alpha}
    print(f"LoRA: {apply_lora(pipe.unet,a.lora_rank,a.lora_alpha)} layers")
    ds=ChairDataset(a.data_dir)
    nt=int(len(ds)*0.8); nv=(len(ds)-nt)//2; ne=len(ds)-nt-nv
    g=torch.Generator().manual_seed(42)
    tr,va,te=torch.utils.data.random_split(ds,[nt,nv,ne],generator=g)
    trl=DataLoader(tr,a.batch_size,shuffle=True,num_workers=4)
    val=DataLoader(va,a.batch_size,num_workers=4)
    tel=DataLoader(te,a.batch_size,num_workers=4)
    params=get_lora_params(pipe.unet)
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,len(trl)*a.epochs)
    best=float("inf")
    for ep in range(a.epochs):
        pipe.unet.train(); ls,n=0,0
        for i,(cond,tgt) in enumerate(trl):
            cond,tgt=cond.to(device),tgt.to(device)
            with torch.no_grad():
                lat=pipe.vae.encode(tgt).latent_dist.sample()*pipe.vae.config.scaling_factor
            t=torch.randint(0,pipe.scheduler.config.num_train_timesteps,(lat.shape[0],),device=device).long()
            noise=torch.randn_like(lat)
            noisy=pipe.scheduler.add_noise(lat,noise,t)
            with torch.no_grad():
                clat=pipe.vae.encode(cond).latent_dist.sample()*pipe.vae.config.scaling_factor
            loss=F.mse_loss(pipe.unet(torch.cat([noisy,clat],1),t).sample,noise)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            opt.step(); sch.step(); ls+=loss.item(); n+=1
            if i%50==0: print(f"Ep[{ep+1}/{a.epochs}] B[{i}/{len(trl)}] {loss.item():.4f}")
        vl=evaluate(pipe,val,device)
        print(f"Ep[{ep+1}] Train:{ls/n:.4f} Val:{vl:.4f}{"*BEST*" if vl<best else ""}")
        if vl<best: best=vl; save_lora(pipe.unet,cfg,f"{a.output_dir}/zero123_lora_best.pt")
    print(f"Test:{evaluate(pipe,tel,device):.4f} BestVal:{best:.4f}")

if __name__=="__main__": main()