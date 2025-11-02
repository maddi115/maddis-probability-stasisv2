#!/usr/bin/env python3
# Nemotron + Probability Stasis v7 Neon (Pure #05D9FF glow)

import os, sys, time, math, threading, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util
from colorama import Style, init
init(autoreset=True)

try:
    import pynvml as nv
    nv.nvmlInit(); _NV = True
except Exception:
    _NV = False

MODEL="Llama-3.1-Nemotron-Nano-4B-v1.1-Q6_K.gguf"; CTX=4096; LAYERS=40
TEMPS=[0.62,0.72,0.82]; WORKERS=3; SEED=42
SYS="""You are Nemotron, a factual reasoning model. Answer clearly and briefly."""

# === Neon constants ===
NEON_HEX = (5, 217, 255)   # #05D9FF
_rgb = lambda r,g,b: f"\033[38;2;{r};{g};{b}m"
RESET = Style.RESET_ALL

def hue(s: float):
    # Keep brightness modulation around the neon base
    s = max(0.0, min(1.0, s))
    r = int(NEON_HEX[0] + s * 30)
    g = int(NEON_HEX[1] + s * 20)
    b = int(NEON_HEX[2])
    return _rgb(r,g,b)

def badge(s): return "🧩 STABLE" if s>0.8 else "⚖️ MIXED" if s>0.55 else "🌪️ CHAOTIC"
def bar(s,w=22): f=int(w*max(0,min(1,s))); return f"{_rgb(*NEON_HEX)}"+("█"*f+"░"*(w-f))+RESET

class Pulse:
    def __init__(self): self.stop = mp.Event()
    def _gpu(self):
        if not _NV: return ""
        try:
            h = nv.nvmlDeviceGetHandleByIndex(0)
            u = nv.nvmlDeviceGetUtilizationRates(h).gpu
            m = nv.nvmlDeviceGetMemoryInfo(h)
            return f"GPU {u:>2d}% | VRAM {m.used/1e9:.1f}/{m.total/1e9:.1f} GB"
        except Exception:
            return ""
    def run(self):
        frames = ["▱▱▱▱▱","▰▱▱▱▱","▰▰▱▱▱","▰▰▰▱▱","▰▰▰▰▱","▰▰▰▰▰"]
        i=0
        while not self.stop.is_set():
            phase = (math.sin(time.time()*3.0)+1)/2
            intensity = 0.6 + 0.4*phase
            glow = _rgb(int(NEON_HEX[0]*intensity),
                        int(NEON_HEX[1]*intensity),
                        int(NEON_HEX[2]*intensity))
            sys.stdout.write(f"\r{glow}⚙️ Running paths... {frames[i%6]} {self._gpu()}{RESET}")
            sys.stdout.flush()
            time.sleep(0.1)
            i+=1
        sys.stdout.write("\r"+" "*100+"\r")
        sys.stdout.flush()

@dataclass
class Out: temp: float; text: str

_LLM=None
def _init(): 
    global _LLM; _LLM=None
def _llm():
    global _LLM
    if _LLM is None:
        _LLM=Llama(model_path=MODEL,n_ctx=CTX,n_gpu_layers=LAYERS,seed=SEED,verbose=False)
    return _LLM
def _run(prompt,t):
    r=_llm()(f"{SYS}\nUser:{prompt}\nAssistant:",temperature=t,max_tokens=256,stop=["User:"],top_p=0.95)
    txt=r["choices"][0]["text"].strip() if isinstance(r,dict) else str(r).strip()
    return Out(t,txt.split("<think>")[0].split("Assistant:")[-1].strip())

_embed=None; _cache={}
def emb(t):
    global _embed
    k=("e",hash(t))
    if k in _cache: return _cache[k]
    if _embed is None: _embed=SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
    e=_embed.encode(t,normalize_embeddings=True); _cache[k]=e
    if len(_cache)>256: _cache.pop(next(iter(_cache)))
    return e
def stab(outs):
    if len(outs)==1:return 1.0
    import numpy as np
    em=[emb(o.text)for o in outs]; s=[]
    for i in range(len(em)):
        for j in range(i+1,len(em)):
            s.append(float(util.cos_sim(em[i],em[j]).item()))
    a=np.array(s); return float(a.mean()-a.std())
def pick(outs):
    import numpy as np
    E=[emb(o.text)for o in outs]; M=np.vstack(E)
    c=M.mean(0,keepdims=True); sims=(M@c.T).ravel(); idx=int(sims.argmax())
    return outs[idx]

def show(ans,sc):
    print(f"\n🏆 Stability [{hue(sc)}{sc:.3f}{RESET}] {bar(sc)} {badge(sc)}\n")
    print("╭"+"─"*46+"╮"); print("Nemotron says:"); print(ans[:1000]); print("╰"+"─"*46+"╯"); print("-"*60)

def main():
    print(f"{_rgb(*NEON_HEX)}🤖 Nemotron + Probability Stasis v7 Neon (#05D9FF Glow){RESET}\n")
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=_init)as ex:
        while True:
            try: p=input("💬 You: ").strip()
            except(EOFError,KeyboardInterrupt): print(); break
            if not p or p.lower()in{"quit","exit"}: break
            pulse=Pulse(); t=threading.Thread(target=pulse.run,daemon=True); t.start()
            fut=[ex.submit(_run,p,x)for x in TEMPS]; outs=[]
            for f in as_completed(fut):
                try: outs.append(f.result())
                except Exception: pass
            pulse.stop.set(); t.join()
            sc=stab(outs); best=pick(outs)
            show(best.text,max(0,min(1,(sc+1)/2)))

if __name__=="__main__":
    try: main()
    finally:
        if _NV:
            try: nv.nvmlShutdown()
            except Exception: pass
