import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
#oath of data
test=r"C:\Users\ballo\OneDrive\Desktop\Code\Independent Study\Version 20 Image Upscaling\mouse.jpg"
_HERE  = os.path.dirname(os.path.abspath(__file__))
SCALE  =2    # upscale factor
N_DEG  = 120   # rational function degree
SRC= os.path.join(_HERE, "demo_src.png")
OUT= os.path.join(_HERE, "demo_upscaled.png")
PLOT= os.path.join(_HERE, "demo_result.png")

#Scaled Null Space Rational Fitting


def _eval_rational(coeffs, s, N):
    
    #Evaluate the rational function at points s given coefficient vector c.
    a=coeffs[:N+1]    # numerator
    b=coeffs[N+1:]    # denominator
    s_pow=np.array([s**j for j in range(N+1)])
    num=a@s_pow
    den=b@s_pow
    # guard against near-zero denominator
    den=np.where(np.abs(den)<1e-14, np.sign(den+1e-300)*1e-14, den)
    return num/den

def fit_rational_1d(x, y, N, x_eval):
    
    #Fit a degree-N rational function to data (x, y) using the scaled null space method, then evaluate at x_eval.
    #Returns 1-D array of fitted values at x_eval.
    smax=float(np.max(np.abs(x))) if np.max(np.abs(x))!=0 else 1.0
    s=x/smax
    se=x_eval/smax

    # build A: numerator coefficients first, denominator second
    A=np.zeros((len(s), 2*N+2))
    for j in range(N+1):
        A[:,j]=-(s**j)         # numerator 
        A[:,N+1+j]=y*(s**j)    # denominator 

    #null space via SVD
    _,sv,Vt=np.linalg.svd(A, full_matrices=True)
    sv_max=sv[0] if len(sv)>0 else 1.0
    rank=int(np.sum(sv>1e-10*sv_max))
    n_cols=Vt.shape[0]
    if rank>=n_cols:
        ns=Vt[-1:].T   # best available: last row of Vt
    else:
        ns=Vt[rank:].T # columns = null space basis vectors

    # evaluate ALL null space candidates on training points, pick by min-max error
    n_cands=ns.shape[1]
    ysv=np.zeros((n_cands, len(x)))
    for j in range(n_cands):
        ysv[j]=_eval_rational(ns[:,j], s, N)

    mynorm=np.max(np.abs(ysv-y), axis=1)   # worst error per candidate
    I=np.argmin(mynorm)                     # best candidate index

    return _eval_rational(ns[:,I], se, N)

#Image support function

def _clamp(arr):
    #float array to uint8 range [0,255]
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)

def _load_rgb(path):
   #Load image as float32 RGB array (H, W, 3)"
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32)

def _save_rgb(arr, path):
    #Save float (H, W, 3) array as PNG
    Image.fromarray(np.stack([_clamp(arr[:,:,c]) for c in range(3)], axis=2)).save(path)

def _channel_upscale(ch, scale, N):
    #Pass 1 — horizontal: fit each row of the original, evaluate at scale*W points.
    #Pass 2 — vertical:   fit each column of the pass-1 result, evaluate at scale*H points.
   
    H,W=ch.shape
    W2,H2=W*scale, H*scale

    # pass 1: row
    x_src=np.arange(W, dtype=float)
    x_dst=np.linspace(0, W-1, W2)
    mid=np.zeros((H, W2), dtype=np.float32)
    for r in range(H):
        mid[r]=fit_rational_1d(x_src, ch[r].astype(float), N, x_dst)

    # pass 2: column
    y_src=np.arange(H, dtype=float)
    y_dst=np.linspace(0, H-1, H2)
    out=np.zeros((H2, W2), dtype=np.float32)
    for c in range(W2):
        out[:,c]=fit_rational_1d(y_src, mid[:,c].astype(float), N, y_dst)

    return out


#Demo using synthetic image

def _run_demo():
    print("=== Scaled Null Space Rational Image Upscaling ===")
    print(f"  Rational degree N = {N_DEG}")
    print(f"  Scale factor      = {SCALE}x\n")

    # build synthetic 64x64 test image
    # H,W=256,256
    # xx,yy=np.meshgrid(np.linspace(0,1,W), np.linspace(0,1,H))
    # r_ch=(128+80*np.sin(2*np.pi*xx*3)).astype(np.float32)
    # g_ch=(100+80*np.cos(2*np.pi*yy*3)).astype(np.float32)
    # b_ch=(80 +60*np.sin(2*np.pi*(xx+yy)*2)).astype(np.float32)
    # img=np.stack([r_ch,g_ch,b_ch], axis=2)
    # _save_rgb(img, SRC)
    # print(f"Source image: {W}x{H}  saved → {SRC}")

    #load external image instead of synthetic ---
    
    img = _load_rgb(test)
    H, W, _ = img.shape
    print(f"Loaded image: {W}x{H}  from → {test}")

    # upscale
    t0=time.time()
    ch_times=[]
    channels=[]
    for ch_i in range(3):
        tc=time.time()
        channels.append(_channel_upscale(img[:,:,ch_i], SCALE, N_DEG))
        ch_times.append(time.time()-tc)
        print(f"  Channel {ch_i+1}/3 done in {ch_times[-1]:.1f}s")

    out_img=np.stack(channels, axis=2)
    total_time=time.time()-t0
    _save_rgb(out_img, OUT)

    # stats vs bicubic
    bicubic=np.array(Image.open(test).convert('RGB').resize((W*SCALE,H*SCALE), Image.BICUBIC), dtype=float)     
    diff=np.abs(out_img.astype(float)-bicubic)
    mse_vs_bic=np.mean(diff**2)
    psnr_vs_bic=10*np.log10(255**2/mse_vs_bic) if mse_vs_bic>0 else float('inf')

    print(f"\n--- Results ---")
    print(f"  Output size          : {W*SCALE}x{H*SCALE}")
    print(f"  Total time           : {total_time:.1f}s")
    print(f"  Max pixel diff vs bicubic : {diff.max():.2f}")
    print(f"  Mean pixel diff vs bicubic: {diff.mean():.2f}")
    print(f"  PSNR vs bicubic      : {psnr_vs_bic:.1f} dB")
    print(f"  Output saved         → {OUT}")

    # before / after plot
    fig,axes=plt.subplots(1,2, figsize=(10,5))
    fig.suptitle(f"Rational Upscaling (N={N_DEG}, {SCALE}x)", fontsize=12, fontweight='bold')
    axes[0].imshow(_clamp(img))
    axes[0].set_title(f"Before  {W}x{H}")
    axes[0].axis('off')
    axes[1].imshow(_clamp(out_img))
    axes[1].set_title(f"After  {W*SCALE}x{H*SCALE}")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(PLOT, dpi=120)
    print(f"  Plot saved           → {PLOT}")
    plt.show()

if __name__=='__main__':
    _run_demo()