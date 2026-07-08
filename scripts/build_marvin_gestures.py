"""Procedurally animate ALL of Marvin's gestures from one neutral frame
(center.png), guaranteeing every clip starts AND ends exactly at the main
pose so transitions never pop. Each gesture is a set of per-eye transforms
(scale x/y, translate x/y, brightness) whose curves are identity at u=0 and
u=1. Eased with smoothstep + subtle overshoot for charm.

Usage: eye_anim2.py <mode> <out_dir>
modes: blink curious nod shake skeptic wake spin
"""
import sys, os, json, math
import numpy as np
from PIL import Image, ImageFilter

N = 40
NEUTRAL = "/Users/skonheten/Documents/WISPER_FLOW/assets/marvin/center.png"

def smooth(x): x=max(0.0,min(1.0,x)); return x*x*(3-2*x)
def bump(u, a, b):
    """0 → 1 (by a) → hold → 1 → 0 (by 1-b). A held pulse, 0 at both ends."""
    if u < a:      return smooth(u/a)
    if u > 1-b:    return smooth((1-u)/b)
    return 1.0

def green_mask(a):
    r,g,b=a[...,0].astype(int),a[...,1].astype(int),a[...,2].astype(int); al=a[...,3].astype(int)
    return (g>150)&(g-r>35)&(g-b>45)&(al>40)

def dilate(mask,r=2):
    im=Image.fromarray((mask*255).astype(np.uint8),"L").filter(ImageFilter.MaxFilter(2*r+1))
    return np.asarray(im)>127

def split_eyes(mask):
    ys,xs=np.nonzero(mask); midx=(xs.min()+xs.max())/2.0; out=[]
    for sel in (xs<midx,xs>=midx):
        ex,ey=xs[sel],ys[sel]
        out.append({"x0":int(ex.min()),"x1":int(ex.max()),"y0":int(ey.min()),
                    "y1":int(ey.max()),"cx":float(ex.mean()),"cy":float(ey.mean())})
    return out

def build_eyeless(img,dm):
    """dm = dilated eye-region mask (already covers the dark triangle border)."""
    a=np.asarray(img.convert("RGBA")).copy(); H,W=dm.shape
    for x in range(W):
        col=dm[:,x]
        if not col.any(): continue
        for y in np.nonzero(col)[0]:
            yy=y
            while yy>0 and dm[yy,x]: yy-=1
            a[y,x,:3]=a[yy,x,:3]
    return Image.fromarray(a,"RGBA")

def eye_layer(img,eye,region):
    """Extract the whole eye — green fill AND its dark outline — via the
    region mask, so the eye keeps its border when it moves/scales."""
    a=np.asarray(img.convert("RGBA")); pad=6
    x0,x1=max(0,eye["x0"]-pad),min(a.shape[1],eye["x1"]+pad+1)
    y0,y1=max(0,eye["y0"]-pad),min(a.shape[0],eye["y1"]+pad+1)
    sub=np.zeros((y1-y0,x1-x0,4),np.uint8); m=region[y0:y1,x0:x1]
    sub[m]=a[y0:y1,x0:x1][m]; sub[m,3]=255
    return Image.fromarray(sub,"RGBA"),x0,y0

def bright(layer,f):
    a=np.asarray(layer).astype(np.float32); a[...,:3]=np.clip(a[...,:3]*f,0,255)
    return Image.fromarray(a.astype(np.uint8),"RGBA")

def curve(mode,u):
    """Return [(sx,sy,dx,dy,bri)_left, (...)_right]. All identity at u=0,1."""
    ident=(1.0,1.0,0.0,0.0,1.0)
    if mode=="blink":
        # close to a slit + dim, reopen
        if u<0.30: o=1.0
        elif u<0.46: o=1-smooth((u-0.30)/0.16)
        elif u<0.54: o=0.06
        elif u<0.72: o=0.06+0.94*smooth((u-0.54)/0.18)
        else: o=1.0
        e=(1.0,max(0.06,o),0.0,0.0,0.35+0.65*o); return [e,e]
    if mode=="curious":
        a=bump(u,0.35,0.35); s=1+0.28*a
        e=(s,s,0.0,-0.055*a,1+0.35*a); return [e,e]
    if mode=="nod":
        # two-beat downward nod, eyes foreshorten slightly at the bottom
        env=math.sin(math.pi*u)
        dy=0.055*(math.sin(math.pi*u)+0.35*math.sin(3*math.pi*u))/1.35
        sy=1-0.12*env
        e=(1.0,sy,0.0,dy,1.0); return [e,e]
    if mode=="shake":
        # damped left-right-left, 0 at both ends
        dx=0.05*math.sin(3*math.pi*u)*math.sin(math.pi*u)
        e=(1.0,1.0,dx,0.0,1.0); return [e,e]
    if mode=="skeptic":
        h=bump(u,0.28,0.30); drift=0.018*h
        L=(1.0,1-0.55*h,drift,0.0,1.0)      # left narrows to a suspicious slit
        R=(1.0,1-0.12*h,drift,0.0,1.0)      # right barely narrows
        return [L,R]
    if mode=="wake":
        # snap bigger+brighter, settle back with a tiny undershoot
        if u<0.16: p=smooth(u/0.16)
        elif u<0.55: p=1-1.12*smooth((u-0.16)/0.39)   # settle past 0 (undershoot)
        else: p=-0.12+0.12*smooth((u-0.55)/0.45)      # ease back to 0
        s=1+0.34*p; e=(s,s,0.0,-0.02*max(0,p),1+0.55*max(0,p)); return [e,e]
    if mode=="spin":
        # a single quick eye-roll loop (circle back to origin) + slight pulse
        ang=2*math.pi*u; R=0.035
        dx=R*math.sin(ang); dy=R*(math.cos(ang)-1)*0.6
        s=1+0.06*math.sin(math.pi*u)
        e=(s,s,dx,dy,1+0.15*math.sin(math.pi*u)); return [e,e]
    return [ident,ident]

def main():
    mode,out_dir=sys.argv[1],sys.argv[2]; os.makedirs(out_dir,exist_ok=True)
    img=Image.open(NEUTRAL).convert("RGBA")
    if img.size!=(256,256): img=img.resize((256,256),Image.LANCZOS)
    a=np.asarray(img); green=green_mask(a); eyes=split_eyes(green)
    # region = green + its dark triangle border (2px). The base is filled a
    # touch wider (3px) so no ghost survives under a moved eye.
    region=dilate(green,2)
    base=build_eyeless(img,dilate(green,3)); W,H=img.size
    layers=[eye_layer(img,e,region) for e in eyes]
    ejson=[]
    for t in range(N):
        u=t/(N-1); params=curve(mode,u); canvas=base.copy()
        for (lay,ox,oy),eye,(sx,sy,dx,dy,bri) in zip(layers,eyes,params):
            lw,lh=lay.size
            nw,nh=max(1,int(round(lw*sx))),max(1,int(round(lh*sy)))
            L=bright(lay.resize((nw,nh),Image.LANCZOS),bri)
            lcx,lcy=eye["cx"]-ox,eye["cy"]-oy
            tcx,tcy=eye["cx"]+dx*W,eye["cy"]+dy*H
            px=int(round(tcx-lcx*sx)); py=int(round(tcy-lcy*sy))
            canvas.alpha_composite(L,(px,py))
        canvas.save(os.path.join(out_dir,f"frame_{t:02d}.png"))
        aa=np.asarray(canvas); m2=green_mask(aa)
        if m2.sum()>=10:
            e2=split_eyes(m2)
            ejson.append([[e2[0]["cx"]/W,e2[0]["cy"]/H],[e2[1]["cx"]/W,e2[1]["cy"]/H]])
        else:
            ejson.append(ejson[-1] if ejson else
                         [[eyes[0]["cx"]/W,eyes[0]["cy"]/H],[eyes[1]["cx"]/W,eyes[1]["cy"]/H]])
    json.dump(ejson,open(os.path.join(out_dir,"eyes.json"),"w"))
    print(f"{mode}: 40 frames + eyes.json -> {out_dir}")

if __name__=="__main__":
    main()
