import numpy as np
fs=100; T=2.0; t=np.arange(int(fs*T))/fs
ss=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
def kampita(rate,width): return width*(0.5-0.5*np.cos(2*np.pi*rate*t))
def jaru(dur,depth,t0=0.4): return -depth*ss((t-t0)/dur)
def sphur(gap,depth,sig,t0=0.6): return -depth*(np.exp(-((t-t0)/sig)**2)+np.exp(-((t-t0-gap)/sig)**2))
sig={'K1 kampita 4 Hz, 100c':kampita(4,100),'K2 kampita 6 Hz, 250c':kampita(6,250),
     'J1 jaru 0.5 s, 500c':jaru(0.5,500),'J2 jaru 0.8 s, 300c':jaru(0.8,300),
     'S1 sphuritam fast, 150c':sphur(0.15,150,0.02),'S2 sphuritam 1.4x slower, 220c':sphur(0.21,220,0.028)}
names=list(sig)
# log-frequency magnitude spectrum, normalised (removes width)
N=2048; f=np.fft.rfftfreq(N,1/fs); G=np.exp(np.linspace(np.log(0.5),np.log(20),240)); dl=np.log(G[1]/G[0])
def logspec(x):
    x=x-x.mean(); X=np.abs(np.fft.rfft(x*np.hanning(len(x)),N)); v=np.interp(G,f,X); return v/np.linalg.norm(v)
def shift_sim(a,b,maxratio=1.6):
    best,bs=-1,0; m=int(round(np.log(maxratio)/dl))
    for s in range(-m,m+1):
        A=a[max(0,s):len(a)+min(0,s)]; B=b[max(0,-s):len(b)+min(0,-s)]
        c=A@B/(np.linalg.norm(A)*np.linalg.norm(B)+1e-12)
        if c>best: best,bs=c,s
    return best,np.exp(bs*dl)
def time_sim(a,b):  # best normalised cross-correlation in time, no Fourier
    a=(a-a.mean())/(np.linalg.norm(a-a.mean())+1e-12); b=(b-b.mean())/(np.linalg.norm(b-b.mean())+1e-12)
    return np.max(np.correlate(a,b,'full'))
S={k:logspec(v) for k,v in sig.items()}
short=[n.split()[0] for n in names]
def show(title,fn):
    print(title); print('      '+' '.join(f'{s:>5}' for s in short))
    for i,a in enumerate(names): print(f'{short[i]:>5} '+' '.join(f'{fn(a,b):5.2f}' for b in names))
    print()
show('A) Time-domain matching (no Fourier), best alignment:',lambda a,b:time_sim(sig[a],sig[b]))
show('B) Fourier: normalised log-frequency spectrum, best shift within 1.6x:',lambda a,b:shift_sim(S[a],S[b])[0])
for a,b in [(0,1),(2,3),(4,5)]:
    c,r=shift_sim(S[names[a]],S[names[b]]); print(f'{names[a]}  vs  {names[b]}: similarity {c:.2f}, estimated speed ratio {r:.2f}')
