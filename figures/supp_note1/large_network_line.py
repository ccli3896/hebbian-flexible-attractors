"""Supplementary Note 1, section 14: the line X = x0*theta in the large (1-50-2) PC network.

Simulates the large PC network at its reward-seeking operating point (alpha=0.0164,
beta=0.0072, eta=0.01; Supplementary Table 2) on the deterministic bandit, following the
model equations in Methods, and records (x0, theta, X) every 10th trial over the last
40,000 trials. theta = sum_k (w_2k - w_1k) * w_k is the effective weight difference
through the hidden layer. Run: python large_network_line.py  ->  big.npz
"""
import numpy as np, sys
def run(seeds=48, T=100_000, alpha=0.0164, beta=0.0072, eta=0.01, H=50, r=1.0, rec_from=60_000, rec_every=10, seed0=0):
    rng=np.random.default_rng(seed0)
    S=seeds
    x0=rng.uniform(0,1,S); h=rng.uniform(0,1,(S,H)); y=rng.uniform(0,1,(S,2))
    Wh=rng.uniform(0,1,(S,H)); Wy=rng.uniform(0,1,(S,2,H))
    act=np.argmax(y,1)
    pc=np.zeros((S,T//100)); rec=[]
    for t in range(T):
        s=np.where(act==0, r, 0.0)               # arm a = output 0 pays r
        ph=Wh*x0[:,None]; py=np.einsum('skh,sh->sk',Wy,h)
        nx0=x0+2*alpha*np.clip(s-x0,-5,5); nh=h+2*alpha*np.clip(ph-h,-5,5); ny=y+2*alpha*np.clip(py-y,-5,5)
        x0=np.maximum(0,nx0+eta*rng.standard_normal(S)); h=np.maximum(0,nh+eta*rng.standard_normal((S,H))); y=np.maximum(0,ny+eta*rng.standard_normal((S,2)))
        eh=h-Wh*x0[:,None]; ey=y-np.einsum('skh,sh->sk',Wy,h)
        Wh=np.maximum(0,Wh+beta*np.clip(eh*x0[:,None],-10,10))
        Wy=np.maximum(0,Wy+beta*np.clip(ey[:,:,None]*h[:,None,:],-10,10))
        pc[:,t//100]+= (act==0)/100
        act=np.argmax(y,1)
        if t>=rec_from and t%rec_every==0:
            theta=np.einsum('sh,sh->s',Wy[:,0]-Wy[:,1],Wh)
            rec.append(np.stack([x0, theta, y[:,0]-y[:,1]],1))
    return pc, np.array(rec)
if __name__=='__main__':
    pc,rec=run()
    np.savez('big.npz',pc=pc,rec=rec)
    print('P(correct) last 20k, mean over seeds:', pc[:,-200:].mean().round(3), 'per-seed>0.9:', (pc[:,-200:].mean(1)>0.9).sum(),'/',pc.shape[0])
