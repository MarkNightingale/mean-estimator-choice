import pandas as pd, numpy as np
sc=pd.read_csv('results/scenarios2.csv'); r=pd.read_csv('results/ml2_results.csv')
EST=['mean','median','trim10','huber','hl','hogg']
fx=[]
for e in EST:
    t=sc[['sid','dist','n','eps','otype','k','mse_oracle']].copy()
    for m in ['mse','bias','mae','big']: t[m]=sc[f'{m}_{e}']
    t['method']=e; fx.append(t)
fx=pd.concat(fx)
def summ(d):
    d=d.assign(rel=d.mse/d.mse_oracle, grp=np.where(d.eps>0,'cont','clean'))
    out={}
    for m,dd in d.groupby('method'):
        c=dd[dd.grp=='clean']; k=dd[dd.grp=='cont']
        out[m]=dict(rel_all=dd.rel.mean(), max_all=dd.rel.max(), rel_clean=c.rel.mean(), rel_cont=k.rel.mean(),
                    rel_eq=(c.rel.mean()+k.rel.mean())/2, big_all=dd.big.mean()*100, big_clean=c.big.mean()*100, big_cont=k.big.mean()*100, mae=dd.mae.mean())
    return pd.DataFrame(out).T.round(3)
pd.set_option('display.width',250)
if __name__=='__main__':
    for st,train,test in [('A',['normal','t3'],['laplace','t2','logistic']),('B',['normal','t3','laplace'],['t2','logistic'])]:
        for part,dists in [('train-CV',train)]+[(f'OOD-{d}',[d]) for d in test]:
            ml=r[(r.setting==st)&(r.dist.isin(dists))]
            f=fx[fx.dist.isin(dists)&fx.method.isin(['mean','median','huber','hl','hogg'])]
            print(f'===== {st} {part}'); print(summ(pd.concat([f,ml[f.columns]])))
