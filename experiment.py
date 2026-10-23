"""KBS research prototype. Run: python experiment.py --csv flows.csv [--out results]."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss,average_precision_score,roc_auc_score
SEED=42; CFN=30.; CFP=1.
def data(path,demo,max_rows):
    if demo:
        x,y=make_classification(n_samples=8500,n_features=18,n_informative=8,n_redundant=4,weights=[.91,.09],flip_y=.025,random_state=SEED)
        return pd.DataFrame(x),y,"SYNTHETIC DEMO—NOT PUBLISHABLE RESULTS"
    if not path:raise ValueError("Use --csv actual_CIC.csv or --demo")
    f=pd.read_csv(path,low_memory=False);f.columns=f.columns.str.strip()
    l=next((c for c in f if c.lower() in ("label","class","target")),None)
    if not l:raise ValueError("Missing Label column")
    y=(~f[l].astype(str).str.strip().str.lower().isin(["benign","normal","0","false"])).astype(int).values
    bad={"flow id","source ip","destination ip","timestamp","src ip","dst ip",l.lower()}
    x=f[[c for c in f if c.lower() not in bad]].select_dtypes(include="number").replace([np.inf,-np.inf],np.nan)
    x=x.loc[:,~x.columns.duplicated()].dropna(axis=1,how="all");x=x.fillna(x.median()).fillna(0)
    if max_rows and len(x)>max_rows:
        keep,_=train_test_split(np.arange(len(y)),train_size=max_rows,random_state=SEED,stratify=y)
        x=x.iloc[keep].reset_index(drop=True);y=y[keep]
    if len(set(y))<2:raise ValueError("Both benign and malicious examples required")
    return x,y,str(Path(path).resolve())
def ece(y,p):
    ids=np.minimum(np.digitize(p,np.linspace(0,1,16)[1:-1]),14)
    return float(sum(np.mean(ids==k)*abs(y[ids==k].mean()-p[ids==k].mean()) for k in range(15) if (ids==k).any()))
def decide(p,mode,cap=.12):
    if mode=="fixed":return p>=.5
    gain=p*CFN-(1-p)*CFP
    if mode=="cost":return gain>0
    eligible=np.flatnonzero(gain>0);k=min(len(eligible),int(np.floor(cap*len(p))))
    mask=np.zeros(len(p),bool);mask[eligible[np.argsort(-gain[eligible],kind="mergesort")[:k]]]=True
    return mask
def score(y,p,a):
    fn=((y==1)&~a).sum();fp=((y==0)&a).sum()
    return dict(cost_per_alert=float((CFN*fn+CFP*fp)/len(y)),miss_rate=float(fn/max(1,(y==1).sum())),
        escalation=float(a.mean()),brier=float(brier_score_loss(y,p)),ece=ece(y,p),
        average_precision=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--csv");ap.add_argument("--demo",action="store_true")
    ap.add_argument("--out",default="results");ap.add_argument("--max-rows",type=int,default=120000);a=ap.parse_args()
    x,y,src=data(a.csv,a.demo,a.max_rows)
    xt,xh,yt,yh=train_test_split(x,y,test_size=.4,stratify=y,random_state=SEED)
    xc,xv,yc,yv=train_test_split(xh,yh,test_size=.5,stratify=yh,random_state=SEED)
    m=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=SEED).fit(xt,yt)
    pc=m.predict_proba(xc)[:,1];raw=m.predict_proba(xv)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip").fit(pc,yc).predict(raw)
    platt=LogisticRegression(random_state=SEED).fit(pc.reshape(-1,1),yc).predict_proba(raw.reshape(-1,1))[:,1]
    policies={"raw_fixed":(raw,"fixed"),"raw_cost":(raw,"cost"),"isotonic_cost":(iso,"cost"),
        "platt_cost":(platt,"cost"),"isotonic_budgeted":(iso,"budgeted"),"platt_budgeted":(platt,"budgeted")}
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame([dict(policy=n,**score(yv,p,decide(p,k))) for n,(p,k) in policies.items()]).to_csv(out/"table_1_policy_results.csv",index=False)
    pd.DataFrame([dict(probability=n,brier=brier_score_loss(yv,p),ece=ece(yv,p)) for n,p in
        [("raw",raw),("platt",platt),("isotonic",iso)]]).to_csv(out/"table_2_calibration.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4.4));ax.plot([0,1],[0,1],"--",color="gray",label="Ideal")
    for name,p in [("Raw",raw),("Isotonic",iso)]:
        bins=np.minimum(np.digitize(p,np.linspace(0,1,11)[1:-1]),9)
        pts=[(p[bins==i].mean(),yv[bins==i].mean()) for i in range(10) if (bins==i).any()]
        ax.plot(*zip(*pts),marker="o",label=name)
    ax.set(xlabel="Predicted attack probability",ylabel="Observed attack fraction",xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(out/"Figure_1_calibration.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.4))
    for name,p in [("Raw",raw),("Isotonic",iso),("Platt",platt)]:
        q=np.linspace(.02,.30,15);cost=[score(yv,p,decide(p,"budgeted",float(z)))["cost_per_alert"] for z in q]
        ax.plot(q,cost,label=name,marker=".")
    ax.set(xlabel="Escalation capacity fraction",ylabel="Simulated decision cost / alert");ax.legend()
    fig.tight_layout();fig.savefig(out/"Figure_2_capacity_cost.pdf");plt.close(fig)
    (out/"run_metadata.json").write_text(json.dumps(dict(source=src,synthetic_demo=a.demo,
        random_seed=SEED,n_total=len(y),n_train=len(yt),n_calibration=len(yc),n_test=len(yv),
        false_negative_cost=CFN,false_positive_cost=CFP,
        caution="Costs are assumed utility units; synthetic demonstration is not real-world evidence."),indent=2))
    print(pd.read_csv(out/"table_1_policy_results.csv").round(4).to_string(index=False))
if __name__=="__main__":main()
