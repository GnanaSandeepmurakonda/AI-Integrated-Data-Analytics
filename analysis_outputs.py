"""Supplementary analysis for the reconstructed synthetic cohort; not original study output."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from generate_data import FEATURES, RULES, knowledge_features

BASE=Path(__file__).parent
OUT=BASE/'results'

def levers(row,top=3):
    actions={'technology':'Secure cloud foundations and cybersecurity hygiene',
             'organization':'Executive sponsorship and digital workforce upskilling',
             'process':'Process redesign and agile operating practices',
             'data':'Data governance and evidence-based experimentation',
             'ecosystem':'Partner co-innovation and portfolio stage gates'}
    ranked=[]
    for name,(cols,t) in RULES.items():
        score=float(np.mean([row[c] for c in cols]))
        ranked.append((name,max(0.,t-score),actions[name]))
    return sorted(ranked,key=lambda a:-a[1])[:top]

def main():
    OUT.mkdir(exist_ok=True)
    df=pd.read_csv(BASE/'synthetic_enterprise_data.csv')
    X=df[FEATURES]; y=df['ready'].to_numpy(); kb=knowledge_features(X)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    p=cross_val_predict(LogisticRegression(max_iter=2000,class_weight='balanced'),kb,y,cv=cv,method='predict_proba')[:,1]
    rng=np.random.default_rng(42); aucs=[]
    for _ in range(400):
        sample=np.concatenate([rng.choice(np.flatnonzero(y==v),size=sum(y==v),replace=True) for v in (0,1)])
        aucs.append(roc_auc_score(y[sample],p[sample]))
    (OUT/'kb_auc_bootstrap.json').write_text(json.dumps({'point_auc':roc_auc_score(y,p),'bootstrap_95_ci':np.quantile(aucs,[.025,.975]).tolist(),'resamples':400,'method':'stratified bootstrap of out-of-fold predictions'},indent=2))
    scores=pd.DataFrame({name:X[cols].mean(axis=1) for name,(cols,_) in RULES.items()})
    gaps=pd.DataFrame({name:(t-scores[name]).clip(lower=0) for name,(_,t) in RULES.items()})
    fig,ax=plt.subplots(figsize=(8,4.5))
    means=pd.DataFrame({'not ready':gaps[y==0].mean(),'ready':gaps[y==1].mean()})
    means.plot(kind='bar',ax=ax,rot=25)
    ax.set(ylabel='Mean rule gap',title='Synthetic cohort: mean threshold shortfalls');fig.tight_layout()
    fig.savefig(OUT/'figure_1_rule_gaps.png',dpi=180);plt.close(fig)
    hybrid=pd.concat([X,kb],axis=1)
    xtr,xte,ytr,yte=train_test_split(hybrid,y,test_size=.30,random_state=42,stratify=y)
    m=LogisticRegression(max_iter=2000,class_weight='balanced').fit(xtr,ytr)
    imp=permutation_importance(m,xte,yte,n_repeats=20,random_state=42,scoring='roc_auc')
    order=np.argsort(imp.importances_mean)[-12:]
    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.barh(np.array(hybrid.columns)[order],imp.importances_mean[order],xerr=imp.importances_std[order])
    ax.set(xlabel='Mean decrease in holdout ROC-AUC',title='Synthetic cohort: permutation importance');fig.tight_layout()
    fig.savefig(OUT/'figure_2_permutation_importance.png',dpi=180);plt.close(fig)
    examples=[]
    for i in [int(np.argmin(abs(p-np.quantile(p,q)))) for q in (.2,.5,.8)]:
        examples.append({'enterprise_id':f'E{i:04d}','observed_ready':int(y[i]),'kb_probability':float(p[i]),'top_3_levers':levers(X.iloc[i].to_dict())})
    (OUT/'illustrative_levers.json').write_text(json.dumps(examples,indent=2))
    print('Generated figures, illustrative levers and bootstrap interval in results/')
if __name__=='__main__':main()
