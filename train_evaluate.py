from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from generate_data import FEATURES,knowledge_features

BASE=Path(__file__).parent

def evaluate(df,seed=42):
    y=df['ready'].to_numpy()
    raw=df[FEATURES]
    kb=knowledge_features(raw)
    variants={'raw':raw,'kb':kb,'hybrid':pd.concat([raw,kb],axis=1)}
    models={'LR':lambda:LogisticRegression(max_iter=2000,class_weight='balanced'),
            'RF':lambda:RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=8,random_state=seed,n_jobs=-1),
            'GBM':lambda:GradientBoostingClassifier(n_estimators=180,learning_rate=.06,max_depth=3,random_state=seed)}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    rows=[]
    for model_name, factory in models.items():
        for name,X in variants.items():
            prob=cross_val_predict(factory(),X,y,cv=cv,method='predict_proba',n_jobs=1)[:,1]
            pred=(prob>=.5).astype(int)
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
            rows.append({'model':model_name,'features':name,'accuracy':accuracy_score(y,pred),'f1':f1_score(y,pred),
                         'roc_auc':roc_auc_score(y,prob),'pr_auc':average_precision_score(y,prob),
                         'brier':brier_score_loss(y,prob),'false_negatives':int(fn),'false_positives':int(fp),
                         'avg_cost_3fn_1fp':(3*fn+fp)/len(y)})
    return pd.DataFrame(rows)

if __name__=='__main__':
    df=pd.read_csv(BASE/'synthetic_enterprise_data.csv')
    result=evaluate(df)
    (BASE/'results').mkdir(exist_ok=True)
    result.to_csv(BASE/'results'/'model_comparison.csv',index=False)
    print(result.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('These are RECONSTRUCTED results, not verified original manuscript results.')
