from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import roc_auc_score
from generate_data import make_dataset,FEATURES,RULES,knowledge_features

BASE=Path(__file__).parent

def run(seeds=40,n=800):
    records=[]
    for regime in ['construct','raw','misspecified']:
        for seed in range(seeds):
            df=make_dataset(n=n,seed=1000+seed,regime='raw' if regime=='raw' else 'construct')
            raw=df[FEATURES]
            rules=RULES
            if regime=='misspecified':
                rules={
                    'technology': (['cloud_adoption','digital_skill_share'],.55),
                    'organization': (['leadership_commitment','process_digitization'],.60),
                    'process': (['agile_maturity','data_maturity'],.55),
                    'data': (['rd_intensity','partner_ecosystem'],.50),
                    'ecosystem': (['cyber_posture','innovation_pipeline'],.50),
                }
            kb=knowledge_features(raw,rules)
            matrices={'raw':raw,'kb':kb,'hybrid':pd.concat([raw,kb],axis=1)}
            y=df['ready'].to_numpy()
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            for name,X in matrices.items():
                p=cross_val_predict(LogisticRegression(max_iter=2000,class_weight='balanced'),X,y,cv=cv,method='predict_proba')[:,1]
                records.append({'regime':regime,'seed':seed,'features':name,'roc_auc':roc_auc_score(y,p)})
    return pd.DataFrame(records)

if __name__=='__main__':
    results=run()
    (BASE/'results').mkdir(exist_ok=True)
    results.to_csv(BASE/'results'/'robustness.csv',index=False)
    print(results.groupby(['regime','features']).roc_auc.agg(['mean','std']).round(4))
    print('Reconstructed stress test: numerical results need not match the manuscript.')
