import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = ['rd_intensity','digital_skill_share','process_digitization','data_maturity','cloud_adoption','agile_maturity','leadership_commitment','partner_ecosystem','cyber_posture','innovation_pipeline']
RULES = {
    'technology': (['cloud_adoption','cyber_posture'],.55),
    'organization': (['leadership_commitment','digital_skill_share'],.60),
    'process': (['process_digitization','agile_maturity'],.55),
    'data': (['data_maturity','rd_intensity'],.50),
    'ecosystem': (['partner_ecosystem','innovation_pipeline'],.50),
}

def knowledge_features(df, rules=RULES):
    scores = pd.DataFrame({name:df[cols].mean(axis=1) for name,(cols,_) in rules.items()},index=df.index)
    gaps = pd.DataFrame({name:(threshold-scores[name]).clip(lower=0) for name,(_,threshold) in rules.items()})
    out = scores.add_prefix('kb_')
    out['kb_min_construct'] = scores.min(axis=1)
    out['kb_rule_pass_rate'] = pd.DataFrame({k:(scores[k]>=t).astype(float) for k,(_,t) in rules.items()}).mean(axis=1)
    out['kb_gap_severity'] = gaps.mean(axis=1)
    return out

def make_dataset(n=800,seed=42,regime='construct',misspecified=False):
    rng=np.random.default_rng(seed)
    corr=np.full((5,5),.16);np.fill_diagonal(corr,1)
    latent=rng.multivariate_normal(np.zeros(5),corr,size=n)
    offsets=np.array([-.17,-.10,-.12,-.11,-.08,-.13,-.10,-.22,-.08,-.19])
    factors=np.array([3,1,2,3,0,2,1,4,0,4])
    x=1/(1+np.exp(-(latent[:,factors]*.85+rng.normal(0,.55,(n,10))+offsets)))
    df=pd.DataFrame(x,columns=FEATURES)
    kb=knowledge_features(df)
    if regime=='raw':
        score=2.8*(df['cloud_adoption']-.5)*(df['digital_skill_share']-.5)+2.4*(df['data_maturity']-.5)*(df['partner_ecosystem']-.5)+1.2*np.sin(5*df['rd_intensity'])
    else:
        score=1.8*kb[['kb_technology','kb_organization','kb_process','kb_data','kb_ecosystem']].mean(axis=1)+.45*df['data_maturity']+.25*df['innovation_pipeline']
    score=np.asarray(score)+rng.normal(0,.19,n)
    cutoff=np.quantile(score,.565)
    y=(score>cutoff).astype(int)
    flips=rng.choice(n,size=round(n*.04),replace=False)
    y[flips]=1-y[flips]
    df['ready']=y
    return df

if __name__=='__main__':
    output=Path(__file__).parent/'synthetic_enterprise_data.csv'
    make_dataset().to_csv(output,index=False)
    print(f'Saved {output}')
