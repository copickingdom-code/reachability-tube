from pathlib import Path
import pandas as pd
df = pd.read_csv(Path("outputs/metrics")/"s2t_eval_with_runs.csv", dtype={"sid":str})
fp = df[(df.ok_by_tube==1)&(df.ok_by_runs==0)]
fn = df[(df.ok_by_tube==0)&(df.ok_by_runs==1)]
print("FP sids:", fp["sid"].tolist())
print("FN sids:", fn["sid"].tolist())
