from pathlib import Path
import pandas as pd
root = Path("outputs/metrics")
df = pd.read_csv(root/"s2t_eval_with_runs.csv", dtype={"sid":str})
def conf(sub):
    TP=int(((sub.ok_by_tube==1)&(sub.ok_by_runs==1)).sum())
    TN=int(((sub.ok_by_tube==0)&(sub.ok_by_runs==0)).sum())
    FP=int(((sub.ok_by_tube==1)&(sub.ok_by_runs==0)).sum())
    FN=int(((sub.ok_by_tube==0)&(sub.ok_by_runs==1)).sum())
    P = TP/(TP+FP+1e-9); R = TP/(TP+FN+1e-9)
    ACC= (TP+TN)/max(len(sub),1)
    SPEC= TN/(TN+FP+1e-9)
    F1 = 2*P*R/(P+R+1e-9)
    return dict(N=len(sub),TP=TP,TN=TN,FP=FP,FN=FN,
                precision=P, recall=R, accuracy=ACC, specificity=SPEC, F1=F1)
rows=[]
for (mu,v0),sub in df.groupby(["mu","v0"]):
    rows.append({"mu":mu,"v0":v0, **conf(sub)})
pd.DataFrame(rows).to_csv(root/"s2t_group_report.csv", index=False, encoding="utf-8")
print("Saved:", (root/"s2t_group_report.csv").resolve())
