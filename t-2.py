import numpy as np
import pandas as pd
import judge

path = judge.name + "/top100.npz"
out_csv = judge.name + "/top100.csv"

print(f"Round: {judge.round_no}")
print(f"Load: {path}")

data = np.load(path)
strategies = data["strategies"]
wins = data["wins"]

df = pd.DataFrame(strategies)
df["wins"] = wins

df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
