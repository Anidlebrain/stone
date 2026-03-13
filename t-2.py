from unicodedata import name
import numpy as np
import pandas as pd
import judge

name = judge.name + "/top100K"

data = np.load(name + ".npz")

strategies = data["strategies"]
wins = data["wins"]

df = pd.DataFrame(strategies)
df["wins"] = wins

df.to_csv(name + ".csv", index=False)
