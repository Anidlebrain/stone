import numpy as np
import pandas as pd

data = np.load("results_round4/top100.npz")

strategies = data["strategies"]
wins = data["wins"]

df = pd.DataFrame(strategies)
df["wins"] = wins

df.to_csv("top100.csv", index=False)