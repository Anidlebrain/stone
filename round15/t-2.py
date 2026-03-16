import numpy as np
import pandas as pd
import judge

path = judge.name + '/top100.npz'
out_csv = judge.name + '/top100.csv'

print(f'Round: {judge.round_no}')
print(f'Load: {path}')

data = np.load(path)
strategies = data['strategies']
wins = data['wins']

cols = ['buyA', 'buyB', 'buyC', 'insurance'] + [f'p{i}' for i in range(1, 11)]
df = pd.DataFrame(strategies, columns=cols)
df['skill_cost'] = [judge.strategy_skill_cost(row) for row in strategies]
df['total_used'] = [judge.strategy_total_used(row) for row in strategies]
df['wins'] = wins

df.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')
