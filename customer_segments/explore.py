
import pandas as pd
import numpy as np

d = {'a': [1, 2, 3], 'b': [0xdead, 0xbeef, 0xbabe]}
df = pd.DataFrame.from_dict(d)

df = pd.DataFrame(index=[0, 1, 2], columns=['a', 'b', 'c'], data=[[10, 20, 30], [100, 200, 300], [1, 2, 3]])

print(df.quantile(0.25))

print(df.quantile(0.75))

print(np.percentile(df['a'], 25))