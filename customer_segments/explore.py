
import pandas as pd


d = {'a': [1,2,3], 'b': [0xdead, 0xbeef, 0xbabe]}
df = pd.DataFrame.from_dict(d)

df = pd.DataFrame(index=[0, 1, 2], columns=['a', 'b', 'c'], data=[[10, 20, 30], [100, 200, 300], [1, 2, 3]])


print(df.drop('a'))