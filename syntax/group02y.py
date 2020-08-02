# GroupBy 집계 메소드와 함수

import numpy as np
import pandas as pd

df = pd.read_csv()

# print(np.arange(6))
# print(np.random.randn(6))
# print(df)

# (1) GroupBy 메소드를 이용한 집계 (GroupBy aggregation using methods)
# Making GroupBy object
depts = df.groupby('dept')
print(depts)

print(depts.count())
print(depts.sum())

 # Series
print(depts.sum()['value_2'])
# DataFrame 
print(pd.DataFrame(depts.sum()['value_2']))




