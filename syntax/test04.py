
import pandas as pd
import numpy as np


dict = {'a':[1,2,3,4], 'b':[4,5,6,7],'c':['a','b','c','d']}

df = pd.DataFrame(dict)
print(df)

df.iloc[0:2,1:3] = 0
print(df)