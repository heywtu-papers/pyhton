import numpy as np
import pandas as pd 

trafic_accidents = {}
a = pd.read_csv('accidents_2005_to_2007.csv',  index_col=False, dtype='unicode')
b = pd.read_csv('accidents_2009_to_2011.csv',  index_col=False, dtype='unicode')
c = pd.read_csv('accidents_2012_to_2014.csv',  index_col=False, dtype='unicode')
d = pd.read_csv('ukTrafficAADF.csv',  index_col=False, dtype='unicode')

e = pd.DataFrame(a)
f = pd.DataFrame(b)
g = pd.DataFrame(c)
h = pd.DataFrame(d)

result = pd.concat(e,f,g,h)
print(result)








