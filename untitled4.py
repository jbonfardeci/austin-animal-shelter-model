# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:27:06 2018

@author: bonfardeci-j
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import numpy as np

df = pd.DataFrame(data=[['cat'],['dog'],['cat'],['cat']],\
                  columns=['AnimalType'])


#sns.set(style="darkgrid")
#ax = sns.factorplot(y='AnimalType', kind='count', data=df, size=5, aspect=2)
#plt.show()
        
#get_class_plots(df)
#print(df.iloc.__doc__)
#print(np.random.permutation.__doc__)
#df.iloc[np.random.permutation(len(df))]
df2 = df.sample(frac=1.0, random_state=12345)
print(df2)