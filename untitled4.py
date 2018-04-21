# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:27:06 2018

@author: bonfardeci-j
"""

import pandas as pd

df = pd.DataFrame(data=[[1,2,3]], columns=['1', '2', '3'])

df.plot.bar()