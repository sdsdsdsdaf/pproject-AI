# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os, sys

# %%
path_64 = r"physionet.org\files\dreamt\2.1.0\data_64Hz"
path_100 = r"physionet.org\files\dreamt\2.1.0\data_100Hz"

# %%
data_64 = glob.glob(os.path.join(path_64, '*.csv'))
data_100 = glob.glob(os.path.join(path_100, '*.csv'))
print(data_64)



# %%
S002 = pd.read_csv(r"physionet.org\files\dreamt\2.1.0\data_64Hz\S002_whole_df.csv")

# %%



