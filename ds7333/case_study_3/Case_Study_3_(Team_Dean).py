# %%
# sys
import warnings

warnings.filterwarnings("ignore")  # ignoring warnings

# SKLearn
import sklearn
from sklearn import datasets, tree, neighbors, decomposition, datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Other Packages
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
from pandas_profiling import ProfileReport
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# %%
from os import listdir, getcwd, chdir, walk
from os.path import isfile, join, dirname, realpath


# %%
# intel patch for running sklearn - this helps performance a lot
from sklearnex import patch_sklearn

patch_sklearn()


# %%
directory = "/home/bensonnd/msds/ds7333/case_study_3/Data2"

sub_dirs = [x[0] for x in walk(directory) if x[0] != directory]

# results_df = pd.DataFrame()

# %%
for d in sub_dirs:
    files = [f for f in listdir(d) if isfile(join(d, f))]

# %%
