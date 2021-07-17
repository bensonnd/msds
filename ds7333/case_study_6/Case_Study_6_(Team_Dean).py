# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# from sklearnex import patch_sklearn

# patch_sklearn()

# %% [markdown]
### Data loading

# %%
# df = pd.read_csv(
#     "https://raw.githubusercontent.com/bensonnd/msds/ds7333-neil/ds7333/case_study_6/Data/all_train.csv"
# )

# df = pd.read_csv("./Data/all_train.csv")


# # sampling to reduce work load on the pc
# df = df.sample(n=50000, random_state=0, ignore_index=True)

# %%
# df.to_csv("./Data/all_train_sample.csv", index=False, encoding='utf-8')

df = pd.read_csv("./Data/all_train_sample.csv")

# %%
df.head(5)
# %% [markdown]
# Because we see that mass is orders of magnitude higher than all other fields, we will normalize/scale the data

# %%
df.info()

# %%
df.columns

# %% [markdown]
### Checking for missing values
# %%
df.columns[df.isnull().any()].tolist()

# %% [markdown]
# No missing values to impute
# %% [markdown]
### Check for duplicate rows
dups = df.duplicated().sum()
f"{dups/len(df)*100:.2f}% of all records are considered duplicates"

# source:
# https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe

# %% [markdown]
### Check for duplicate columns
# %%
df.columns.duplicated()


# Since all the attributes are continuous, checking for any colinearity between them by plotting correlation matrix
# %%
# plot the correlation matrix using seaborn
sns.set(style="darkgrid")

# Compute the correlation matrix
corr = df.loc[:, df.columns != "# label"].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)

f.tight_layout()
# %% [markdown]
# Most of the columns show little to no correlation, however
# we see high degrees of correlation between f6 & f10, f6 & f26, and f10 & f26


# %%
print(f"Correlation between f10 and f6: {df['f10'].corr(df['f6']):.2f}")
print(f"Correlation between f10 and f26: {df['f10'].corr(df['f26']):.2f}")
print(f"Correlation between f26 and f6: {df['f26'].corr(df['f6']):.2f}")

# %%
# Removing f6 because it is redundant, and highly correlated with both f26 and f10
df = df.drop(["f6"], axis=1)

# %% [markdown]
### Setting data and target Sets
# %%
X = df.loc[:, df.columns != "# label"]
X_col_names = list(X.columns)
y = df["# label"]

# deleting to save space
del df

# %% [markdown]
# Now let's check out the distribution of all the predictors

# %%
a = 1
# %%
# checking to see if the attributes are normally distributed, if not transform
import math

n_rows = 9
n_cols = 3


def draw_histograms(df, n_cols=3):
    n_rows = math.ceil(len(df.columns) / n_cols)
    fig = plt.figure(figsize=(10, 30))
    for i, var_name in enumerate(df.columns):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=10, ax=ax)
        ax.set_title(var_name + " Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()


draw_histograms(df=X)


# %% [markdown]
# We see that there is some right skew in f0, f5, f10, f14, f18, f22, f23, f24, f25, and f26
# Similarly, we see some slieght left skew in f3

# %%
# right_skew = ["f0", "f5", "f10", "f14", "f18", "f22", "f23", "f24", "f25", "f26"]
# left_skew = ["f3"]

# for skew in right_skew:
#     X[skew] = np.log(X[skew] + 0.01)

# for skew in left_skew:
#     X[skew] = X[skew] ** 2

# %% [markdown]
# checking the transformed distributions
# %%
# draw_histograms(df=X[right_skew])

# # %% [markdown]
# # checking the transformed distributions
# # %%
# draw_histograms(df=X[left_skew])


# %% [markdown]
# Checking the balance of the target
# %%
sns.countplot(x=y)
on = y.sum()
off = len(y) - on

print(
    f"\nThere are {on} classified as 1, and {off} classified 0 making this an almost perfectly balanced target."
)
# %% [markdown]
# Because the class label is balanced, we don't need to stratify the split, we will simply
# use `train_test_split`. Setting the X and y test and train data.
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=0,
)
del X  # to save space
del y  # to save space

# %% [markdown]
### Scaling the Data
# Keeping our test and training sets separate and scaling them independently to avoid data snooping.

# %%
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# %% [markdown]
### Convert np arrays back to dataframes

# %%
y_train, y_test = y_train.to_frame(name="label"), y_test.to_frame(name="label")
X_train = pd.DataFrame(X_train, columns=X_col_names)
X_test = pd.DataFrame(X_test, columns=X_col_names)

# resetting the indeces
y_test.reset_index()
y_train.reset_index()
X_test.reset_index()
X_train.reset_index()

# %% [markdown]
### Setting input nodes
input_nodes = len(X_col_names)

# %%
