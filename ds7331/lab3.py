# %%python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Tuple
import time
from itertools import cycle
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import (
    KFold,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    cross_validate,
)
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, MiniBatchKMeans, KMeans
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn import tree

from dataframe_column_identifier import DataFrameColumnIdentifier


from sklearnex import patch_sklearn

patch_sklearn()

# %%python
# Data Init
url = "https://raw.githubusercontent.com/bensonnd/msds/master/ds7331/data/hotel_bookings.csv"
hotel = pd.read_csv(url)

# add `arrival_date` - converting arrival y/m/d columns to a singular column
hotel["arrival_date"] = pd.to_datetime(
    [
        f"{y}-{m}-{d}"
        for y, m, d in zip(
            hotel.arrival_date_year,
            hotel.arrival_date_month,
            hotel.arrival_date_day_of_month,
        )
    ]
)

# source:
# https://stackoverflow.com/questions/54487059/pandas-how-to-create-a-single-date-column-from-columns-containing-year-month

# add `length_of_stay`
# we will use this as predictor in the classification task, as well as the continuous variable
# we want to regress to
hotel["length_of_stay"] = (
    hotel["stays_in_weekend_nights"] + hotel["stays_in_week_nights"]
)


# set `length_of_stay` as a pandas time delta
length = (
    hotel["length_of_stay"].apply(np.ceil).apply(lambda x: pd.Timedelta(x, unit="D"))
)

# source:
# https://stackoverflow.com/questions/42768649/add-days-to-date-in-pandas


# add `departure_date`
hotel["departure_date"] = hotel["arrival_date"] + length


# add `total_revenue`
hotel["total_revenue"] = abs(hotel["adr"]) * hotel["length_of_stay"]


# add `country_cancelation_rate`
# first we aggregate number of cancelations per country, then divide by total records per country
# once we have the rate, we join back on country name
hotel["is_canceled_int"] = pd.to_numeric(hotel["is_canceled"])

contry_cancellation_rate_df = pd.DataFrame(
    hotel.groupby(["country"])["is_canceled_int"].count()
)

contry_cancellation_rate_df.columns = ["country_count"]
contry_cancellation_rate_df["cancelations"] = pd.DataFrame(
    hotel.groupby(["country"])["is_canceled_int"].sum()
)

contry_cancellation_rate_df["country_cancelation_rate"] = (
    contry_cancellation_rate_df["cancelations"]
    / contry_cancellation_rate_df["country_count"]
)

hotel = hotel.join(contry_cancellation_rate_df, on="country")

total_cancelations = hotel.is_canceled_int.sum()


# add `stays_in_week_nights_bool` and `stays_in_weekend_nights`
# by changing `stays_in_week_nights` and `stays_in_weekend_nights` to Boolean
hotel["stays_in_week_nights_bool"] = np.where(hotel["stays_in_week_nights"] > 0, 1, 0)
hotel["stays_in_weekend_nights_bool"] = np.where(
    hotel["stays_in_weekend_nights"] > 0, 1, 0
)


# add `company_booking_bool` by changing `company` to Boolean
hotel["company"] = hotel["company"].fillna(0)
hotel["company_booking_bool"] = np.where(hotel["company"] > 0, 1, 0)


# add `used_agent_bool` by changing `agent` to Boolean
hotel["agent"] = hotel["agent"].fillna(0)
hotel["used_agent_bool"] = np.where(hotel["agent"] > 0, 1, 0)


# add `right_room_bool`
hotel["right_room_bool"] = np.where(
    (
        hotel["reserved_room_type"].astype(str)
        == hotel["assigned_room_type"].astype(str)
    ),
    1,
    0,
)


# add `previously_canceled_bool` by changing `previous_cancellations` to Boolean
hotel["previously_canceled_bool"] = (
    hotel["previous_cancellations"].astype(bool).astype(int)
)


# add `lead_time_cat` by descretizing `lead_time`
# `lead_time` categories 0 days to 1 week, 1 week to 1 month, 1 month to 6 months, greater than 6 months
hotel["lead_time_cat"] = pd.cut(
    hotel["lead_time"],
    bins=[0, 7, 31, 180, 737],
    labels=["booked_wk_out", "booked_mnth_out", "booked_6_mnths_out", "booked_long"],
)


# add `country_group_cat` changed to top_ten and other_country by grouping the top 10 countries, and all others
hotel["country_group_cat"] = hotel["country"].apply(
    lambda x: "top_ten"
    if x in ["PRT", "GBR", "BEL", "NLD", "DEU", "ESP", "ITA", "IRL", "BRA", "FRA"]
    else "other_country"
)


# add `parking_space_required_bool` by changing `required_car_parking_spaces` to Boolean
hotel["parking_space_required_bool"] = np.where(
    hotel["required_car_parking_spaces"] > 0, 1, 0
)

# %%python
# drop the unneeded temp columns created in order to create `country_cancelation_rate`
hotel = hotel.drop(["country_count", "cancelations", "is_canceled_int"], axis=1)


# dropping redundant date columns as this data is now available in `arrival_date`
hotel = hotel.drop(
    ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"], axis=1
)


# dropping redundant stay length columns as this data is now available in
# `stays_in_week_nights_cat` and `stays_in_weekend_nights_cat`
hotel = hotel.drop(["stays_in_week_nights", "stays_in_weekend_nights"], axis=1)


# dropping redundant columns that changed to boolean or categorical or in grouped attributes
hotel = hotel.drop(
    [
        "company",
        "agent",
        "lead_time",
        "country",
        "previous_cancellations",
        "required_car_parking_spaces",
    ],
    axis=1,
)

# dropping redundant columns `reservation_status` as it's nearly identical to `is_canceled` - the target in classification
hotel = hotel.drop(["reservation_status"], axis=1)
hotel = hotel.drop(["babies", "is_repeated_guest"], axis=1)

# %%python
print(hotel.columns[hotel.isnull().any()].tolist())

hotel["lead_time_cat"] = hotel["lead_time_cat"].astype("object")

# replacing missing values for categorical attributes to 'Unknown'
cat_cols = ["lead_time_cat"]
hotel[cat_cols] = hotel[cat_cols].replace({np.nan: "Unknown"})

# replacing missing values for continuous attributes to 0
con_cols = ["children", "country_cancelation_rate"]
hotel[con_cols] = hotel[con_cols].replace({np.nan: 0})

# Source:
# https://stackoverflow.com/questions/45416684/python-pandas-replace-multiple-columns-zero-to-nan

# missing columns sanity check
assert len(hotel.columns[hotel.isnull().any()].tolist()) == 0

# %%python
# check for duplicate rows
dups = hotel.duplicated().sum()

# source:
# https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe

print(f"{round(dups/len(hotel)*100,2)}% of all records are considered duplicates")

# check for duplicate columns
hotel.columns.duplicated()

# %%python
# converting these columns to string type
hotel[cat_cols] = hotel[cat_cols].astype(str)

# converting `children` to int since you can't have a half a baby
hotel["children"] = hotel["children"].astype(int)

# make all adr (average daily rate) values positive. (only one is actually negative)
hotel["adr"] = hotel["adr"].abs()

# list of continuous attributes
hotel_continuous = [
    "arrival_date_week_number",
    "adults",
    "children",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
    "length_of_stay",
    "total_revenue",
    "country_cancelation_rate",
]

# hotel df of continuos variables in the data set
hotelCont = hotel[hotel_continuous]


# list of categorical attributes
hotel_categoricals = [
    "hotel",
    "is_canceled",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "stays_in_week_nights_bool",
    "stays_in_weekend_nights_bool",
    "company_booking_bool",
    "used_agent_bool",
    "right_room_bool",
    "previously_canceled_bool",
    "lead_time_cat",
    "country_group_cat",
    "parking_space_required_bool",
]

# setting categoricals as that type.
for cat in hotel_categoricals:
    hotel[cat] = hotel[cat].astype("category")


# hotel df of categorical variables
hotelCats = hotel[hotel_categoricals]


# converting reservation_status_date to datetime
hotel["reservation_status_date"] = pd.to_datetime(hotel["reservation_status_date"])


# hotel df of datetime variables
hotelDates = hotel.select_dtypes(include=["datetime64"])

# %%python
#######################################
#
# Normalized the entire data set here only to identify and remove outliers. We'll later split the data prior to scaling, and
# and will then normalize the test and training splits separately to avoid data snooping.
#
#######################################

# Mean Normalization of the Continous Variables -  still contains large outliers pictured in graphs below
hotelCont_mean_normed = (hotelCont - hotelCont.mean()) / (hotelCont.std())

# Removing outliers greater than 5 standard deviations away
hotel_nol = hotelCont_mean_normed[(np.abs(hotelCont_mean_normed) < 5).all(axis=1)]


# Grabbing indices of the non-outlier rows
no_outlier_indices = pd.DataFrame(hotel_nol.index)
no_outlier_indices.rename(columns={0: "indices"}, inplace=True)
# no_outlier_indices

# This data set has removed the outliers and un-normed the data so that we can use it without snooping on our test data
hotel_no_outliers = pd.concat([hotelCont, hotelCats], axis=1, join="inner")
hotel_no_outliers = hotel_no_outliers.iloc[
    no_outlier_indices.indices,
]

# %%python
# resetting the dataframe index and dropping the extra column it creates
hotel_no_outliers.reset_index(drop=True, inplace=True)

# classification task
data_clf = hotel_no_outliers.loc[:, hotel_no_outliers.columns != "is_canceled"]
target_clf = hotel_no_outliers["is_canceled"]


# classification task
data_clf = pd.get_dummies(data_clf, drop_first=True)

# %%python
# setting a global number of jobs for parallel processing in the rest of the notebook
nj = -1

# classification task
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=24)

# setting the data and targets to numpy arrays
data_Array_clf, target_Array_clf = data_clf.to_numpy(), target_clf.to_numpy()

# split into training and test sets
for train_index, test_index in sss.split(data_Array_clf, target_Array_clf):
    train_data_clf, df_test_data_clf = (
        data_Array_clf[train_index],
        data_Array_clf[test_index],
    )
    train_target_clf, df_test_target_clf = (
        target_Array_clf[train_index],
        target_Array_clf[test_index],
    )

# %%python
# scl = StandardScaler()


# # classification Task
# # Train and Test Sets
# scl_clf = scl.fit(train_data_clf)

# # scaling the training set
# train_data_clf = scl_clf.transform(train_data_clf)

# # scaling the test set
# test_data_clf = scl_clf.transform(df_test_data_clf)

# %%python
# classification task
# training set
df_train_data_clf = pd.DataFrame(train_data_clf, columns=data_clf.columns)
df_train_target_clf = pd.DataFrame(train_target_clf, columns=["is_canceled"])

# test set
df_test_data_clf = pd.DataFrame(df_test_data_clf, columns=data_clf.columns)
df_test_target_clf = pd.DataFrame(df_test_target_clf, columns=["is_canceled"])

# %%python
# PCA
pca = PCA(2)
pca.fit(df_train_data_clf)
X_train = pd.DataFrame(pca.transform(df_train_data_clf))
X_test = pd.DataFrame(pca.transform(df_test_data_clf))
y_train = df_train_target_clf
y_test = df_test_target_clf

# %%python

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    applies standard scaler (z-scores) to training data and predicts z-scores for the test set
    """
    scaler = StandardScaler()
    to_scale = [col for col in X_train.columns.values]
    scaler.fit(X_train[to_scale])
    X_train[to_scale] = scaler.transform(X_train[to_scale])
    
    # predict z-scores on the test set
    X_test[to_scale] = scaler.transform(X_test[to_scale])

    return X_train, X_test

X_train, X_test = scale_features(X_train, X_test)

# %%python
# classification task
cv_clf = StratifiedKFold(n_splits=10, shuffle=True, random_state=24)

# %%python
def grid_search(model, grid, score):
    # define search
    search_grid_search_measure = GridSearchCV(
        model, grid, scoring=score, cv=cv_clf, n_jobs=nj, verbose=1
    )

    labels_true = y_train.to_numpy()
    labels_true = np.squeeze(np.asarray(np.transpose(labels_true)))

    # perform the search
    results_grid_search_measure = search_grid_search_measure.fit(X_train, labels_true)

    return results_grid_search_measure


def plot_pca(model):
    model.fit(X_train)
    labels = model.predict(X_test)

    #filter rows of original data
    filtered_label0 = X_test[labels == 0]
    filtered_label0 = filtered_label0.to_numpy()
    filtered_label1 = X_test[labels == 1]
    filtered_label1 = filtered_label1.to_numpy()

    #Plotting the results
    plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
    plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'black')
    plt.show()

# %%python
############ KMM ############
# MiniBatchKMeans - kmm
cluster_kmm = MiniBatchKMeans(random_state=24, n_clusters=2)

# setting the parameters - kmm
grid_kmm = {}
grid_kmm["batch_size"] = np.arange(20, 140, 20)
grid_kmm["max_iter"] = np.arange(20, 140, 20)
grid_kmm["n_init"] = np.arange(3, 13, 2)
grid_kmm["reassignment_ratio"] = np.arange(0.01, 0.05, 0.01)

# %%python
kmm_results_v_measure = grid_search(
    model=cluster_kmm, grid=grid_kmm, score="v_measure_score"
)

# summarize kmm
print("Mean V Measure: %.3f" % kmm_results_v_measure.best_score_)
print("Config V Measure: %s" % kmm_results_v_measure.best_params_)
print(f"Best estimator: {kmm_results_v_measure.best_estimator_}\n")


# setting the best paramaters for kmm
clf_kmm = kmm_results_v_measure.best_estimator_

plot_pca(clf_kmm)


# %%python
############ KM ############
# kmeans
cluster_km = KMeans(random_state=24, n_clusters=2)

# setting the parameters - kmeans
grid_km = {}
grid_km["max_iter"] = np.arange(20, 140, 20)
grid_km["n_init"] = np.arange(3, 13, 2)


# %%python
km_results_v_measure = grid_search(
    model=cluster_km, grid=grid_km, score="v_measure_score"
)

# summarize kmeans
print("Mean V Measure: %.3f" % km_results_v_measure.best_score_)
print("Config V Measure: %s" % km_results_v_measure.best_params_)
print(f"Best estimator: {km_results_v_measure.best_estimator_}\n")


# setting the best paramaters for kmeans
clf_km = km_results_v_measure.best_estimator_

plot_pca(clf_km)


# %%python
############ BIRCH ############
# birch
cluster_bir = Birch(n_clusters=2)

# setting the parameters - birch
grid_bir = {}
grid_bir["threshold"] = np.arange(0.50, 2, 0.5)
grid_bir["branching_factor"] = np.arange(10, 60, 20)


# %%python
bir_results_v_measure = grid_search(
    model=cluster_bir, grid=grid_bir, score="v_measure_score"
)

# summarize birch
print("Mean V Measure: %.3f" % bir_results_v_measure.best_score_)
print("Config V Measure: %s" % bir_results_v_measure.best_params_)
print(f"Best estimator: {bir_results_v_measure.best_estimator_}\n")


# setting the best paramaters for birch
clf_bir = bir_results_v_measure.best_estimator_

plot_pca(clf_bir)

# %%python
############ DECISTION TREE ############
# decision tree
clf_decision = tree.DecisionTreeClassifier(random_state=24)

# setting the parameters - dc
grid_dc = {}
grid_dc["criterion"] = ["gini", "entropy"]
grid_dc["splitter"] = ["best", "random"]
grid_dc["max_features"] = ["auto", "sqrt", "log2"]


# %%python
%%time
dc_results_v_measure = grid_search(model=clf_decision, grid=grid_dc, score="f1")

# summarize dc
print("Mean f1 Measure: %.3f" % dc_results_v_measure.best_score_)
print("Config f1 Measure: %s" % dc_results_v_measure.best_params_)
print(f"Best estimator: {dc_results_v_measure.best_estimator_}\n")


# setting the best paramaters for dc
clf_dc = dc_results_v_measure.best_estimator_



def cluster_predict(model, X_train, X_test, y_train, y_test):
    
    def get_clusters(
       model, X_train: pd.DataFrame, X_test: pd.DataFrame, n_clusters: 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        model.fit(X_train)
        
        # apply the labels
        train_labels = model.labels_
        X_train_clstrs = X_train.copy()
        X_train_clstrs["clusters"] = train_labels

        # predict labels on the test set
        test_labels = model.predict(X_test)
        X_test_clstrs = X_test.copy()
        X_test_clstrs["clusters"] = test_labels

        return X_train_clstrs, X_test_clstrs

    X_train_clstrs, X_test_clstrs = get_clusters(model, X_train, X_test, 2)

    X_train_scaled, X_test_scaled = X_train_clstrs, X_test_clstrs

    # X_train_scaled, X_test_scaled = scale_features(X_train_clstrs, X_test_clstrs)

    # to divide the df by cluster, we need to ensure we use the correct class labels, we'll use pandas to do that
    train_clusters = X_train_scaled.copy()
    test_clusters = X_test_scaled.copy()
    train_clusters['y'] = y_train
    test_clusters['y'] = y_test

    # locate the "0" cluster
    train_0 = train_clusters.loc[train_clusters.clusters == 0] # after scaling, 0 went negative
    test_0 = test_clusters.loc[test_clusters.clusters == 0]
    y_train_0 = train_0.y.values
    y_test_0 = test_0.y.values

    # locate the "1" cluster
    train_1 = train_clusters.loc[train_clusters.clusters == 1] # after scaling, 1 dropped slightly
    test_1 = test_clusters.loc[test_clusters.clusters == 1]
    y_train_1 = train_1.y.values
    y_test_1 = test_1.y.values

    # the base dataset has no "clusters" feature
    X_train_base = X_train_scaled.drop(columns=['clusters'])
    X_test_base = X_test_scaled.drop(columns=['clusters']) # drop the targets from the training set


    X_train_0 = train_0.drop(columns=['y'])
    X_test_0 = test_0.drop(columns=['y'])
    X_train_1 = train_1.drop(columns=['y'])
    X_test_1 = test_1.drop(columns=['y'])
    datasets = {
        'base': (X_train_base, y_train, X_test_base, y_test),
        'cluster-feature': (X_train_scaled, y_train, X_test_scaled, y_test),
        'cluster-0': (X_train_0, y_train_0, X_test_0, y_test_0),
        'cluster-1': (X_train_1, y_train_1, X_test_1, y_test_1),
    }

    def run_exps(datasets: dict) -> pd.DataFrame:
        '''
        runs experiments on a dict of datasets
        '''
        # initialize a logistic regression classifier
        model = clf_dc
        
        dfs = []
        results = []
        conditions = []
        scoring = ['accuracy','precision_weighted','recall_weighted','f1_weighted']
        for condition, splits in datasets.items():
            X_train = splits[0]
            y_train = splits[1]
            X_test = splits[2]
            y_test = splits[3]
            
            kfold = cv_clf
            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results.append(cv_results)
            conditions.append(condition)
            this_df = pd.DataFrame(cv_results)
            this_df['condition'] = condition
            dfs.append(this_df)
            final = pd.concat(dfs, ignore_index=True)
        
        # We have wide format data, lets use pd.melt to fix this
        results_long = pd.melt(final,id_vars=['condition'],var_name='metrics', value_name='values')
        
        # fit time metrics, we don't need these
        time_metrics = ['fit_time','score_time'] 
        results = results_long[~results_long['metrics'].isin(time_metrics)] # get df without fit data
        results = results.sort_values(by='values')
        
        return results
    return run_exps(datasets)


# %%python
cluster_models = {"MiniBatchKMeans": clf_kmm, "KMeans": clf_km, "Birch": clf_bir}
for name, model in cluster_models.items():
    df = cluster_predict(model, X_train, X_test, y_train, y_test)
    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="condition", y="values", hue="metrics", data=df, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(f'Comparison of Dataset by Classification Metric - {name}')
    
    print(
        name,
        pd.pivot_table(
            df,
            index="condition",
            columns=["metrics"],
            values=["values"],
            aggfunc="mean",
        ),
        "\n"
    )


# %%
