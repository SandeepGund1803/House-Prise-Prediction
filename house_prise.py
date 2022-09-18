# import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SequentialFeatureSelector
import warnings
warnings.filterwarnings('ignore')


# data gathering
df=pd.read_csv(r'kc_house_data.csv')

df.rename(columns={'long':'longi'},inplace=True)


# target selection
x = df.drop(['price','date'],axis=1)
y = df['price']

# model instance
knn_model = KNeighborsRegressor()

# feature selections
sfs = SequentialFeatureSelector(knn_model, n_features_to_select= 6, direction='forward', cv = 7)
sfs.fit(x,y)
features = x.columns[sfs.get_support()]

# training of model
knn_model.fit(x[features],y)

# pickle dump
with open('house_prise.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
