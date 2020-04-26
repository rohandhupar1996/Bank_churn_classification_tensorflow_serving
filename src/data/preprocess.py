import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
import os

train_data_dir=os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/raw/Churn_Modelling.csv'))


# Import the csv data as pands's dataframe
data=pd.read_csv(train_data_dir)

#%%
# Delete unimportant featurres from the dataset
data=data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

#%%

# Get the features from the dataframe
features=data.iloc[:,0:10].values
#Get the labels from the dataset
labels=data.iloc[:,10].values
       
#%%
# Create an instance of LabelEncoder() class
labelEncoder = LabelEncoder()
   
# Encode the feature "Geography", swap the old feature with the new encoded feature
features[:,1]=labelEncoder.fit_transform(features[:, 1])
# Encode the feature "Gender", swap the old feature with the new encoded feature
features[:,2]=labelEncoder.fit_transform(features[:, 2])

#%%

# Create an instance of LabelEncoder() class, provide the column indices of categorical features
ohe=OneHotEncoder(categorical_features=[1, 2, 6, 7, 8])

# Perform one hot encoding of the categorical features.
features=ohe.fit_transform(features).toarray()
#%%    

#Create an instance of sklearn's MinMaxScalar and use it to map the feature values in the range [0,1]
sc = MinMaxScaler()
features = sc.fit_transform(features)

#%%

# Split the dataset into training and testset. The train/test/validation ratio is 70%/20%/10%

# Split the data into train and test sets with the ratio of 70/30 %
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)
# Split the testset again into a new testset and a validation set 
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.333, random_state=42, shuffle=True)

print(x_val)






















    