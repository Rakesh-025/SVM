#Support vector machine for forest fire data set
#Loading the packages

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Loading the data set
fire_data = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\svm\\forestfires.csv")
fire_data.describe()
fire_data.drop(['month','day'], axis = 1)

#Converting the non numeric data into numeric data
fire_data['size_category']= labelencoder.fit_transform(fire_data['size_category'])

#Converted training and testing data input and output
train,test = train_test_split(fire_data, test_size = 0.20)

train_X = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
test_X  = test.iloc[:, :-1]
test_y  = test.iloc[:, -1]


#Generating linear kernal model
fire_linear = SVC(kernel = "linear")
fire_linear.fit(train_X, train_y)
pred_test_linear = fire_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

#  Generating RBF kernal model
fire_rbf = SVC(kernel = "rbf")
fire_rbf.fit(train_X, train_y)
pred_test_rbf = fire_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)


