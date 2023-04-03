grid search model building.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import pylab as py
import scipy
from scipy import stats

## now reading the dataset into python by using pandas

data = pd.read_csv(r"C:\Users\sanan\OneDrive\Desktop\data_makino.csv")

## displays the top 5 rows of the dataset
data.head()

## gives the description of each  numeric variable in the dataset
d2 =data.describe()

## data preprocessing

## checking for the datatypes of the variables
data.dtypes## as the  all variables are in appropriate data types there is no need of typecasting


## checking for duplicates in the dataset

duplicate = data.duplicated()
duplicate

sum(duplicate) ## there are no duplicates present in the dataset


# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information

d1 = data.corr()## from the above we can that splindle_speed and cutting(N) has more correlation between each other and other variables are not so highly correlated we can retain all the columns for the futher analysis.


## zero varinace columns
## the  date seems to have nothing important for analysis hence dropping them.

data = data.drop(['Date'],axis = 1)


## missing values

# Check for count of NA's in each column
data.isna().sum() ## there are NA's values in almost every column in the dataset

## as the variables contains outliers we will fill nan values with median imputer as the mean imputer is influenced by the outliers


## median imputer for column Hydraulic_Pressure(bar)
imp = SimpleImputer(strategy='median')
data['Hydraulic_Pressure'] = imp.fit_transform(data['Hydraulic_Pressure'].values.reshape(-1, 1) )
data['Hydraulic_Pressure'].isna().sum()

## median imputer for columnCoolant_Pressure(bar)
imp = SimpleImputer(strategy='median')
data['Coolant_Pressure'] = imp.fit_transform(data['Coolant_Pressure'].values.reshape(-1, 1) )
data['Coolant_Pressure'].isna().sum()

## median imputer for columnCoolant_Pressure(bar)
imp = SimpleImputer(strategy='median')
data['Coolant_Pressure'] = imp.fit_transform(data['Coolant_Pressure'].values.reshape(-1, 1) )
data['Coolant_Pressure'].isna().sum()

## median imputer for Air_System_Pressure(bar)
imp = SimpleImputer(strategy='median')
data['Air_System_Pressure'] = imp.fit_transform(data['Air_System_Pressure'].values.reshape(-1, 1) )
data['Air_System_Pressure'].isna().sum()

## median imputer for Coolant_Temperature(?C) 
imp = SimpleImputer(strategy='median')
data['Coolant_Temperature'] = imp.fit_transform(data['Coolant_Temperature'].values.reshape(-1, 1) )
data['Coolant_Temperature'].isna().sum()


## median imputer for Spindle_Bearing_Temperature(°C)
imp = SimpleImputer(strategy='median')
data['Spindle_Bearing_Temperature'] = imp.fit_transform(data['Spindle_Bearing_Temperature'].values.reshape(-1, 1) )
data['Spindle_Bearing_Temperature'].isna().sum()

## median imputer for Hydraulic_Oil_Temperature(°C)
imp = SimpleImputer(strategy='median')
data['Hydraulic_Oil_Temperature'] = imp.fit_transform(data['Hydraulic_Oil_Temperature'].values.reshape(-1, 1) )
data['Hydraulic_Oil_Temperature'].isna().sum()






## median imputer for Spindle_Vibration(?m)
imp = SimpleImputer(strategy='median')
data['Spindle_Vibration'] = imp.fit_transform(data['Spindle_Vibration'].values.reshape(-1, 1) )
data['Spindle_Vibration'].isna().sum()

## median imputer for Tool_Vibration(?m)
imp = SimpleImputer(strategy='median')
data['Tool_Vibration'] = imp.fit_transform(data['Tool_Vibration'].values.reshape(-1, 1) )
data['Tool_Vibration'].isna().sum()

## median imputer for Tool_Vibration(?m)
imp = SimpleImputer(strategy='median')
data['Spindle_Speed'] = imp.fit_transform(data['Spindle_Speed'].values.reshape(-1, 1) )
data['Spindle_Speed'].isna().sum()



## median imputer for Voltage(volts)
imp = SimpleImputer(strategy='median')
data['Voltage'] = imp.fit_transform(data['Voltage'].values.reshape(-1, 1) )
data['Voltage'].isna().sum()

## median imputer for Torque(Nm)
imp = SimpleImputer(strategy='median')
data['Torque'] = imp.fit_transform(data['Torque'].values.reshape(-1, 1) )
data['Torque'].isna().sum()


## median imputer for Cutting(N)
imp = SimpleImputer(strategy='median')
data['Cutting'] = imp.fit_transform(data['Cutting'].values.reshape(-1, 1) )
data['Cutting'].isna().sum()

data.isna().sum()## all the missing values arw e filled with median imputer and there are no NA'S values in the dataset now


## outliers assignment/treatment
## plotiing boxplot to check otliers are present or not
sns.boxplot(data['Hydraulic_Pressure']) ## there are outliers on the lower side

sns.boxplot(data['Coolant_Pressure']) ## outliers present at both ends

sns.boxplot(data['Air_System_Pressure']) ## outliers present at both ends
 
sns.boxplot(data['Coolant_Temperature']) ## outliers present on the higher side

sns.boxplot(data['Hydraulic_Oil_Temperature'])## outliers present at both ends

sns.boxplot(data['Spindle_Bearing_Temperature'])# outliers present at both ends

sns.boxplot(data['Spindle_Vibration'])# outliers present at both ends

sns.boxplot(data['Tool_Vibration'])# outliers present at both ends

sns.boxplot(data['Spindle_Speed']) ## there are outliers on the lower side

sns.boxplot(data['Voltage'])# outliers present at both ends

sns.boxplot(data['Torque'])# outliers present at both ends

sns.boxplot(data['Cutting']) ## no outliers

## every columns has outliers in them in order to remove them we are uisng winorsization technique
## define the lower and upper value used for winorsization
pct_lower = 0.05
pct_upper = 0.95
## applying winorsization for numerical values using lambda function
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
data[numeric_cols] = data[numeric_cols].apply(lambda x: np.clip(x, np.percentile(x, pct_lower*100), np.percentile(x, pct_upper*100)))

## EDA

## auto eda
import sweetviz as sv

s = sv.analyze(data)
s.show_html()

##business moments

## first business moments
## measure of central trendency

data.mean()
data.median()


## measure of dispersion
data.var()
data.std()


## skewness
data.skew()


## kurtosis

data.kurt()


## graphical presentation

## univariate for numeric features

## histogram

sns.histplot(data['Hydraulic_Pressure']) ## distribution is approximately symmetrical with a very slight tendency towards the right tail.
sns.histplot(data['Coolant_Pressure']) ## distribution is approximately symmetrical with a very slight tendency towards the right tail.
sns.histplot(data['Air_System_Pressure']) ##  the distribution is approximately symmetrical with a very slight tendency towards the left tail.
sns.histplot(data['Coolant_Temperature']) ## distribution is approximately symmetrical with a very slight tendency towards the right tail.
sns.histplot(data['Hydraulic_Oil_Temperature']) ##indicates that the distribution is very close to symmetrical with a very slight tendency towards the left tail.
sns.histplot(data['Spindle_Bearing_Temperature']) ##  the distribution is approximately symmetrical with a very slight tendency towards the left tail.
sns.histplot(data['Spindle_Vibration']) #### distribution is approximately symmetrical with a very slight tendency towards the right tail.
sns.histplot(data['Tool_Vibration']) ##  the distribution is approximately symmetrical with a very slight tendency towards the left tail.
sns.histplot(data['Spindle_Speed']) ##  the distribution is approximately symmetrical with a very slight tendency towards the left tail.
sns.histplot(data['Voltage'])##  the distribution is approximately symmetrical with a very slight tendency towards the left tail.
sns.histplot(data['Torque']) #### distribution is approximately symmetrical with a very slight tendency towards the right tail.
sns.histplot(data['Cutting'])  ####  the distribution is approximately symmetrical with a very slight tendency towards the left tail.

## boxplot

sns.boxplot(data['Hydraulic_Pressure']) ## both righ tand left whisker almost are equal 

sns.boxplot(data['Coolant_Pressure']) ## both righ tand left whisker  almost are equal 

sns.boxplot(data['Air_System_Pressure']) ## both righ tand left whisker almost are equal 

sns.boxplot(data['Coolant_Temperature']) ## both righ tand left whisker almost are equal

sns.boxplot(data['Hydraulic_Oil_Temperature'])## both righ tand left whisker almost are equal

sns.boxplot(data['Spindle_Bearing_Temperature'])## both righ tand left whisker almost are equal

sns.boxplot(data['Spindle_Vibration'])##right whisker is long then left whisker we can say that most of the values in the upper range

sns.boxplot(data['Tool_Vibration'])## both righ tand left whisker almost are equal

sns.boxplot(data['Spindle_Speed'])##right whisker is long then left whisker we can say that most of the values in the upper range

sns.boxplot(data['Voltage'])## both righ tand left whisker almost are equal


sns.boxplot(data['Torque'])## left whisker is long then right whisker we can say that most of the values in the lower range


sns.boxplot(data['Cutting'])## both righ tand left whisker almost are equal


## Q-Q plot

sm.qqplot(data['Hydraulic_Pressure'],fit=True,line='45')## the data is normally disturbed with little skewnesss

sm.qqplot(data['Coolant_Pressure'],fit=True,line='45') ## the data is normally disturbed with little skewnesss

sm.qqplot(data['Air_System_Pressure'],fit=True,line='45')  ## the data is normally disturbed with little skewnesss

sm.qqplot(data['Coolant_Temperature'],fit=True,line='45') ## the data is normally disturbed with little skewness

sm.qqplot(data['Hydraulic_Oil_Temperature'],fit=True,line='45')  ## the data is normally disturbed with little skewness

sm.qqplot(data['Spindle_Bearing_Temperature'],fit=True,line='45') ## the data is normally disturbed with little skewness

sm.qqplot(data['Spindle_Vibration'],fit=True,line='45')  ## the data is normally disturbed with little skewness

sm.qqplot(data['Tool_Vibration'],fit=True,line='45')  ## the data is normally disturbed with little skewness

sm.qqplot(data['Spindle_Speed'],fit=True,line='45')  ## the data is normally disturbed with little skewness

sm.qqplot(data['Voltage'],fit=True,line='45')  ## the data is normally disturbed with little skewness

sm.qqplot(data['Torque'],fit=True,line='45')  ## the data is normally disturbed with little skewness

sm.qqplot(data['Cutting'],fit=True,line='45') ## the data is normally disturbed with little skewness



## univariate for categorical features

## countplot


data['Machine_ID'].value_counts()
sns.countplot(data['Machine_ID'])

data['Assembly_Line_No'].value_counts()
sns.countplot(data['Assembly_Line_No'])

data['Downtime'].value_counts()
sns.countplot(data['Downtime'])## as the both output labels are more than 30 percent the dataset is the balanced dataset

##multivariate plot
## pairplot
sns.pairplot(data,hue= data['Machine_ID'].all())

##label encoding
from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()
data['Assembly_Line_No'] = labelencoder.fit_transform(data['Assembly_Line_No'])
data['Machine_ID'] = labelencoder.fit_transform(data['Machine_ID'])
data['Downtime'] = labelencoder.fit_transform(data['Downtime'])

## input variables
X = data.iloc[:,:14]
## output variable
Y = data.iloc[:,14:]
## logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
param_grid = {'C': [0.1, 1, 10], 'penalty': ['none', 'l2','elasticnet'],'solver':['lbfgs', 'newton-cg', 'liblinear', 'sag']}
# Define the logistic regression model
model = LogisticRegression()
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')

accuracy = grid_search.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain_sc = sc.fit_transform(X_train)
Xtest_sc = sc.fit_transform(X_test)

lgr = LogisticRegression(C=0.1,penalty='l2',solver='lbfgs')
# Train model
model = lgr.fit(Xtrain_sc, y_train)
from sklearn import metrics
predicted_values = model.predict(Xtest_sc)
x = metrics.accuracy_score(y_test, predicted_values)
x
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
cm = confusion_matrix(y_test, predicted_values)
print ("Confusion Matrix : \n", cm)
f1 = f1_score(y_test,predicted_values)
f1
pre_values = model.predict(Xtrain_sc)
x1 = metrics.accuracy_score(y_train, pre_values)
x1
cm1 = confusion_matrix(y_train, pre_values)
print ("Confusion Matrix : \n", cm1)
f2 = f1_score(y_train,pre_values)
f2
result =model.predict_proba(Xtest_sc)
print(result)

## Knn
from sklearn.neighbors import KNeighborsClassifier
param_grid1 = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'],'metric':['euclidean', 'manhattan', 'cosine']}
model = KNeighborsClassifier()

# Perform grid search using cross-validation
grid_search = GridSearchCV(model, param_grid=param_grid1, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')

# Evaluate the model on the testing set
accuracy = grid_search.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')

classifier= KNeighborsClassifier(n_neighbors=3, weights = 'uniform' ,metric='manhattan')  
classifier.fit(Xtrain_sc, y_train) 

predicted_values1 = classifier.predict(Xtest_sc)
k = metrics.accuracy_score(y_test, predicted_values1)
k
f3 = f1_score(y_test,predicted_values1)
f3
kcm= confusion_matrix(y_test, predicted_values1)
print ("Confusion Matrix : \n", kcm)

pre_values1 = classifier.predict(Xtrain_sc)
k1 = metrics.accuracy_score(y_train, pre_values1)
k1
kcm1 = confusion_matrix(y_train, pre_values1)
print ("Confusion Matrix : \n", kcm1)
f4 = f1_score(y_train,pre_values1)
f4

## decision tree
from sklearn.tree import DecisionTreeClassifier
param_grid2 = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_leaf': [1, 2, 3, 4, 5],'min_samples_split':[2,3,4,5]}
model = DecisionTreeClassifier()

# Perform grid search using cross-validation
grid_search = GridSearchCV(model, param_grid=param_grid2, cv=5)
grid_search.fit(X_train, y_train)

print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')

# Evaluate the model on the testing set
accuracy = grid_search.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')

DecisionTree = DecisionTreeClassifier(criterion="entropy",min_samples_leaf = 3,max_depth=9,min_samples_split=2)

DecisionTree.fit(X_train,y_train)

predicted_values2 = DecisionTree.predict(X_test)

## test accuracy
d = metrics.accuracy_score(y_test, predicted_values2)
d

dcm= confusion_matrix(y_test, predicted_values2)
print ("Confusion Matrix : \n", dcm)

f5= f1_score(y_test,predicted_values2)
f5

## train accuracy
pre_values2 =  DecisionTree.predict(X_train)
d1 = metrics.accuracy_score(y_train, pre_values2)

d1

dcm1 = confusion_matrix(y_train, pre_values2)
print ("Confusion Matrix : \n", dcm1)
f6= f1_score(y_train,pre_values2)
f6
##naive bayes
from sklearn.naive_bayes import GaussianNB
param_grid3= {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

# Define the naive Bayes model
model = GaussianNB()

# Perform grid search using cross-validation
grid_search = GridSearchCV(model, param_grid=param_grid3, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')

# Evaluate the model on the testing set
accuracy = grid_search.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')

from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB(var_smoothing=1e-09)

NaiveBayes.fit(X_train,y_train)
## test accuracy
predicted_values3 = NaiveBayes.predict(X_test)
n = metrics.accuracy_score(y_test, predicted_values3)
n
ncm= confusion_matrix(y_test, predicted_values3)
print ("Confusion Matrix : \n", ncm)

f7= f1_score(y_test,predicted_values3)
f7
## train accuracy
pre_values3 =  NaiveBayes.predict(X_train)
n1 = metrics.accuracy_score(y_train, pre_values3)
n1

ncm1 = confusion_matrix(y_train, pre_values3)
print ("Confusion Matrix : \n", ncm1)
f8= f1_score(y_train,pre_values3)
f8
## SVM
from sklearn.svm import SVC
param_grid4 = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4]}

# Define the SVM model
model = SVC()

# Perform grid search using cross-validation
grid_search = GridSearchCV(model, param_grid=param_grid4, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')

# Evaluate the model on the testing set
accuracy = grid_search.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')


from sklearn.svm import SVC

SVM = SVC(kernel='rbf')

SVM.fit(X_train,y_train)
## test accuracy
predicted_values4 = SVM.predict(X_test)
s = metrics.accuracy_score(y_test, predicted_values4)
s

f9= f1_score(y_test,predicted_values4)
f9

scm= confusion_matrix(y_test, predicted_values4)
print ("Confusion Matrix : \n", scm)
## train accuracy
pre_values4 =  SVM.predict(X_train)
s1 = metrics.accuracy_score(y_train, pre_values4)
s1
f10= f1_score(y_train,pre_values4)
f10
scm1 = confusion_matrix(y_train, pre_values4)
print ("Confusion Matrix : \n", scm1)

## random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Create a random forest classifier object
rfc = RandomForestClassifier(random_state=42)

# Use grid search to find the optimal hyperparameters
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(grid_search.best_params_)


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=50, min_samples_split=2,max_depth=10)
RF.fit(X_train,y_train)

predicted_values5= RF.predict(X_test)
r = metrics.accuracy_score(y_test, predicted_values5)
r
rcm= confusion_matrix(y_test, predicted_values5)
print ("Confusion Matrix : \n", rcm)
f11= f1_score(y_test,predicted_values5)
f11
## train accuracy
pre_values5 =  RF.predict(X_train)
r1 = metrics.accuracy_score(y_train, pre_values5)
r1

rcm1 = confusion_matrix(y_train, pre_values5)
print ("Confusion Matrix : \n", rcm1)

f12= f1_score(y_train,pre_values5)
f12
##adboost algorithm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)]
}

# Create an AdaBoost classifier object
abc = AdaBoostClassifier(random_state=42)

# Use grid search to find the optimal hyperparameters
grid_search = GridSearchCV(estimator=abc, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(grid_search.best_params_)


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=200, learning_rate=1.0)
clf.fit(X_train, y_train)

## test accuracy
predicted_values6 = clf.predict(X_test)
a = metrics.accuracy_score(y_test, predicted_values6)
a
f13= f1_score(y_test,predicted_values6)
f13
acm= confusion_matrix(y_test, predicted_values6)
print ("Confusion Matrix : \n", acm)

## train accuracy
pre_values6 =  clf.predict(X_train)
a1 = metrics.accuracy_score(y_train, pre_values6)
a1

acm1 = confusion_matrix(y_train, pre_values6)
print ("Confusion Matrix : \n", acm1)
f14= f1_score(y_train,pre_values6)
f14
## gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()

# Define the hyperparameters to be tuned
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7],
}

# Perform grid search
grid_search = GridSearchCV(gb, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)

from sklearn.ensemble import GradientBoostingClassifier
glf = GradientBoostingClassifier(n_estimators=200, random_state=42,learning_rate=1,max_depth=3)
glf.fit(X_train, y_train)

## test accuracy
predicted_values7 = glf.predict(X_test)
g = metrics.accuracy_score(y_test, predicted_values7)
g
f15= f1_score(y_test,predicted_values7)
f15
gcm= confusion_matrix(y_test, predicted_values7)
print ("Confusion Matrix : \n", gcm)

## train accuracy
pre_values7 =  glf.predict(X_train)
g1 = metrics.accuracy_score(y_train, pre_values7)
g1
f16= f1_score(y_train,pre_values7)
f16
gcm1 = confusion_matrix(y_train, pre_values7)
print ("Confusion Matrix : \n", gcm1)


##XGBoosting
import xgboost as xgb
xgb_clf = xgb.XGBClassifier()

# Define the hyperparameters to be tuned
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7],
}

# Perform grid search
grid_search = GridSearchCV(xgb_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)


xlf = xgb.XGBClassifier(n_estimators=200, learning_rate=1,max_depth=7)
xlf.fit(X_train, y_train)

## test accuracy
predicted_values8 = xlf.predict(X_test)
xg= metrics.accuracy_score(y_test, predicted_values8)
xg
f15= f1_score(y_test,predicted_values8)
f15
xgcm= confusion_matrix(y_test, predicted_values8)
print ("Confusion Matrix : \n", xgcm)

## train accuracy
pre_values8 =  xlf.predict(X_train)
xg1 = metrics.accuracy_score(y_train, pre_values8)
xg1
f16= f1_score(y_train,pre_values8)
f16
xgcm1 = confusion_matrix(y_train, pre_values8)
print ("Confusion Matrix : \n", xgcm1)

##ANN
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
clf = MLPClassifier()
# Define the parameter grid to search over
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam']
}
# Define the grid search object
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
# Fit the grid search object to the iris dataset
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
# Generate a synthetic dataset for classification




 ####  save the model   
import pickle
pickle.dump(RandomForestClassifier,open('RandomForestClassifier','wb'))


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
  

####import model

model = pickle.load(open('RandomForestClassifier','rb'))

def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs

def prediction(Machine_ID,Assembly_Line_No,Hydraulic_Pressure,Coolant_Pressure,Air_System_Pressure,Coolant_Temperature,Spindle_Vibration,Spindle_Speed,Tool_Vibration,Voltage,Torque,Cutting):     
    prediction = classifier.predict(
        [[Machine_ID,Assembly_Line_No,Hydraulic_Pressure,Coolant_Pressure,Air_System_Pressure,Coolant_Temperature,Spindle_Vibration,Spindle_Speed,Tool_Vibration,Voltage,Torque,Cutting]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
    st.title("OPTIMIZATION OF MACHINE DOWNTIME")
      
   
      # the font and background color, the padding and the text to be displayed
      html_temp = """
      <div style ="background-color:yellow;padding:13px">
      <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
      </div>
      """
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    Machine_ID= st.text_input("Machine_ID", "Type Here")
    Hydraulic_Pressure = st.text_input("Hydraulic_Pressure", "Type Here")
    Coolant_Pressure = st.text_input("Coolant_Pressure", "Type Here")
    Air_System_Pressure = st.text_input("Air_System_Pressure ", "Type Here")
    Coolant_Temperature = st.text_input("Coolant_Temperature", "Type Here")
    Spindle_Vibration = st.text_input("Spindle_Vibration", "Type Here")
    Spindle_Speed = st.text_input(" Spindle_Speed", "Type Here")
    Tool_Vibration = st.text_input("Tool_Vibration", "Type Here")
    Voltage = st.text_input("Voltage", "Type Here")
    Torque = st.text_input("Torque", "Type Here")
    Cutting = st.text_input("Cutting", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(Machine_ID,Assembly_Line_No,Hydraulic_Pressure,Coolant_Pressure,Air_System_Pressure,Coolant_Temperature,Spindle_Vibration,Spindle_Speed,Tool_Vibration,Voltage,Torque,Cutting)
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()