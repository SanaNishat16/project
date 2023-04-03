import pandas as pd
import numpy as np

makino=pd.read_csv(r"C:\Users\sanan\OneDrive\Desktop\data_makino.csv")
makino.info()
#EDA
#CENTRAL TENDENCY/ FIRST MEASURE OF BUSINESS MOMENT

makino.Hydraulic_Pressure.mean()
makino.Hydraulic_Pressure.median()
makino.Hydraulic_Pressure.mode()

makino.Coolant_Pressure.mean()
makino.Coolant_Pressure.median()
makino.Coolant_Pressure.mode()

makino.Air_System_Pressure.median()
makino.Air_System_Pressure.mean()
makino.Air_System_Pressure.mode()

makino.Coolant_Temperature.mean()
makino.Coolant_Temperature.median()
makino.Coolant_Temperature.mode()

makino.Hydraulic_Oil_Temperature.mean()
makino.Hydraulic_Oil_Temperature.median()
makino.Hydraulic_Oil_Temperature.mode()

makino.Spindle_Bearing_Temperature.mean()
makino.Spindle_Bearing_Temperature.median()
makino.Spindle_Bearing_Temperature.mode()

makino.Spindle_Vibration.mean()
makino.Spindle_Vibration.median() 
makino.Spindle_Vibration.mode() 

makino.Tool_Vibration.mean()
makino.Tool_Vibration.median()
makino.Tool_Vibration.mode()

makino.Spindle_Speed.mean()
makino.Spindle_Speed.median()
makino.Spindle_Speed.mode()

makino.Voltage.mean()
makino.Voltage.median()
makino.Voltage.mode()

makino.Torque.mean()
makino.Torque.median()
makino.Torque.mode()

makino.Cutting.mean()
makino.Cutting.median()
makino.Cutting.mode()

from scipy import stats

#MEASURES OF DISPERSION / 2ND MOMENT OF BUSINESS DECISION
makino.Hydraulic_Pressure.var()
makino.Hydraulic_Pressure.std()
range = max(makino.Hydraulic_Pressure) - min(makino.Hydraulic_Pressure)
range

makino.Coolant_Pressure.var()
makino.Coolant_Pressure.std()
range = max(makino.Coolant_Pressure) - min(makino.Coolant_Pressure)
range

makino.Air_System_Pressure.var()
makino.Air_System_Pressure.std()
range = max(makino.Air_System_Pressure) - min(makino.Air_System_Pressure)
range

makino.Coolant_Temperature.var()
makino.Coolant_Temperature.std()
range = max(makino.Coolant_Temperature) - min(makino.Coolant_Temperature)
range

makino.Hydraulic_Oil_Temperature.var()
makino.Hydraulic_Oil_Temperature.std()
range = max(makino.Hydraulic_Oil_Temperature) - min(makino.Hydraulic_Oil_Temperature)
range

makino.Spindle_Bearing_Temperature.var()
makino.Spindle_Bearing_Temperature.std()
range = max(makino.Spindle_Bearing_Temperature) - min(makino.Spindle_Bearing_Temperature)
range

makino.Spindle_Vibration.var()
makino.Spindle_Vibration.std()
range = max(makino.Spindle_Vibration) - min(makino.Spindle_Vibration)
range

makino.Tool_Vibration.var()
makino.Tool_Vibration.std()
range = max(makino.Tool_Vibration) - min(makino.Tool_Vibration)
range


makino.Spindle_Speed.var()
makino.Spindle_Speed.std()
range = max(makino.Spindle_Speed) - min(makino.Spindle_Speed)
range
makino.Voltage.var()
makino.Voltage.std()
range = max(makino.Voltage) - min(makino.Voltage)
range

#THIRD MOMENT OF BUSINESS DECISION

makino.Hydraulic_Pressure.skew()           
makino.Coolant_Pressure.skew()
makino.Air_System_Pressure.skew()
makino.Coolant_Temperature.skew()          
makino.Hydraulic_Oil_Temperature.skew()  
makino.Spindle_Bearing_Temperature.skew() 
makino.Spindle_Vibration.skew()           
makino.Tool_Vibration.skew()
makino.Spindle_Speed.skew()                
makino.Voltage.skew()                      
makino.Torque.skew()

#FOURTH MOMENT OF BUSINESS DECISION

makino.Hydraulic_Pressure.kurt()           
makino.Coolant_Pressure.kurt()
makino.Air_System_Pressure.kurt()
makino.Coolant_Temperature.kurt()          
makino.Hydraulic_Oil_Temperature.kurt()  
makino.Spindle_Bearing_Temperature.kurt() 
makino.Spindle_Vibration.kurt()           
makino.Tool_Vibration.kurt()
makino.Spindle_Speed.kurt()                
makino.Voltage.kurt()                      
makino.Torque.kurt()

#GRAPHICAL REPRESENTATION
import sweetviz as sv
report = sv.analyze(makino)
report.show_html

import matplotlib.pyplot as plt
import numpy as np
plt.bar(height = makino["Hydraulic_Pressure"], x = np.arange(0,2500,1))
plt.hist(makino.Hydraulic_Pressure)
plt.hist(makino.Coolant_Pressure)
plt.hist(makino.Air_System_Pressure)
plt.hist(makino.Coolant_Temperature)
plt.hist(makino.Hydraulic_Oil_Temperature, color = "red")
plt.hist(makino.Spindle_Bearing_Temperature, color = "blue")
plt.hist(makino.Spindle_Vibration, color = "purple")
plt.hist(makino.Tool_Vibration, color = "pink")
plt.hist(makino.Spindle_Speed, color = "green")
plt.hist(makino.Voltage, color = "violet")
plt.hist(makino.Torque, color = "brown")
plt.hist(makino.Cutting, color = "black")


#data preprocessing
makino.dtypes

#duplicate

duplicate=makino.duplicated()
duplicate
sum(duplicate)
 
import seaborn as sns
sns.boxplot(makino.Hydraulic_Pressure)
sns.boxplot(makino.Coolant_Pressure)
sns.boxplot(makino.Air_System_Pressure)
sns.boxplot(makino.Coolant_Temperature)
sns.boxplot(makino.Hydraulic_Oil_Temperature)
sns.boxplot(makino.Spindle_Bearing_Temperature)
sns.boxplot(makino.Spindle_Vibration)
sns.boxplot(makino.Tool_Vibration)
sns.boxplot(makino.Spindle_Speed)
sns.boxplot(makino.Voltage)
sns.boxplot(makino.Torque)
sns.boxplot(makino.Cutting)

#missing values
makino.isna().sum()

from sklearn.impute import SimpleImputer

mean_imputer=SimpleImputer(missing_values=np.nan,strategy = "mean")

makino["Air_System_Pressure"] = pd.DataFrame(mean_imputer.fit_transform(makino[["Air_System_Pressure"]]))
makino["Air_System_Pressure"].isna().sum()

makino["Hydraulic_Oil_Temperature"] = pd.DataFrame(mean_imputer.fit_transform(makino[["Hydraulic_Oil_Temperature"]]))
makino["Hydraulic_Oil_Temperature"].isna().sum()


makino["Spindle_Bearing_Temperature"] = pd.DataFrame(mean_imputer.fit_transform(makino[["Spindle_Bearing_Temperature"]]))
makino["Spindle_Bearing_Temperature"].isna().sum()

makino["Spindle_Vibration"] = pd.DataFrame(mean_imputer.fit_transform(makino[["Spindle_Vibration"]]))
makino["Spindle_Vibration"].isna().sum()

makino["Tool_Vibration"] = pd.DataFrame(mean_imputer.fit_transform(makino[["Tool_Vibration"]]))
makino["Tool_Vibration"].isna().sum()

median_imputer = SimpleImputer(missing_values=np.nan,strategy="median")
makino["Spindle_Speed"] = pd.DataFrame(median_imputer.fit_transform(makino[["Spindle_Speed"]]))
makino["Voltage"]= pd.DataFrame(median_imputer.fit_transform(makino[["Voltage"]]))
makino["Torque"] = pd.DataFrame(median_imputer.fit_transform(makino[["Torque"]]))
makino["Cutting"] = pd.DataFrame(median_imputer.fit_transform(makino[["Cutting"]]))

mode_imputer= SimpleImputer(missing_values=np.nan, strategy="most_frequent")
makino["Downtime"] = pd.DataFrame(mode_imputer.fit_transform(makino[["Downtime"]]))
makino["Machine_ID"] = pd.DataFrame(mode_imputer.fit_transform(makino[["Machine_ID"]]))
makino["Assembly_Line_No"] = pd.DataFrame(mode_imputer.fit_transform(makino[["Assembly_Line_No"]]))

makino.fillna(makino.median(),inplace=True)

makino.isna().sum()


from feature_engine.outliers import Winsorizer


winsor=Winsorizer(capping_method = 'iqr',tail = "both",fold=1.5,variables=["Hydraulic_Pressure","Coolant_Pressure", "Air_System_Pressure", "Coolant_Temperature" ,"Hydraulic_Oil_Temperature","Spindle_Bearing_Temperature" ,"Spindle_Vibration", "Tool_Vibration" ,"Voltage", "Torque"])
makino[["Hydraulic_Pressure","Coolant_Pressure","Coolant_Temperature","Air_System_Pressure","Hydraulic_Oil_Temperature","Spindle_Bearing_Temperature","Spindle_Vibration","Tool_Vibration","Spindle_Speed","Voltage","Torque","Cutting"]]=winsor.fit_transform(makino[['Hydraulic_Pressure','Coolant_Pressure','Coolant_Temperature','Air_System_Pressure','Hydraulic_Oil_Temperature','Spindle_Bearing_Temperature','Spindle_Vibration','Tool_Vibration','Spindle_Speed','Voltage','Torque','Cutting']])

###VARIANCE####
makino.var()
makino.var==0

###DROP COLUMNS####
makino.drop(['Date','Machine_ID','Assembly_Line_No'], axis = 1, inplace = True)
makino.dtypes
makino1=pd.get_dummies(makino)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
makino['Downtime'] =le.fit_transform(makino['Downtime'])



#Q-Q plot
import scipy.stats as stats 
import pylab

stats.probplot(makino1.Hydraulic_Pressure, dist='norm', plot=pylab)
stats.probplot(makino1.Coolant_Pressure, dist='norm', plot=pylab)
stats.probplot(makino1.Air_System_Pressure, dist='norm', plot=pylab)
stats.probplot(makino1.Coolant_Temperature, dist='norm', plot=pylab)
stats.probplot(makino1.Hydraulic_Oil_Temperature, dist='norm', plot=pylab)
stats.probplot(makino1.Spindle_Bearing_Temperature,dist='norm', plot=pylab)
stats.probplot(makino1.Spindle_Vibration,dist='norm',plot=pylab)
stats.probplot(makino1.Tool_Vibration, dist='norm', plot=pylab)
stats.probplot(makino1.Spindle_Speed, dist='norm', plot=pylab)
stats.probplot(makino1.Voltage, dist='norm', plot=pylab)
stats.probplot(makino1.Torque, dist='norm',plot=pylab)
stats.probplot(makino1.Cutting,dist='norm',plot=pylab)
stats.probplot(makino1.Downtime, dist='norm', plot=pylab)


#normalization

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

makino1_norm = norm_func(makino1)

#######MODEL BUILDING#######

from sklearn.model_selection import train_test_split
import joblib
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib as mpl
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics as skmet
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
# Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

makino.info()

makino1["Hydraulic_Pressure"] = np.log(makino1["Hydraulic_Pressure"])
makino1['Coolant_Temperature']=np.log(makino1['Coolant_Temperature'])
makino1['Cutting']=np.log(makino1[ 'Cutting'])
makino1['Torque']=np.log(makino1['Torque'])



makino1_n= norm_func(makino1.iloc[:, 1:])
norm_data = makino1_n.describe()

predicted=makino1.iloc[:,:-1]
target=makino1.iloc[:,-1]

#train test splite data
X_train, X_test, Y_train, Y_test = train_test_split(predicted, target, test_size = 0.2, stratify = target, random_state = 0) 

#### GridSearchCV ##
# Define the Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_train, Y_train)
#confusion_matrix on test data
cm=confusion_matrix(Y_test, model.predict(X_test))
print(cm)
#confusion_matrix on train data
cm2=confusion_matrix(Y_train, model.predict(X_train))
print(cm2)
#accuracy
print("train_accuracy",accuracy_score(Y_test, model.predict(X_test)))
print("test_accuracy",accuracy_score(Y_train, model.predict(X_train)))


# Define the hyperparameter grid
param_grid = {'n_estimators': [5,10,50,100,200],
              'max_depth': [3,4,5],
              'criterion': ['gini', 'entropy'],
              "bootstrap" : [True, False]}

# Create a GridSearchCV object and fit the data
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_train, Y_train)

# Print the best parameters and accuracy score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

rf_clf=grid_result.best_estimator_
rf_clf

cm3=confusion_matrix(Y_test, rf_clf.predict(X_test))
print("test:", cm3)

cm4=confusion_matrix(Y_train, rf_clf.predict(X_train))
print("train:", cm4)

print("test_accuracy:", accuracy_score(Y_test, rf_clf.predict(X_test)))
print("train_accuracy:", accuracy_score(Y_train, rf_clf.predict(X_train)))

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm3, display_labels = ['No_Machine_Failure', 'Machine_Failure'])
cmplot.plot()
cmplot.ax_.set(title = 'No_Machine_Failure  Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

print(skmet.classification_report(Y_test,rf_clf.predict(X_test)))
print(skmet.classification_report(Y_train,rf_clf.predict(X_train)))


# ### Hyperparameter optimization with RandomizedSearchCV
rf_Random = RandomizedSearchCV(estimator = model, param_distributions = param_grid, cv = 10, verbose = 2, n_jobs = -1)

rf_Random.fit(X_train, Y_train)

rf_Random.best_params_


cv_rf_random = rf_Random.best_estimator_

# Evaluation on Test Data
test_pred_random = cv_rf_random.predict(X_test)
cm_random1 = skmet.confusion_matrix(Y_test, test_pred_random)
print("test:",cm_random1)
accuracy_test_random = np.mean(test_pred_random == Y_test)
print("test_accuracy:",accuracy_test_random)
print("\n")

# Evaluation on Train Data
train_pred_random = cv_rf_random.predict(X_train)
cm_random2 = skmet.confusion_matrix(Y_train, train_pred_random)
print("train:",cm_random2)
accuracy_train_random = np.mean(train_pred_random == Y_train)
print("train_accuracy: ",accuracy_train_random)



cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm_random1, display_labels = ['No_Machine_Failure','Machine_Failure'])
cmplot.plot()
cmplot.ax_.set(title = 'Machine_Failure Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


print(skmet.classification_report(Y_test,cv_rf_random.predict(X_test))


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
  
# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

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
      
    # here we define some of the front end elements of the web page like 
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

