from django.shortcuts import render
from django.http import JsonResponse
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split 
 

#

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
        result1=""
        
        Pregnancies =float(request.GET['Pregnancies'])
        Glucose = float(request.GET['Glucose'])
        BloodPressure = float(request.GET['BloodPressure'])
        SkinThickness = float(request.GET['SkinThickness'])
        Insulin = float(request.GET['Insulin'])
        BMI = float(request.GET['BMI'])
        DiabetesPedigreeFunction = float(request.GET['DiabetesPedigreeFunction'])
        Age = float(request.GET['Age'])
   

        data_frame = pd.read_csv("C:/Users/nazil/Downloads/MY_ML/Notebook/Modified.csv")
        X = data_frame[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age']].values
        y = data_frame['Outcome'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=78) 
        
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
        rf_model.fit(X_train,y_train.ravel())
        user_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        predict = rf_model.predict(user_data)
    
        if predict == 1:
            result1 = "Positive"
        else:
            result1 = "Negative"

   
        return render(request, 'predict.html',{"prediction":result1})
