#take the request and return a response
from django.shortcuts import render
import sklearn.externals.joblib
import pandas as pd
from django.http import JsonResponse

def index(request):
    csv = pd.read_csv('docs/source/test_data_for_candidate_python.csv')
    X = csv.drop(["Has_Fancy_Title","Name_Length","Survived", "Predictions"],axis=1).astype({"Pclass":"int", "Sex":"bool", "Age":"int", "SibSp":"int", "Parch":"int", "Fare":"float", "Embarked_S": "bool", "Embarked_C":"bool"})
    #Above: drop unneeded columns and set column types
    model = sklearn.externals.joblib.load('docs/source/model_python.pkl')
    predictions = model.predict_proba(X) 
    results = predictions[:,1].tolist() #creates a list of the returned predictions
    return JsonResponse(results, safe=False) #false to allow for non dict to returned as json
