import urllib.request
import pandas as pd

results = urllib.request.urlopen("http://127.0.0.1:8000/clinicalAnalysis/").read() #GET API results
s = str(results,'utf-8') #CONVERT RESPONSE TO A STRING
list = s[1:-1].split(",") #DROP "[" AND "]" & CONVERT TO LIST
df=pd.DataFrame(list).rename(columns={0: "Predictions"}) #CONVERT TO PANDAS DATAFRAME

print (df)
