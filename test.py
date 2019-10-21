import urllib.request
import pandas as pd

results = urllib.request.urlopen("http://127.0.0.1:8000/clinicalAnalysis/").read()
s = str(results,'utf-8')
list = s[1:-1].split(",")
df=pd.DataFrame(list).rename(columns={0: "Predictions"})

print (df)
