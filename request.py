import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'FULL Name':'عبدالله محمد امين'})

print(r.json())