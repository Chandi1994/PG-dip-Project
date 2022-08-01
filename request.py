import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'age':25})

print(r.json())
