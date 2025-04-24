import requests

wine = {'vector': [12.37, 0.94, 1.36, 10.6, 88.0, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520.0]}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=wine)
print(response.json())