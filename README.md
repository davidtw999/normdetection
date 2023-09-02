# ccu-norm-detect-api
A REST service for norm detection

## Installation
```bash
python3 -m venv .
pip install django djangorestframework django-cors-headers requests
pip install django-rest-authtoken
python -m django --version

cd ccu-norm-dectect-api && pip install -r requirements.txt
```

### Init project
```bash
django-admin startproject ccu
cd ccu && django-admin startapp norm_detect
python manage.py migrate
```

### Create tokens/users
```bash
python manage.py createsuperuser --username johndoe --email johndoe@gmail.com
```

## Usage
```
API: 127.0.0.1:8000/ccu/norm-detect 
Token: d5d5248841ed1526a081dc9c29ad5a0e5f58a05c
```

Add the following into POST request header
```
'Authorization: Token d5d5248841ed1526a081dc9c29ad5a0e5f58a05c'
```

Example
```python
import requests
url_api = "http://localhost:8000/ccu/norm-detect"
headers = { "Authorization" : "Token d5d5248841ed1526a081dc9c29ad5a0e5f58a05c"}
input_data = {
    "uuid": "3d43ee00-70e3-4a16-84ae-4f43e9a3c8d3",
    "datetime": "2023-03-01 22:37:35.518093",
    "asr_text": "Hi, I'll get right down to business."
}
response = requests.post(url_api, json=input_data, headers=headers)
print(response.json())
```

Input
```json
{
    "uuid": "3d43ee00-70e3-4a16-84ae-4f43e9a3c8d3",
    "datetime": "2023-03-01 22:37:35.518093",
    "asr_text": "Hi, I'll get right down to business."
}
```

Output
```json
{"name": "noann", 
  "status": "EMPTY_NA", 
  "llr": "-0.67", 
  "trigger_id": "3d43ee00-70e3-4a16-84ae-4f43e9a3c8d3", 
  "timestamp": "1680760338.214526"
}
```
