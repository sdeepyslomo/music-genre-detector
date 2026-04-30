import requests

#download a web page
response = requests.get("https://api.github.com")
print(response.status_code)