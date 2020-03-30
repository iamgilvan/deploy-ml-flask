import requests

# Change the value of experience that you want to test
payload = {
	'exp':4.8
}

if __name__ == '__main__':
    # exemplo para requisição
    url = 'http://localhost:5000/api'
    response = requests.post(url,json=payload)
    print(response.json())