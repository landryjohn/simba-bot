import requests 

TOKEN = None
USERNAME = 'simba_api'
PASSWORD = 's1mb@1234'
BASE_URL = 'http://192.168.8.5:9000/api'

def get_user_token(username, password):
    global TOKEN
    payload = {'username':username, 'password':password}
    res = requests.post('http://192.168.8.5:9000/api-token-auth/', payload)
    return res.json()['token']

def auth_user():
    global TOKEN, USERNAME, PASSWORD
    if not TOKEN :  
        TOKEN = get_user_token(USERNAME, PASSWORD)

def api_get(resource):
    global TOKEN
    auth_user()
    return requests.get(f'{BASE_URL}/{resource}', headers={'Authorization': f'Token {TOKEN}'})

def api_post(resource, payload):
    global TOKEN
    auth_user()
    return requests.post(
        f'{BASE_URL}/{resource}',
        data=payload, 
        headers={'Authorization': f'Token {TOKEN}'}
    )

if __name__ == '__main__' :
    auth_user()

    res = api_get('users')
    print(res.text)
