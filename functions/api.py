import requests 

USER_API_USERNAME = None
USER_API_PASSWORD = None 
TOKEN = None
BASE_URL = None

# Authenticate the API user 
def get_user_token(username:str, password:str) -> str:
    global TOKEN
    payload = {'username':username, 'password':password}
    res = requests.post(f'{BASE_URL}/api-token-auth/', payload)
    return res.json()['token']

# Factory to authenticate the api user 
def auth_user() -> None:
    global TOKEN, USER_API_USERNAME, USER_API_PASSWORD
    if not TOKEN :  
        TOKEN = get_user_token(USER_API_USERNAME, USER_API_PASSWORD)

# function to make HTTP:GET request 
def get(resource:str):
    global TOKEN
    auth_user()
    return requests.get(f'{BASE_URL}/{resource}', headers={'Authorization': f'Token {TOKEN}'})

# Function to make HTTP:POST request
def post(resource:str, payload:dict):
    global TOKEN
    auth_user()
    return requests.post(
        f'{BASE_URL}/{resource}',
        data=payload, 
        headers={'Authorization': f'Token {TOKEN}'}
    )
