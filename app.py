import requests, pyttsx3, speech_recognition as sr

import utils, os
from random import choice
from decouple import config
from functions import api
from datetime import datetime
import brain

USERNAME = config('USER_NAME')
BOTNAME = config('BOTNAME')

# set the TTS engine
engine = pyttsx3.init('sapi5')

# Set the rate of the assistant
engine.setProperty('rate', 170)

# Set the volume of the assistant
engine.setProperty('volume', 1.0)

# Set the voice of the assistant (Male) 
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

api.USER_API_USERNAME = config('API_USERNAME')
api.USER_API_PASSWORD = config('API_PASSWORD')
api.BASE_URL = config('BASE_URL')

def speak(text:str) -> None :
    """Write the text passed in parameter"""
    print(f"🤖 {BOTNAME} parle...")
    engine.say(text)
    engine.runAndWait()

def greet_user() -> None:
    """Greets the user according to the time"""

    hour = datetime.now().hour
    if 12 <= hour < 16 :
        speak(f"Bon après-midi {USERNAME}")
    elif 16 <= hour < 19:
        speak(f"Good Evening {USERNAME}") 
    else :
        speak(f"Bonjour {USERNAME}")
    speak(f"Je suis {BOTNAME}. Comment puis-je vous aider")

def listen_to_user_input() -> str : 
    """Listen to user, make STT conversion using SAPI5"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('👂 En écoute...')
        r.pause_threshold = 2
        audio = r.listen(source)

    try:
        print('🤖 Traitement...')
        query = r.recognize_google(audio, language='fr-FR')
        if any([el in query for el in ['arrêter', 'sortir', 'arrêt', 'fin', 'terminer']]):
            speak('Au revoir')
            exit()
        else : 
            speak(choice(utils.opening_text))
            
    except Exception:
        speak('Désolé, Je ne comprend pas. Pouvez vous repérer ?')
        query = 'None'
    
    return query

if __name__ == '__main__' :
    output = os.system("cls")
    count = 0 
    Authenticated = False 
    while not Authenticated : 
        access_key = input("Saisir la clé d'accès : ")
        Authenticated = access_key == config("ACCESS_KEY")
        if not Authenticated : 
            print("Clé d'authentifcation incorrecte")
            count += 1 
        if count > 3 : 
            print("Echec de l'authentification")
            exit()
    print(f"Bienvenue dans votre sessions {config('USER_NAME')}")
    exit()
    api.auth_user()

    # greet_user()
    # query = listen_to_user_input()
    # print(query)
    # speak(f'Votre requêtes est : {query}')
    # speak('Voici le rapport d\'intrusions le plus à jour dans le réseau 192.168.8.0/24.')
    # resp = api.post("api/system_call/", {'method':'get_intrusion_report'})
    # print(resp.json()['message'])
    # input()
    while True:
        message = input("")
        intents = brain.class_predication(message.lower(), brain.words, brain.classes)
        result = choice(brain.get_intent(intents, brain.data)["responses"])
        print(result)
