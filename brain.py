import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.saving.save import load_model
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# initialisation de lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()
# création des listes
words = []
classes = []
doc_X = []
doc_y = []

# Utilisation d'un dictionnaire pour représenter un fichier JSON d'intentions
# Ensemble de intents du pro
data = {
    "intents": [
        {
            "tag": "grettings",
            "patterns": ["salut à toi!", "hello", "comment vas tu?", "salutations!", "enchanté", "hey"
                    "hey hey", "he", "heyyy"
                    "bonjour!",
                    "salut, comment ca va",
                    "bonjour, comment ca va",
                    "salut, comment vas-tu",
                    "comment vas-tu",
                    "enchantée.",
                    "salut, content de te connaitre.",
                    "un plaisir de te connaitre.",
                    "passe une bonne journée",
                    "quoi de neuf"],
            "responses" : ["Salut", "Bonjour", "Hello !", "Hi"]
        },
        {
            "tag": "services_status",
            "patterns": ["afficher l'état des services", "statut des services", "Je veux connaitre l'état des services", 
                    "comment fonctionne le serveur", "fonctionnement du serveur", 
                    "Etat de marche du serveur", "regime de fonctionnement", "fonctionnement des services"],
            "responses": ["Voici le statut des services", "voici le rapport de fonctionnement des services", "fonctionnement des services"]
        },
        {
            "tag": "signature_database",
            "patterns": ["afficher la base de signatures", "montrer la base virale", "afficher la liste des règles", 
                "règle de l'IDS", "signatures des attaques", "liste des attaques", "liste des règles", "Montre moi les règles"],
            "responses": ["Voici la base de signature", "Voici la base de signature la plus à jour"]
        },
        {
            "tag": "simba_rules",
            "patterns": ["quelles sont le fichier des règles d'alerte", "quelles sont tes règles", "montre moi le fichier règles", 
            "fichier de règle de simba", "affiche le fichier des règles", "fichier de règle"],
            "responses": ["voici le contenu du fichier de règle personnalisé"]
        },
        {
            "tag" : "intrusion_report",
            "patterns" : ["je veux le rapport d'intrusion dans le réseau", "rapport d'intrusion dans le réseau", 
                    "liste des intrusions dans le réseau", "rappoort d'alertes", "liste alertes", "log des alertes", 
                    "afficher les attaques", "affiches les alertes", "montre moi les alertes", "liste des attaques "], 
            "responses" : ["Voici la liste des alertes de ce jours"]
        },
        {
            "tag" : "send_intrusion_report", 
            "patterns" : ["envoi moi le rapport d'intrusion", "envoi du rapport d'alert", "envoyer le rapport par mail", 
                    "envoyer les alertes dans le réseau par mail"],
            "responses" : ["Envoi du rapport d'intrusion"]
        }, 
        {
            "tag" : "block_user_rule",
            "patterns" : ["bloque un utilisateur", "bloque une machine", "stop une machine", 
                        "stop une adresse machine", "arrêter un utilisateur"],
            "responses" : ["blocage d'une utilisateur"]
        },
        {
            "tag" : "add_rule", 
            "patterns" : ["ajouter une règle", "définir une règle","ajout d'une règle", 
                        "ajoute une règle", "je veux ajouter une règle", "je veux modifier les règles"], 
            "responses" : ["Ajout d'une règle"]
        },
        {
            "tag" : "firewall",
            "patterns" : ["pare-feu", "pare feu", "parefeu" "afficher la configuration du pare-feu", 
                        "afficher les règles du pare-feu", "afficher le pare-feu", "afficher la table ACL"
                        "configuration pare-feu"],
            "responses" : ["Voici la configuration actuel du pare-feu"]
        }, 
        {
            "tag" : "red_code",
            "patterns" : ["code rouge", "code code rouge", "arrêter tout les services", "éteindre le réseau", 
                        "arrêter les serveurs", "stoper les serveur", "éteindre les serveurs", 
                        "éteindre les services"],
            "responses" : ["Code rouge activé"]
        },
        {
            "tag": "ssh_connections",
            "patterns": ["afficher la liste des connexions SSH", "Afficher les dernière connexions SSH",
                    "connexion SSH"],
            "responses": ["Voici la liste des dernière connexions SSH"]
        },
        {
            "tag": "stop_simba_client",
            "patterns": ["Au revoir", "A plus", "Bye", "Stop", "cya", "Au revoir"],
            "responses": ["C'était sympa de vous parler", "à plus tard", "A plus!"]
        }
]}

# parcourir avec une boucle For toutes les intentions
# tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
# le tag associé à l'intention sont ajoutés aux listes correspondantes
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    # ajouter le tag aux classes s'il n'est pas déjà là
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatiser tous les mots du vocabulaire et les convertir en minuscule
# si les mots n'apparaissent pas dans la ponctuation
words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in string.punctuation]
# trier le vocabulaire et les classes par ordre alphabétique et prendre le
# set pour s'assurer qu'il n'y a pas de doublons
words = sorted(set(words))
classes = sorted(set(classes))
"""
def train_model() -> None:
    # liste pour les données d'entraînement
    training = []
    out_empty = [0] * len(classes)
    # création du modèle d'ensemble de mots
    for idx, doc in enumerate(doc_X):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        for word in words:
            bow.append(1) if word in text else bow.append(0)
        # marque l'index de la classe à laquelle le pattern atguel est associé à
        output_row = list(out_empty)
        output_row[classes.index(doc_y[idx])] = 1
        # ajoute le one hot encoded BoW et les classes associées à la liste training
        training.append([bow, output_row])
    # mélanger les données et les convertir en array
    random.shuffle(training)
    training = np.array(training, dtype=object)
    # séparer les features et les labels target
    train_X = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # définition de quelques paramètres
    input_shape = (len(train_X[0]),)
    output_shape = len(train_y[0])
    epochs = 200

    # modèle Deep Learning
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation="softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                optimizer=adam, metrics=["accuracy"])

    model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

    model.save('simba_model.hdf5')
    # del model 
"""

model = load_model('simba_model.hdf5')

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def class_prediction(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_intent(intents_list, json_intents):
    tag = intents_list[0]
    list_of_intents = json_intents["intents"]
    for intent in list_of_intents:
        if intent["tag"] == tag:
            break
    return intent

# lancement de l'agent
if __name__ == '__main__' : 
    print("Ready !")
    while True:
        message = input("")
        intents = class_prediction(message.lower(), words, classes)
        result = random.choice(get_intent(intents, data)["responses"])
        print(result)
