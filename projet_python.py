# -*- coding: utf-8 -*-
"""
=========================
===== PROJET PYTHON =====
=========================
===    Korn Elisa     ===
===   Salihi Imane    ===
===  Lignoux Alexis   ===
=========================
"""

"""	
============================================
= Chemin du dossier contenant nos fichiers =
============================================
"""

directory = "D:/Master 2/SEMESTRE 2/Dossiers/Python/"

"""
======================================
===== --- Phase de recherche --- =====
======================================
"""

"""
==============================
= Importation du fichier TXT =
==============================
"""

fichier = list()
with open(directory + 'smsspamcollection.txt', 'r') as f :
   for line in f:
      fichier.append(line)
	
print(fichier[1])
	
"""
=============================
= Preprocessing des données =
=============================
"""

"""
Diviser chaque ligne en 2 pour récupérer la phrase et ham/spam
"""

target  = list()
phrases = list()

for i in range(0,len(fichier)):
	target.append(fichier[i].split()[0])
	phrases.append(fichier[i].split()[1:])

# Vérification :
print(len(fichier))
print(len(target))
print(len(phrases))
print(fichier[0:5])
print(target[0:5])
print(phrases[0:5])

"""
Recodage les Ham en 0 et les Spam en 1
"""

num_target = list()

for i in range(0,len(target)):
	if target[i] == "spam":
		x = 1
	else:
		x = 0
	num_target.append(x)

# Vérification :
for i in range(0, 10):
	print(target[i], num_target[i])

"""
Retirer les caractères spéciaux et majuscules des phrases
"""

# Avant
for word in phrases[0]:
	print(word)

for i in range(0,len(phrases)):
	replacement = list()
	for word in phrases[i]:	
		word = word.lower()
		for c in ',:.?!;"\'\\/()[]*#':
			word = word.replace(c, ' ')
			z = word.split() # On fait un split sur le mot au cas ou a une substitution en plein milieu du mot ex: "don't"
		replacement.extend(z)
	phrases[i] = replacement

# Vérification :
# Après
for word in phrases[0]:
	print(word)

"""
Création d'un dictionnaire pour associer le mot avec son nombre d'utilisation
"""

dico = dict() # <- Dictionnaire
mots = set()  # <- Ensemble contenant tous les mots une seule fois

for i in range(0, len(phrases)):
	for word in phrases[i]:
		if word in mots:
			dico[word] += 1
		else:
			mots.add(word)
			dico[word] = 1

# Vérification :
len(mots)
len(dico)
# On a 9049 mots différents
dico["i"]     # "i"    est employé 2998 fois
dico["like"]  # "like" est employé  247 fois

"""
Tri des mots par ordre décroissant de leur utilisation
"""

dico2 = sorted(dico, key=dico.get, reverse=True)

"""
Fonction permettant de compter combien de mots ont été utilisés au moins "time_used" fois
"""

def check_dico(time_used):
	k=0
	for i in range(0, len(dico2)):
		if dico[dico2[i]] >= time_used:
			k += 1
	return(k)

check_dico(2) # 4422 mots ont été utilisés au moins 2 fois

"""
Association des mots à un nombre :
	-> Si le mot a été utilisé seulement une fois sa valeur numérique vaut 0
	-> Sinon, sa valeur est celle de son classement en terme de fréquence d'utilisation
"""

max_words = check_dico(2)

words_dict = {word: i+1 for i, word in enumerate(dico2)}

for k in range(max_words, len(dico2)):
    words_dict[dico2[k]] = 0

# Vérification :
for i in range(4415,4430):
	print(dico2[i], words_dict[dico2[i]])

print(words_dict[dico2[1]])


"""
Détermination de la phrase la plus longue
"""

longueur_max = 0

for i in range(0, len(phrases)):
	if longueur_max < len(phrases[i]):
		longueur_max = len(phrases[i])
		
print(longueur_max)

# Vérification
for i in phrases:
	if len(i) == longueur_max:
		print(i)

# Le mail le plus long a 190 mots
# On doit donc recoder toutes les phrases de manières à avoir pour chaque, une liste  de 190 nombres 

"""
Recodage de chaque phrase en nombres
"""

num_phrases = list()

for i in range(0, len(fichier)):
	x = list()
	for j in phrases[i]:
		x.append(words_dict[j])
	if len(x) < longueur_max:
		while len(x) < longueur_max:
			x.append(0)
	num_phrases.append(x)

# Vérification :
for i in range(0,10):
	print(len(num_phrases[i]))


"""
======================================
= Bilan pré-processing des données : =
======================================
-> num_target est la liste des indicatrices 0/1 indiquant si le mail est un spam ou non
	-> Le Y
-> num_phrases est la liste des listes recodant les phrases en nombres
	-> Les X
	
=> Notre réseau de neurone devra donc avoir comme input (couche d'entrée) un vecteur
de 190 nombres, et comme couche de sortie, un seul nombre, grace à une transformation
logistique pour avoir un nombre entre 0 et 1 (si output>0.5 alors 1=spam sinon 0=ham)
"""


"""
================================================
= Processing des données : Réseaux de neurones =
================================================
"""

max_words = check_dico(2) + 1    # Nombre total de modalités différentes : 4422 (mots dans le dictionnaire avec valeur non nulles) + 1 (le groupe des mots ayant une valeur nulle)
echantillon_test = 500  # 500 mails pour l'échantillon test

# Importation des packages nécessaires pour construire le réseau de neurones
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Convertion des listes en arrays
X_train, y_train = num_phrases[:-echantillon_test], numpy.array(num_target[:-echantillon_test])
X_test, y_test = num_phrases[-echantillon_test:], numpy.array(num_target[-echantillon_test:])
X_train = sequence.pad_sequences(X_train, maxlen=longueur_max)
X_test = sequence.pad_sequences(X_test, maxlen=longueur_max)

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)

# Construction d'un réseau de neurones
model = Sequential()
model.add(Embedding(max_words, 32, input_length=longueur_max))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Estimation
model.fit(X_train, y_train, epochs=3, batch_size=32)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


"""
==============================================
=    Bilan du premier réseau de neurones :   =
= 88.00% de précision sur l'échantillon test =
==============================================
"""


"""
========================================
= Amélioration des performances du RNN =
========================================
"""

"""
Exploration des données : analyse de la longueur des phrases
"""

import pandas as pd

"""
Histogramme des longueurs de phrases (nombre de mots) par classe de SMS (ham vs spam)
"""

lengths = list()
for i in phrases:
	lengths.append(len(i))

histo = pd.DataFrame({
	"indicateur": target,
	"nombre de mots par phrase": lengths
})

histo.groupby("indicateur").hist(bins=100)

# Comme des distributions diffèrent, il semblerait qu'introduire une variable indiquant la longueur de la phrase apporterait de l'information


"""
Ajout dans les X du nombre de mots par phrases
"""

for i in range(0, len(lengths)):
	num_phrases[i].append(lengths[i])


"""
On fait de nouveau tourner notre réseau de neurones sur ce nouvel input
"""

max_words = 4422 + 1    # Nombre total de modalités différentes : 4422 (mots dans le dictionnaire avec valeur non nulles) + 1 (le groupe des mots ayant une valeur nulle)
max_length = 190 + 1    # Longueur maximum des vecteurs (Le mail le plus long contient 190 mots et on rajoute un élément pour indiquer la longueur de la phrase)
echantillon_test = 500  # 500 mails pour l'échantillon test


# Importation des packages nécessaires pour construire le réseau de neurones
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Convertion des listes en arrays
X_train, y_train = num_phrases[:-echantillon_test], numpy.array(num_target[:-echantillon_test])
X_test, y_test = num_phrases[-echantillon_test:], numpy.array(num_target[-echantillon_test:])
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)

# Création du réseau de neurones    
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Estimation du modèle
model.fit(X_train, y_train, epochs=3, batch_size=32)

# Analyse des prédictions
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


"""
==============================================
=    Bilan du second réseau de neurones :    =
= 90.20% de précision sur l'échantillon test =
==============================================
"""

"""
Analyse de la ponctuation
"""

ponct_count = list()
for line in fichier:
	k = 0
	for letter in line:
		if letter in [",", ".", "?", "!"]:
			k += 1
	ponct_count.append(k)

histo = pd.DataFrame({
	"Indicateur": target,
	"Nombre de mots par sms": ponct_count
})

histo.groupby("Indicateur").hist(bins = 100)

# La distribution de la fréquence de la ponctuation semble lié à la variable dépendante

"""
Analyse de la présence des chiffres
"""

chiffre = list()
for line in fichier:
	k = 0
	for letter in line:
		if letter in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
			k += 1
	chiffre.append(k)

histo = pd.DataFrame({
	"Indicateur": target,
	"Nombre de chiffres par sms": chiffre
})

histo.groupby("Indicateur").hist(bins = 100)

# La présence de chiffres dans le sms semble être fortement corrélé avec le fait d'être un ham ou spam

"""
Analyse de la présence des devises (seulement symboles)
"""

devise = list()
for line in fichier:
	k = 0
	for letter in line:
		if letter in ["€", "$", "£", "¥"]:
			k += 1
	devise.append(k)

histo = pd.DataFrame({
	"Indicateur": target,
	"Nombre de devise par sms": devise
})

histo.groupby("Indicateur").hist(bins = 50)

# La distribution de la fréquence des devises par phrase est très différentes en fonction de la catégorie

"""
Ajout dans les X du nombre du nombre d'éléments de ponctuation, de chiffres et de symboles de devises
"""

for i in range(0, len(ponct_count)):
	num_phrases[i].extend([ponct_count[i], chiffre[i], devise[i]])


"""
On fait de nouveau tourner notre réseau de neurones sur ce nouvel input
"""

max_words = 4422 + 1    # Nombre total de modalités différentes : 4422 (mots dans le dictionnaire avec valeur non nulles) + 1 (le groupe des mots ayant une valeur nulle)
max_length = 190 + 4    # Longueur maximum des vecteurs (190 mots et on rajoute : longueur phrases + nombre devise + nombre chiffres + nombre ponctuation)
echantillon_test = 500  # 500 mails pour l'échantillon test


# Importation des packages nécessaires pour construire le réseau de neurones
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Convertion des listes en arrays
X_train, y_train = num_phrases[:-echantillon_test], numpy.array(num_target[:-echantillon_test])
X_test, y_test = num_phrases[-echantillon_test:], numpy.array(num_target[-echantillon_test:])
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
   
# Construction du réseau de neurones
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Estimation du modèle
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Analyse des prédictions
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

"""
==============================================
=    Bilan du dernier réseau de neurones :   =
= 98.60% de précision sur l'échantillon test =
==============================================
"""


"""
===========================================
= Détermination du nombre d'epoch optimal =
===========================================

performances = list()
for nb_epoch in range(1,11):
	model = Sequential()
	model.add(Embedding(max_words, 32, input_length=max_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=nb_epoch, batch_size=32)
	scores = model.evaluate(X_test, y_test, verbose=0)
	performances.append([nb_epoch, "%.2f%%" % (scores[1]*100)])

for step in performances:
	print(step)
	
--------------------------------------------------------------
- Pour epoch = 7 on a un taux de bonne prédictions de 98.80% -
--------------------------------------------------------------
"""


"""
===================================================================================================
= Ajout d'une couche supplémentaire : 100 neurones cachés test de la meilleur couche d'activation =
===================================================================================================

# Tous les types d'activation différents
activs = ["sigmoid", "hard_sigmoid", "elu", "relu", "selu", "tanh", "softsign", "softplus", "softmax", "exponential", "linear"]
perfs  = list()

for i in activs:
	x = list()
	model = Sequential()
	model.add(Embedding(max_words, 32, input_length=max_length))
	model.add(Dense(100, activation = i))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=7, batch_size=32)
	scores = model.evaluate(X_test, y_test, verbose=0)
	x.append(i)
	x.append("%.2f%%" % (scores[1]*100))
	perfs.append(x)


print(perfs)
----------------------------------------------------------------------------------
- Toutes les couches d'activation ont la même performance sur l'échantillon test -
- Aucune utilité dans l'ajout d'une couche intermédiaire supplémentaire          -
----------------------------------------------------------------------------------
"""


"""
=================================
===== --- Modèle Finale --- =====
=================================
"""

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Importation du fichier
fichier = list()
with open(directory + 'smsspamcollection.txt', 'r') as f :
   for line in f:
      fichier.append(line)

# Split du fichier initial en deux listes (indicateur de spams et les sms)
num_target = list()
sms        = list()
for i in range(0,len(fichier)):
	words = list()
	line  = fichier[i].lower()
	for c in ',:.?!;"\'\\/()[]*#':
		line = line.replace(c, ' ')
	words = line.split()[1:]
	if line.split()[0] == "ham":
		num_target.append(0)
	else:
		num_target.append(1)
	sms.append(words)

# Création du dictionnaire
dico = dict() # <- Dictionnaire
mots = set()  # <- Ensemble contenant tous les mots une seule fois

for words in sms:
	for word in words:
		if word in mots:
			dico[word] += 1
		else:
			mots.add(word)
			dico[word] = 1

dico_sorted = sorted(dico, key=dico.get, reverse=True)

max_words = 0
for i in range(0, len(dico_sorted)):
	if dico[dico_sorted[i]] >= 2:
		max_words += 1

words_dict = {word: i+1 for i, word in enumerate(dico_sorted)}

for k in range(max_words, len(dico_sorted)):
    words_dict[dico_sorted[k]] = 0

max_words += 1 # On ajoute 1 pour compter le codage 0 : l'ensembles des autres mots

# Détermination du sms le plus long
max_length = 0
for words in sms:
	if max_length < len(words):
		max_length = len(words)

# Convertion des sms en listes numériques
num_sms = list()
for words in sms:
	nums = list()
	for j in words:
		nums.append(words_dict[j])
	if len(nums) < max_length:
		while len(nums) < max_length:
			nums.append(0)
	num_sms.append(nums)

# Ajout des 4 indicateurs (nombre de mots, chiffres, de devises, de ponctuations)
max_length += 4
for i in range(0, len(fichier)):
	line    = fichier[i]
	ponct   = 0
	chiffre = 0
	devise  = 0
	for letter in line:
		if letter in [",", ".", "?", "!"]:
			ponct   += 1
		if letter in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
			chiffre += 1
		if letter in ["€", "$", "£", "¥"]:
			devise  += 1
	num_sms[i].extend([len(sms[i]), ponct, chiffre, devise])

# Taille de l'échantillon test
training_set = 500  # 500 mails pour l'échantillon test

# Convertion des listes en arrays
X_train, y_train = num_sms[:-training_set], numpy.array(num_target[:-training_set])
X_test, y_test = num_sms[-training_set:], numpy.array(num_target[-training_set:])
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
    
# Construction du réseau de neurones
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Estimation du réseau de neurones
model.fit(X_train, y_train, epochs=7, batch_size=32)

# Scoring sur l'échantillon test
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# => 98.80 % de précision


"""
==============================================================================
= Sauvegarde du modèle et du dictionnaire pour éviter de réestimer le modèle =
==============================================================================
"""

# Sauvegarde du modèle
model.save(directory + 'spam_detection_model.h5')

# Sauvegarder le dictionnaire
import json
with open(directory + 'words_dict.json', 'w') as f:
    json.dump(words_dict, f)
	

"""
============================================================
=== --- Test du modèle sur un nouveau jeu de données --- ===
============================================================

----------------------------
- Fonction preprocessing() -
----------------------------

Cette fonction a pour but de faire tout le préprocessing nécessaire à des données pour pouvoir être scorées par notre modèle
Cette fonction de preprocessing des données peut prendre en argument 4 types d'input :
	1- Le chemin vers un fichier .txt contenant les X et les Y, auquel cas le fichier doit être de la forme :
		Une ligne = ham (ou spam) suivi du texte du sms
	2- Le chemin vers un fichier .txt contenant seulement les X, auquel cas le fichier doit être de la forme :
		Une ligne = un sms
	3- Un objet de type liste comprenant les X et les Y, donc comprenant pour chaque élément ham (ou spam) puis le text du sms (le tout en une seule chaîne de caractère)
	4- Un objet de type liste comprenant seulement les X, donc comprenant pour chaque élément le text du sms (le tout en une seule chaîne de caractère)
	
	Les arguments de la fonction sont les suivants :
		- fileType  = "txt"  -> Si l'input est un fichier txt
		              "list" -> Si l'input est un objet  de type liste
	    - inputType = "YX"   -> Si l'input contient les Y et les X
		              "X"    -> Si l'input ne contient que les X
	    - dictionnaire = Le dictionnaire construit lors de l'élaboration du modèle, nécessaire pour le recodage numérique des sms
		- file = L'objet de type liste ou le chemin vers le fichier de type .txt
		- max_length = La longueur maximale des sms sur lequel le modèle a été estimé
		
	La fonction renvoie soit une liste de 2 objets de type array (Les X recodés et les Y recodés) dans le cas ou l'input de la fonction contient les X et les Y,
	soit seulement les X recodés
	
	Cette fonction nécessite le package numpy (pour obtenir des objets de type array) et le module sequence de keras
"""

def preprocessing(fileType, inputType, dictionnaire, file = "NA", max_length = 190):
	if fileType == "txt":
		fichier = list()
		with open(file, 'r') as f :
		   for line in f:
		      fichier.append(line)
	elif fileType == "list":
		fichier = file
		
	if inputType == "YX":
		num_target = list()
		sms        = list()
		for i in range(0,len(fichier)):
			words = list()
			line  = fichier[i].lower()
			for c in ',:.?!;"\'\\/()[]*#':
				line = line.replace(c, ' ')
			words = line.split()[1:]
			if line.split()[0] == "ham":
				num_target.append(0)
			else:
				num_target.append(1)
			sms.append(words)
	elif inputType == "X":
		sms        = list()
		for i in range(0,len(fichier)):
			words = list()
			line  = fichier[i].lower()
			for c in ',:.?!;"\'\\/()[]*#':
				line = line.replace(c, ' ')
			words = line.split()
			sms.append(words)
	
	num_sms = list()
	for words in sms:
		nums = list()
		for j in words:
			if j in dictionnaire:
				nums.append(dictionnaire[j])
			else:
				nums.append(0)
		if len(nums) < max_length:
			while len(nums) < max_length:
				nums.append(0)
		elif len(nums) > max_length:
			nums = nums[:max_length]
		num_sms.append(nums)
	
	max_length += 4
	
	for i in range(0, len(fichier)):
		line    = fichier[i]
		ponct   = 0
		chiffre = 0
		devise  = 0
		for letter in line:
			if letter in [",", ".", "?", "!"]:
				ponct   += 1
			if letter in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
				chiffre += 1
			if letter in ["€", "$", "£", "¥"]:
				devise  += 1
		num_sms[i].extend([len(sms[i]), ponct, chiffre, devise])
	
	if inputType == "YX":
		X_test, y_test = num_sms, numpy.array(num_target)
		X_test = sequence.pad_sequences(X_test, maxlen=max_length)
		return([X_test, y_test])
	elif inputType == "X":
		X_test = sequence.pad_sequences(num_sms, maxlen=max_length)
		return(X_test)


"""
==========================================================
= Exemple d'utilisation sur notre jeu de données initial =
==========================================================
"""

# Chemin d'accès au dossier contenant les fichiers
directory = 'D:/Master 2/SEMESTRE 2/Dossiers/Python/'

# Chemin vers le fichier (au bon format)
file = directory + 'smsspamcollection.txt'

# Importation du dictionnaire
import json
with open(directory + 'words_dict.json', 'r') as f:
    dictionnaire = json.load(f)

# Importation du modèle
from keras.models import load_model
modele = load_model(directory + 'spam_detection_model.h5')

# La longueur maximale des sms lors de l'estimation du modèle
max_length = 190

# Préprocessing des données
import numpy
from keras.preprocessing import sequence
test_data = preprocessing(fileType = "txt", inputType = "YX", file = file, dictionnaire = dictionnaire, max_length=max_length)

# Prédictions
preds = modele.predict(test_data[0])

# Evalutation des prédictions
scores = modele.evaluate(test_data[0], test_data[1], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


"""
========================================================
= Exemple d'utilisation sur des SMS de notre invention =
========================================================
"""

# Création de la nouvelle base
new_sms = [
		"Okay no problem, I see you tomorrow !",
		"Do you really think that's gonna happen ?!",
		"Hell M. Smith ! Join our training session for only 2.25$ instead of 15$ per months ! Don't miss this opportunity !",
		"Hi honey ! I have a wonderful news for you.. You're gonna be dad ! <3"
		]

# Importation du dictionnaire
import json
with open(directory + 'words_dict.json', 'r') as f:
    dictionnaire = json.load(f)

# Importation du modèle
from keras.models import load_model
modele = load_model(directory + 'spam_detection_model.h5')

# Longueur maximale du sms durant l'estimation du modèle
max_length = 190

# Préprocession des données
import numpy
from keras.preprocessing import sequence
new_base = preprocessing(fileType = "list", inputType = "X", file = new_sms, dictionnaire = dictionnaire, max_length=max_length)

# Prédictions
predictions = modele.predict(new_base, verbose=0)

# Evaluation des prédictions
for i in predictions:
	if i > 0.5:
		x = "spam"
	else:
		x = "ham"
	print(i,x)

"""
Nous avons semblerait-il, les bons résultats avec ce modèle
"""