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
======================================
===== --- Phase de recherche --- =====
======================================
"""

'''
==============================
= Importation du fichier TXT =
==============================
'''	
	
fichier = list()
with open('D:/Master 2/SEMESTRE 2/Dossiers/Python/smsspamcollection/smsspamcollection.txt', 'r') as f :
   for line in f:
      fichier.append(line)
	
print(fichier[1])
	
'''
=============================
= Preprocessing des données =
=============================
'''

'''
Diviser chaque ligne en 2 pour récupérer la phrase et ham/spam
'''

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

'''
Recodage les Ham en 0 et les Spam en 1
'''

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

'''
Retirer les caractères spéciaux et majuscules des phrases
'''

# Avant
for word in phrases[0]:
	print(word)

for i in range(0,len(phrases)):
	replacement = list()
	for word in phrases[i]:	
		word = word.lower()
		for c in ',:.?!;"\'\\/()[]*#':
			word = word.replace(c, ' ')
			z = word.split()            # On fait un split sur le mot au cas ou a une substitution en plein milieu du mot ex: "don't"
		replacement.extend(z)
	phrases[i] = replacement

# Vérification :
# Après
for word in phrases[0]:
	print(word)

'''
Création d'un dictionnaire pour associer le mot avec son nombre d'utilisation
'''

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

'''
Tri des mots par ordre décroissant de leur utilisation
'''

dico2 = sorted(dico, key=dico.get, reverse=True)

'''
Fonction permettant de compter combien de mots ont été utilisés au moins "time_used" fois
'''

def check_dico(time_used):
	k=0
	for i in range(0, len(dico2)):
		if dico[dico2[i]] >= time_used:
			k += 1
	return(k)

check_dico(2) # 4422 mots ont été utilisés au moins 2 fois

'''
Association des mots à un nombre :
	-> Si le mot a été utilisé seulement une fois sa valeur numérique vaut 0
	-> Sinon, sa valeur est celle de son classement en terme de fréquence d'utilisation
'''

max_words = check_dico(2)

words_dict = {word: i+1 for i, word in enumerate(dico2)}

for k in range(max_words, len(dico2)):
    words_dict[dico2[k]] = 0

# Vérification :
for i in range(4415,4430):
	print(dico2[i], words_dict[dico2[i]])

print(words_dict[dico2[1]])


'''
Détermination de la phrase la plus longue
'''

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

'''
Recodage de chaque phrase en nombres
'''

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


'''
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
'''


'''
================================================
= Processing des données : Réseaux de neurones =
================================================
'''

max_words = 4422 + 1    # Nombre total de modalités différentes : 4422 (mots dans le dictionnaire avec valeur non nulles) + 1 (le groupe des mots ayant une valeur nulle)
max_length = 190        # Longueur maximum des vecteurs (Le mail le plus long contient 190 mots)
echantillon_test = 500  # 500 mails pour l'échantillon test


# Importation des packages nécessaires pour construire le réseau de neurones
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

X_train, y_train = num_phrases[:-echantillon_test], numpy.array(num_target[:-echantillon_test])
X_test, y_test = num_phrases[-echantillon_test:], numpy.array(num_target[-echantillon_test:])
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
    
model = Sequential()
# Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] // This layer can only be used as the first layer in a model.
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=32)

# Sauvegarder le modèle
# model.save('model_spam_detection.h5')

# Charger le modèle
# from keras.models import load_model
# model2 = load_model('model_spam_detection.h5')

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



'''
==============================================
=    Bilan du premier réseau de neurones :   =
= 88.00% de précision sur l'échantillon test =
==============================================
'''


'''
========================================
= Amélioration des performances du RNN =
========================================
'''

'''
Exploration des données : analyse de la longueur des phrases
'''

import pandas as pd

'''
Histogramme des longueurs de phrases (nombre de mots) par classe de SMS (ham vs spam)
'''

lengths = list()
for i in phrases:
	lengths.append(len(i))

histo = pd.DataFrame({
	"indicateur": target,
	"nombre de mots par phrase": lengths
})

histo.groupby("indicateur").hist(bins=100)

# Comme des distributions diffèrent, il semblerait qu'introduire une variable indiquant la longueur de la phrase apporterait de l'information


'''
Ajout dans les X du nombre de mots par phrases
'''

for i in range(0, len(lengths)):
	num_phrases[i].append(lengths[i])


'''
On fait de nouveau tourner notre réseau de neurones sur ce nouvel input
'''

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

X_train, y_train = num_phrases[:-echantillon_test], numpy.array(num_target[:-echantillon_test])
X_test, y_test = num_phrases[-echantillon_test:], numpy.array(num_target[-echantillon_test:])
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
    
model = Sequential()
# Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] // This layer can only be used as the first layer in a model.
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=32)

# Sauvegarder le modèle
# model.save('model_spam_detection.h5')

# Charger le modèle
# from keras.models import load_model
# model2 = load_model('model_spam_detection.h5')

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


'''
==============================================
=    Bilan du second réseau de neurones :    =
= 90.20% de précision sur l'échantillon test =
==============================================
'''

'''
Analyse de la ponctuation
'''

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


'''
Analyse de la présence des chiffres
'''

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

'''
Analyse de la présence des devises (seulement symboles)
'''

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

histo.groupby("Indicateur").hist(bins = 4)

# La distribution de la fréquence des devises par phrase est très différentes en fonction de la catégorie


'''
Ajout dans les X du nombre du nombre d'éléments de ponctuation, de chiffres et de symboles de devises
'''

for i in range(0, len(ponct_count)):
	num_phrases[i].extend([ponct_count[i], chiffre[i], devise[i]])


'''
On fait de nouveau tourner notre réseau de neurones sur ce nouvel input
'''

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

X_train, y_train = num_phrases[:-echantillon_test], numpy.array(num_target[:-echantillon_test])
X_test, y_test = num_phrases[-echantillon_test:], numpy.array(num_target[-echantillon_test:])
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
    
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=5, batch_size=32)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


'''
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
	
# Le nombre optimal est 7
'''



'''
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

Toutes les couches d'activation ont la même performance sur l'échantillon test
Aucune utilité dans l'ajout d'une couche intermédiaire supplémentaire
'''




"""
=================================
===== --- Modèle Finale --- =====
=================================
"""

"""
Alexis, t'as juste à comprimer tout ça et foutre ca en fonction !
Good luck !
Cordialement,
Moi-même
"""

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

fichier = list()
with open('D:/Master 2/SEMESTRE 2/Dossiers/Python/smsspamcollection/smsspamcollection.txt', 'r') as f :
   for line in f:
      fichier.append(line)

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

max_length = 0
for words in sms:
	if max_length < len(words):
		max_length = len(words)

num_sms = list()
for words in sms:
	nums = list()
	for j in words:
		nums.append(words_dict[j])
	if len(nums) < max_length:
		while len(nums) < max_length:
			nums.append(0)
	num_sms.append(nums)

max_length += 4 # Ajout des 4 indicateurs (nombre de mots, chiffres, de devises, de ponctuations)

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

print(num_sms[0])
len(num_sms)
len(num_sms[0])
training_set = 500  # 500 mails pour l'échantillon test

X_train, y_train = num_sms[:-training_set], numpy.array(num_target[:-training_set])
X_test, y_test = num_sms[-training_set:], numpy.array(num_target[-training_set:])
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
    
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=7, batch_size=32)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
















