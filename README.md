# Notice d'utilisation

## Introduction


Les spams sont à notre époque, pour tous les utilisateurs de smartphone, potentiellement une nuisance ou dans le pire des cas
une tentative d'escroquerie. Ce sont souvent des SMS invitant à contacter un numéro surtaxé ou à cliquer sur un lien frauduleux. C'est pourquoi, il semblerait judicieux de trouver une méthode permettant de prédire au
vu de son contenu, si un SMS est un spam ou non.  
Il s'agira donc de l'objet de notre étude, arriver à prédire avec précision si un SMS est un spam ou non.


## Présentation des données


Pour ce faire, nous avons utilisé un jeu de données contenant 5 574 SMS, dont la nature est renseignée. Le terme _"ham"_
s'opposera ici au terme _"spam"_ et signifie _"message légitime"_. Voici un extrait de cette base, vous en présentant
ainsi la structure :


|ham/spam  |                        Contenu du SMS                                                                            |
| -------- | :---------------------------------------------------------------------------------------------                   |
| ham      | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...  |
| ham      | Ok lar... Joking wif u oni...                                                                                    | 
| spam     | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's                                                                                          |
| ham      | U dun say so early hor... U c already then say...                                                                |
|ham       | Nah I don't think he goes to usf, he lives around here though                                                    |

La première colonne indique si le message est un _"spam"_ ou un _"ham"_. La seconde colonne renseigne le contenu du SMS.
Sur ces 5 574 messages SMS, 86.6% des messages sont légitimes et 13.4% sont des spams.  

_Vous pouvez retrouver ce jeu de données en suivant ce [lien](http://archive.ics.uci.edu/ml/machine-learning-databases/00228/)_


## Méthodologie :

L'approche utilisée par la suite sera l'usage d'un réseau de neurones dans un cadre d'analyse de sentiment (NLP). Une
grosse part du travail va donc consister à retraiter notre base initiale pour la rendre exploitable par notre réseau de
neurones, qui ne peut prendre en argument que des inputs numériques.

Dans une première partie, nous allons construire un premier réseau de neurones à partir des SMS que nous auront
transformé en substituant un mot par un nombre. Puis, dans une seconde partie, nous tenterons d'améliorer les
performances prédictives de notre réseau de neurones par une analyse plus approfondie des données, principalement par
l'ajout de nouveaux prédicteurs.

#### Premier réseau de neurones

La première étape consiste à créer un dossier dans lequel nous stockerons tous les fichiers, et parmi ceux-ci, le
jeu de données initial. Il vous faudra si vous voulez réexécuter ce code, changer la modalité de la variable _directory_
pour lui indiquer le chemin jusqu'à votre dossier.

##### 1 - Pré-processing
>- Chargement les données depuis le fichier .txt
>- Division des données pour obtenir une liste contenant les indicateurs ham/spam et une liste comprenant le
contenu des messages.
>- Recodage des ham en 0 et spam en 1.
>- Construction d'un dictionnaire recensant les mots de l'ensemble des messages, et attribution d'une valeur numérique
non nulle différente pour chaque mot ayant été utilisé plus de 2 fois (valeur 0 pour tous les autres).
>- Recodage des SMS en remplaçant tous les mots par leur valeur numérique. La longueur des SMS recodés est standardisée
pour être de la longueur du SMS le plus long rencontré dans la base.
>- Transformation des objets de type liste en objet de type array.

##### 2 - Division des données en échantillons d'apprentissage et test
>- 500 observations forment l'échantillon test et le reste l'échantillon d'apprentissage.

##### 3 - Entraînement du réseau de neurones
>- Création d'un réseau de neurones avec 3 couches: la couche d'entrée, la couche intermédiaire qui est de type LSTM
(Long Short Term Memory) et enfin la couche de sortie avec une transformation logistique.
>- On entraîne ce réseau de neurone sur l'échantillon d'apprentissage.

##### 4 - Évaluation des capacités prédictives du modèle
>- Prédiction du modèle sur l'échantillon test
>- Calcul du taux de bonnes prédictions : 88.00 %

#### Amélioration des performances du réseau de neurones

La seconde phase consiste à améliorer les performances de notre modèle selon 3 axes : les inputs, la forme du réseau
de neurones, le choix de certains hyper-paramètres.

##### 1 - La recherche de nouveaux inputs
>- Analyse de la distribution en terme de longueur de SMS (nombre de mots), de nombre de chiffres, de nombre de
ponctuation ("?", "!",...), et enfin de nombre de devises ("€", "$", ...).
>- Incorporation de 4 inputs supplémentaire.
>- Augmentation des performances du réseau de neurones : 98,60% (échantillon test).

##### 2 - Choix optimal d'hyperparamètres
>- Choix du nombre de fois où le modèle est estimé (nombre d'epoch pour maximiser les performances prédictives).
>- Augmentation des performances du réseau de neurones : 98.80% (échantillon test).

##### 3 - Incorporation d'une nouvelle couche
>- Analyse des performances du modèle avec ajout d'une nouvelle couche de 100 neurones cachés, avec différentes
fonctions d'activation.
>- Pas d'amélioration de performance.

## Le Modèle Optimal

Le modèle optimal vous est présenté avec un code plus condensé, mais reprenant les étapes précédentes.
Le modèle finalement obtenu sera sauvegardé ainsi que le dictionnaire nécessaire pour son élaboration.

## Prédictions sur de nouveaux échantillons

Le modèle ayant été sauvegardé (et le dictionnaire), il n'est pas nécessaire pour scorer une nouvelle base de SMS de
réestimer le modèle, mais simplement de charger ce dernier avec son dictionnaire et procéder au pré-processing des
données avant de pouvoir faire les prédictions.

Pour ce faire, une fonction a été élaborée pour vous rendre la tâche plus facile et faire tout le travail de préparation
des données. Deux exemples vous sont fournis ainsi que le détail de ses arguments pour comprendre facilement comment
employer cette fonction.

Il vous sera ainsi facile de tester ce modèle sur des SMS que vous pourriez vous même retranscrire (comme illustré dans
l'ultime exemple).

## Pistes d'amélioration

Dans le but d'améliorer les performances du modèle il serait possible de jouer sur les points suivants :
>- La taille de l'échantillon sur lequel notre réseau de neurones peut s'entraîner (plus la taille d'échantillon est grande
et meilleures sont les performances)
>- Le dictionnaire de mots employé pour recoder les phrases lors du préprocessing
>- Introduire ou écarter de nouveaux inputs
>- Gérer la taille de l'input : doit-on raccourcir les SMS recodés ? 
>- Essayer de nouvelles structures de réseaux de neurones : le nombre de couches cachées, le nombre de neurones cachés sur
chaque couche
