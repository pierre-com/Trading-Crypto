import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential #type de modèle adapté à notre série
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization #def dans le rapport
from tensorflow.keras.callbacks import TensorBoard # pour le reporting (affichage graph etc)
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint #pour le suivi pendant l'execution
import time
from sklearn import preprocessing

######## ici on def notre projection #######
SEQ_LEN = 60  # la longueur de la séquence précédente à collecter pour le RNN
FUTURE_PERIOD_PREDICT = 3  # à quelle distance dans le futur essayons-nous de prédire ?
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10  # nombre de total du parcours de nos données
BATCH_SIZE = 64  #  taille de nos échantillons
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}" #nom pour le suivi et reporting


######## ici on def notre fonction de décision #######
def classify(current, future):
    if float(future) > float(current): # si le cours prévu plus haut que l'acctuel
        return 1 # on achete
    else:
        return 0 #on vend

######## ici on def notre fonction pour le mise en forme des données pour les entrainements #######
def preprocess_df(df):
    df = df.drop("future", 1)  

    for col in df.columns:  # on itère sur nos colonnes
    ######## on normalise nos datas (une mise à l'échelle des cours des monnaies entre elle) ########
        if col != "target":  #on normalize tout sauf la target 
            df[col] = df[col].pct_change()  # La variation en pourcentage "normalise" les différentes monnaies (chaque crypto-monnaie a des valeurs très différentes), on s'intèresse à leur variation relative.
            df.dropna(inplace=True)  # sup les nans du au pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale entre 0 et 1.

    df.dropna(inplace=True)  # cleanup again... jic. Those nasty NaNs love to creep in.

    sequential_data = []  #list contenant les séquences 
    prev_days = deque(maxlen=SEQ_LEN)  # Séquences actuelles. On utilise deque qui garde la longueur max en sortant les vieilles valeurs quand les nouvelles rentres 
    ######## on séquence nos datas en plusieurs lot de 60 secondes ########
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  
        if len(prev_days) == SEQ_LEN:  # on vérifie qu'on a 60 seq
            sequential_data.append([np.array(prev_days), i[-1]])  

    random.shuffle(sequential_data)  # on mélange le seq data

    buys = []  # liste de toutes nos séquences d'achats et nos targets
    sells = []  # liste de toutes nos séquences de vente et nos targets

    for seq, target in sequential_data:  # parcourt notre lot de datas
        if target == 0:  # si c'est une vente
            sells.append([seq, target])  # on ajoute à la liste des ventes
        elif target == 1:  # sinon c'est un ordre d'achat
            buys.append([seq, target])  # on ajoute à la liste des achats
    # on mélange nos listes #
    random.shuffle(buys) 
    random.shuffle(sells)  

    lower = min(len(buys), len(sells))  # on récupère la plus courte?
    #on s'assure que les deux listes sont seulement jusqu'à la longueur la plus courte#
    #pour avoir des listes de même taille#
    buys = buys[:lower]  
    sells = sells[:lower]  

    sequential_data = buys+sells  # additionne nos listes
    random.shuffle(sequential_data)  # on mélange pour que le modèle ne soit pas embrouillé avec une classe puis une autre

    X = []
    y = []

    for seq, target in sequential_data:  # parcourt notre nouvelle sequence de datas
        X.append(seq)  # X = sequences
        y.append(target)  # y = targets  (achat ou vente)

    return np.array(X), y  


######## ici on apprend à lire des CSV et récup ce qu'on veut dans 1 seul frame #######
main_df = pd.DataFrame() # begin empty

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration
    print(ratio)
    dataset = f'crypto_data/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
######## ici on insère la colonne prédiction dans le frame #######
#nos prédictions seront dans "FUTURE_PERIOD_PREDICT = 3" secondes
main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)

######## ici on utilise la fonction "classify" pour comparer #######
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
# elle nous laisse un tableau avec le cours futur et sont ordre d'achat/vente, en fct du temps
##
main_df.dropna(inplace=True)

##### ici on décpoupe nos data en deux, 95% pour l'entrainement et les 5% derniers pour la validation #######
times = sorted(main_df.index.values)  
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # on prend les 5% 
validation_main_df = main_df[(main_df.index >= last_5pct)]  # index
main_df = main_df[(main_df.index < last_5pct)]  # maintenant le main_df est toutes les données jusqu'au dernier 5%.

train_x, train_y = preprocess_df(main_df) # nos datas d'entrainement
validation_x, validation_y = preprocess_df(validation_main_df) #nos datas de test

### fonction affichage et vérification (tableaux retournés par preprocess_df doivent etre de même taille)###
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

#### definition de notre réseau de neurones ####
model = Sequential() # création du modele
# création 1ere couche de neuronnes #
# 32 noeuds de type : Long Short-Term Memory
model.add(LSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2)) #La couche Dropout place aléatoirement les unités d'entrée à 0 avec une fréquence de taux à chaque étape de la formation, ce qui permet d'éviter un overfitting.Dropout s'applique que lorsque l'apprentissage est défini sur True.
model.add(BatchNormalization())#La normalisation des lots applique une transformation qui maintient la sortie moyenne proche de 0 et l'écart type de la sortie proche de 1.

model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu')) #fonction d'activation par éléments passée comme argument d'activation, ici "rectified linear"
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax')) #dense layer pour l'output, 2 car choix binaire, et activation de la couche de sortie à softmax

####descente de gradient stochastique####
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

####Compilation du model #### 
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
####mise en place d'un tableau de suivi graphique ####
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
####mise en place suivi avancement de la compilation ####
filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  # nom de fichier unique qui comprendra l'époque et l'acc de validation pour cette époque
checkpoint = ModelCheckpoint("models/{}.model".format(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max') # sauvegarde les meilleurs


####training de notre modele####
# définition des variables avec les données
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)
#lancement de l'entrainement
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)
####reporting du score de notre modele####
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# sauvgarde du modele
model.save("models/{}".format(NAME))
