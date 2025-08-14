from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from converti_in_secondi import converti_tempo_in_secondi

# import the data
df = pd.read_csv("Activities(1).csv")
selezione=['Distanza','Tempo','Calorie','Passo medio']
df_selezione=df[selezione]

df['Tempo_Secondi'] = df['Tempo'].apply(converti_tempo_in_secondi)
df['Passo_Medio_Secondi'] = df['Passo medio'].apply(converti_tempo_in_secondi)

tempo= df['Tempo_Secondi']
passi= df['Passo_Medio_Secondi']
calorie= df['Calorie']
distanza= df['Distanza']


def svr_calorie_vs_tempo(tempo, calorie):
    x_train = np.array(tempo).reshape(-1, 1)
    y_train = np.array(calorie)

    model=SVR()
    model.fit(x_train, y_train)

    predictions = model.predict(x_train)
    #print(f"Predizioni calorie bruciate: {predictions}")


    plt.figure(figsize=(10, 6))
    plt.scatter(tempo, calorie, color='blue', label='Dati reali')
    plt.scatter(tempo, predictions, color='red', label='Predizioni SVR')
    plt.xlabel('Tempo (secondi)')
    plt.ylabel('Calorie bruciate')
    plt.title('SVR: Calorie bruciate in base al tempo')
    plt.legend()
    plt.grid()
    plt.savefig('svr_calorie_vs_tempo.pdf')
   #plt.show()

svr_calorie_vs_tempo(tempo, calorie)


def svr_calorie_vs_passi(passi, calorie):
    x_train = np.array(passi).reshape(-1, 1)
    y_train = np.array(calorie)

    model=SVR()
    model.fit(x_train, y_train)

    predictions = model.predict(x_train)
    #print(f"Predizioni calorie bruciate: {predictions}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(passi, calorie, color='blue', label='Dati reali')
    plt.scatter(passi, predictions, color='red', label='Predizioni SVR')
    plt.xlabel('Passi medi (secondi)')
    plt.ylabel('Calorie bruciate')
    plt.title('SVR: Calorie bruciate in base ai passi medi')
    plt.legend()
    plt.grid()
    plt.savefig('svr_calorie_vs_passi.pdf')
   #plt.show()

svr_calorie_vs_passi(passi, calorie)