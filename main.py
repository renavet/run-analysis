import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


path="/home/renato/Desktop/runanalysis/run-analysis/Activities(1).csv"
df=pd.read_csv(path)
#print(dataframe.head())

selezione=['Distanza','Tempo','Calorie','Passo medio']
df_selezione=df[selezione]

# Funzione corretta per convertire il tempo (formato hh:mm:ss, mm:ss o anche in formato decimale) in secondi
def converti_tempo_in_secondi(tempo_str):
    if pd.isna(tempo_str):
        return None
    try:
        # Se il valore è già numerico (es. 27.1), gestiscilo qui
        return float(tempo_str)
    except ValueError:
        # Se il valore è una stringa (es. "4:27"), prosegui con la conversione
        parti = str(tempo_str).split(':')
        if len(parti) == 3:
            # Formato hh:mm:ss
            h, m, s = map(float, parti)
            return h * 3600 + m * 60 + s
        elif len(parti) == 2:
            # Formato mm:ss
            m, s = map(float, parti)
            return m * 60 + s
    return None


valori_tempo_da_mostrare = [30, 60, 90, 120]  # Esempio: Mostra solo 30, 60, 90, 120 minuti
etichette_personalizzate = ['30 min', '1 h', '1.5 h', '2 h']

df['Tempo_Secondi'] = df['Tempo'].apply(converti_tempo_in_secondi)
df['Passo_Medio_Secondi'] = df['Passo medio'].apply(converti_tempo_in_secondi)

passi_da_mostrare = [5 * 60, 6 * 60, 7 * 60, 8 * 60]  
etichette_passo = ['5:00', '6:00', '7:00', '8:00']


def T_C():

    res=stats.linregress(df['Tempo_Secondi'],df['Calorie'])


    plt.figure(figsize=(10,6))
    plt.scatter(df['Tempo_Secondi'], df['Calorie'], c=df['Calorie'], cmap='magma') # Aggiunto: 'c' per specificare i valori per la colormap
    plt.plot(df['Tempo_Secondi'], res.intercept + res.slope*df['Tempo_Secondi'], 'black', label='fitted line')
    plt.xlabel('Tempo') # Aggiunto: Etichetta asse x
    plt.ylabel('Calorie') # Aggiunto: Etichetta asse y
    plt.colorbar(label='Calorie') # Aggiunto: Barra colori per interpretare la colormap
    plt.title('Calorie vs Tempo') # Aggiunto: Titolo per il grafico

    plt.savefig('Tempo-Calorie_Scatter.pdf')


T_C()

def T_D():

    res=stats.linregress(df['Tempo_Secondi'],df['Distanza'])

    plt.figure(figsize=(10,6))
    plt.scatter(df['Tempo_Secondi'], df['Distanza'], c=df['Distanza'], cmap='magma') # Aggiunto: 'c' per specificare i valori per la colormap
    plt.plot(df['Tempo_Secondi'], res.intercept + res.slope*df['Tempo_Secondi'], 'black', label='fitted line')
    plt.xlabel('Tempo') # Aggiunto: Etichetta asse x
    plt.ylabel('Distanza') # Aggiunto: Etichetta asse y
    plt.colorbar(label='Distanza') # Aggiunto: Barra colori per interpretare la colormap
    plt.title('Tempo vs Distanza') # Aggiunto: Titolo per il grafico
 
    plt.savefig('Tempo-Distanza_Scatter.pdf')

T_D()

def Pm_C():


    res=stats.linregress(df['Passo_Medio_Secondi'],df['Calorie'])

    plt.figure(figsize=(10,6))
    plt.scatter(df['Passo_Medio_Secondi'], df['Calorie'], c=df['Calorie'], cmap='magma') # Aggiunto: 'c' per specificare i valori per la colormap
    plt.plot(df['Passo_Medio_Secondi'], res.intercept + res.slope*df['Passo_Medio_Secondi'], 'black', label='fitted line')
    plt.xlabel('Passo Medio') # Aggiunto: Etichetta asse x
    plt.ylabel('Calorie') # Aggiunto: Etichetta asse y
    plt.colorbar(label='Calorie') # Aggiunto: Barra colori per interpretare la colormap
    plt.title('Calorie vs Passo Medio') # Aggiunto: Titolo per il grafico

    
    
    plt.xticks(passi_da_mostrare, etichette_passo)

    plt.savefig('Passo Medio-Calorie_Scatter.pdf')
Pm_C()

def Pm_C_T():
    #res=stats.linregress(df['Passo_Medio_Secondi'],df['Calorie'])

    #plt.figure(figsize=(10,6),projection='3d')
    #plt.scatter(df['Passo_Medio_Secondi'], df['Calorie'],df['Tempo_Secondi'],c=df['Calorie'], cmap='magma') # Aggiunto: 'c' per specificare i valori per la colormap
    #plt.plot(df['Passo_Medio_Secondi'], res.intercept + res.slope*df['Passo_Medio_Secondi'], 'black', label='fitted line')
    #plt.xlabel('Passo Medio') # Aggiunto: Etichetta asse x
    #plt.ylabel('Calorie') # Aggiunto: Etichetta asse y
    
    #plt.colorbar(label='Calorie') # Aggiunto: Barra colori per interpretare la colormap
    #plt.title('Calorie vs Passo Medio vs Tempo') # Aggiunto: Titolo per il grafico

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(df['Passo_Medio_Secondi'], df['Tempo_Secondi'], df['Calorie'], c=df['Calorie'], cmap='magma')

    ax.set_xlabel('Passo Medio')
    ax.set_ylabel('Tempo (Secondi)')
    ax.set_zlabel('Calorie')
    ax.set_title('Calorie vs Passo Medio vs Tempo (3D)')

    plt.colorbar(scatter)
    plt.savefig('Passo Medio-Calorie-Tempo_Scatter.pdf')

    
    
    #plt.xticks(passi_da_mostrare, etichette_passo)

    

Pm_C_T()