import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


###MADE WITH AI###

# 1. Caricamento e preparazione dei dati
# Assicurati di avere il file CSV nella stessa directory dello script.
try:
    data = pd.read_csv("Activities(1).csv")
except FileNotFoundError:
    print("Errore: File 'Activities(1).csv' non trovato.")
    exit()

# Pulisci i dati
data = data.replace('--', np.nan)

# Se presente, rimuovi la virgola come separatore delle migliaia nella colonna 'Passi'
if 'Passi' in data.columns:
    data['Passi'] = data['Passi'].astype(str).str.replace(',', '', regex=False)
    data['Passi'] = pd.to_numeric(data['Passi'], errors='coerce')

# Colonne che devono essere numeriche, convertendo la virgola in punto
cols_to_convert = ['Distanza', 'Normalized Power® (NP®)', 'Training Stress Score®']  # Aggiungi altre colonne se necessario
for col in cols_to_convert:
    if col in data.columns:
        data[col] = data[col].astype(str).str.replace(',', '.', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce') # Converte in numerico e NaN se non riesce


# Converte la colonna 'Tempo' in secondi
def time_to_seconds(time_str):
    if isinstance(time_str, str):
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
    return np.nan  # Gestisce i valori mancanti o non validi

data['Tempo_secondi'] = data['Tempo'].apply(time_to_seconds)

# Seleziona le features (colonne di input) e il target (colonna da predire)
features = ['Distanza', 'Tempo_secondi', 'FC Media', 'FC max'] # esempio. aggiungi altre colonne numeriche rilevanti
target = 'Calorie'

# Rimuovi righe con valori mancanti nelle colonne selezionate
data = data.dropna(subset=features + [target])

X = data[features]
y = data[target]

# 2. Divisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalizzazione delle features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Definizione del modello della rete neurale
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]), # Strato di input
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Strato di output (regressione, quindi una sola unità)
])

# 5. Compilazione del modello
model.compile(optimizer='adam',
              loss='mse',  # Mean Squared Error (adatto per la regressione)
              metrics=['mae']) # Mean Absolute Error (utile per interpretare i risultati)

# 6. Allenamento del modello
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# 7. Valutazione del modello
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("Mean Absolute Error sul test set: {:5.2f} Calorie".format(mae))

# 8. (Opzionale) Visualizzazione della perdita durante l'allenamento
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# 9. (Opzionale) Predizione su nuovi dati
# Esempio:
# new_data = pd.DataFrame([[5.0, 1800, 150, 170]], columns=features) # Distanza, tempo (sec), FC media, FC max
# new_data_scaled = scaler.transform(new_data)
# prediction = model.predict(new_data_scaled)
# print("Predizione calorie: ", prediction[0][0])

