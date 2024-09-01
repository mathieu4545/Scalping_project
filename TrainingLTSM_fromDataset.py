import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
import chardet
import logging

#Variables
NB_LINE_CSV_TO_READ = 100000
LOOKBACK_WDW_DATA_ANALYSE = 10

##
PATH_TO_DATASET_FOLDER = 'training_dataset/dataset_TF10MIN'
PATH_TRAINED_MODEL_1M = 'models/trained_models/model_1m_40x100000/lstm_model.h5'
PATH_SCALER_1M = 'models/trained_models/model_1m_40x100000/scaler.pkl'

# Configurer le logger pour inclure les timestamps
logging.basicConfig(
    filename='training_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Fonction pour détecter l'encodage du fichier
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(NB_LINE_CSV_TO_READ))  # Lire un bout du fichier
    return result['encoding']


# Fonction pour charger les données d'un fichier CSV
def load_data(file_path):
    encoding = detect_encoding(file_path)
    data = pd.read_csv(
        file_path,
        sep='\s+',  # Utilisation des espaces comme séparateur
        header=0,  # La première ligne contient les noms des colonnes
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close'],  # Nommer explicitement les colonnes
        encoding=encoding,
        engine='python'
    )
    data['Time'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y.%m.%d %H:%M')
    data.drop(columns=['Date'], inplace=True)
    data.set_index('Time', inplace=True)
    return data


def preprocess_data(data, lookback=LOOKBACK_WDW_DATA_ANALYSE):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, :])
        y.append(scaled_data[i, 3])  # Nous prédisons la colonne 'Close'

    X, y = np.array(X), np.array(y)

    return X, y, scaler


# Construction du Modèle LSTM
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# Entraînement du Modèle
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    logging.info("Starting training...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stop])
    logging.info("Training finished.")
    return model, history


# Sauvegarder le Modèle et le Scaler
def save_model(model, scaler, model_path='models/training_models/lstm_model.h5', scaler_path='models/training_models/scaler.pkl'):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

# Chargement du Modèle et Inférence en Temps Réel
def load_model(model_path='models/trained_models/model_1m_40x100000/lstm_model.h5', scaler_path='scaler.pkl'):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Exemple de pipeline d'entraînement
if __name__ == "__main__":
    logging.info("Process started.")

    directory_path = PATH_TO_DATASET_FOLDER  # Remplacez ceci par le chemin vers votre dossier contenant les fichiers CSV
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    model, scaler = load_model()

    for file_path in csv_files:
        logging.info(f"Processing file: {file_path}")

        # 1. Charger et prétraiter les données
        data = load_data(file_path)
        X, y, file_scaler = preprocess_data(data)

        # 2. Initialiser ou charger le modèle
        if model is None:
            model = build_model(input_shape=(X.shape[1], X.shape[2]))  # Créer le modèle avec la forme correcte
        else:
            scaler = file_scaler  # Utiliser le scaler du fichier actuel

        # Diviser les données en ensembles de formation et de validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 3. Entraîner le modèle
        logging.info(f"Training model with file: {file_path}")
        model, history = train_model(model, X_train, y_train, X_val, y_val)

        # 4. Sauvegarder le modèle et le scaler
        save_model(model, scaler)

        logging.info(f"Model trained and saved for file: {file_path}")

