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

# Configurer le logger pour inclure les timestamps
logging.basicConfig(
    filename='training_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Fonction pour détecter l'encodage du fichier
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))  # Lire un bout du fichier
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


def preprocess_data(data, lookback=10):
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
def save_model(model, scaler, model_path='lstm_model.h5', scaler_path='scaler.pkl'):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)


# Chargement du Modèle et Inférence en Temps Réel
def load_model(model_path='lstm_model.h5', scaler_path='scaler.pkl'):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_action(model, scaler, recent_data):
    if 'Time' in recent_data.columns:
        recent_data = recent_data.drop(columns=['Time'])

    recent_data_scaled = scaler.transform(recent_data)
    recent_data_reshaped = np.expand_dims(recent_data_scaled, axis=0)
    predicted_price = model.predict(recent_data_reshaped)
    data_for_inverse_transform = np.zeros((1, recent_data_scaled.shape[1]))
    data_for_inverse_transform[0][3] = predicted_price[0][0]
    scaled_back = scaler.inverse_transform(data_for_inverse_transform)
    predicted_price_original = scaled_back[0][3]
    return predicted_price_original


def generate_trading_signal(current_price, predicted_price, threshold=0.001):
    if predicted_price > current_price * (1 + threshold):
        action = 'BUY'
        stop_loss = current_price * 0.995
        take_profit = current_price * 1.005
    elif predicted_price < current_price * (1 - threshold):
        action = 'SELL'
        stop_loss = current_price * 1.005
        take_profit = current_price * 0.995
    else:
        action = 'HOLD'
        stop_loss = None
        take_profit = None
    return action, stop_loss, take_profit


# Exemple de pipeline d'entraînement
if __name__ == "__main__":
    logging.info("Debut du script de prediction.")

    directory_path = 'training_dataset'  # Remplacez ceci par le chemin vers votre dossier contenant les fichiers CSV
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    model = None
    scaler = None


    data = load_data('training_dataset/AUDCAD_PERIOD_M1_2019.08.29_to_2024.08.27.csv')

    # Exemple de prédiction avec des données récentes du dernier fichier traité
    model, scaler = load_model()
    recent_data = data.tail(10)  # Supposons que ce sont les dernières 10 minutes du dernier fichier
    predicted_price = predict_action(model, scaler, recent_data)
    current_price = recent_data['Close'].values[-1]
    action, stop_loss, take_profit = generate_trading_signal(current_price, predicted_price)

    logging.info(f"Action: {action}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
    logging.info("Process finished.")
