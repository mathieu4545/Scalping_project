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
import matplotlib.pyplot as plt


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


# Fonction de prétraitement des données
def preprocess_data(data, lookback=10, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)

    X = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, :])

    X = np.array(X)
    return X, scaler


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
def save_model(model, scaler, model_path='training_models/lstm_model.h5', scaler_path='training_models/scaler.pkl'):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)


# Chargement du Modèle et Inférence en Temps Réel
def load_trained_model(model_path='training_models/lstm_model.h5', scaler_path='training_models/scaler.pkl'):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# Fonction de backtest modifiée pour calculer la précision des prédictions
def backtest_accuracy(model, scaler, data, lookback=10, step=5):
    X_test, _ = preprocess_data(data, lookback=lookback, scaler=scaler)
    close_prices = data['Close'].values

    correct_predictions = 0
    total_predictions = 0

    for i in range(lookback, len(data)-1, step):
        current_price = close_prices[i]
        recent_data = data.iloc[i - lookback:i]
        X = X_test[i - lookback]
        X = np.expand_dims(X, axis=0)

        # Prédire le prix
        predicted_price = model.predict(X)

        # Créer un tableau de la même forme que les données initiales
        dummy_array = np.zeros((1, X.shape[2]))  # Ajustement des dimensions pour correspondre aux données originales
        dummy_array[0, 3] = predicted_price  # Placer la prédiction dans la colonne 'Close'

        # Inverse transformation pour obtenir le prix d'origine
        predicted_price_original = scaler.inverse_transform(dummy_array)[0, 3]

        # Vérifier si la prédiction était correcte (au-dessus ou en-dessous du prix actuel)
        if (predicted_price_original > current_price and close_prices[i + 1] > current_price) or \
                (predicted_price_original < current_price and close_prices[i + 1] < current_price):
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions
    logging.info(f"Accuracy: {accuracy:.2%}")
    return accuracy


# Fonction pour afficher la précision du backtest
def print_backtest_accuracy(accuracy):
    print(f"Prediction Accuracy: {accuracy:.2%}")


def test_different_lookbacks(model, scaler, data, lookbacks=[10, 20, 30]):
    results = {}
    for lookback in lookbacks:
        print(f"Testing with lookback = {lookback}")
        accuracy = backtest_accuracy(model, scaler, data, lookback=lookback, step=20)
        results[lookback] = accuracy
    return results

# Fonction pour tracer les courbes de précision
def plot_accuracy(results):
    lookbacks = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.plot(lookbacks, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy vs. Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    logging.info("Backtesting process started.")

    # Chemin vers le modèle entraîné
    model_path = 'training_models/lstm_model.h5'
    scaler_path = 'training_models/scaler.pkl'

    # Chargement du modèle et du scaler
    model, scaler = load_trained_model(model_path=model_path, scaler_path=scaler_path)

    # Charger les données historiques pour le backtest
    test_file_path = 'training_dataset/dataset_prediction test/AUDNZD_PERIOD_M10_2024.05.22_to_2024.08.30.csv'
    test_data = load_data(test_file_path)

    # Exécuter le backtest et obtenir la précision
    accuracy = backtest_accuracy(model, scaler, test_data)

    # Exemple d'utilisation
    lookback_values = [10]
    results = test_different_lookbacks(model, scaler, test_data, lookbacks=lookback_values)
    print("Lookback Results:")
    for lb, acc in results.items():
        print(f"Lookback {lb}: Accuracy {acc:.2%}")

    # Tracer les courbes de précision
    plot_accuracy(results)
    # Afficher les résultats du backtest
    # print_backtest_accuracy(accuracy)

    logging.info("Backtesting process finished.")

# Exemple de pipeline d'entraînement
# if __name__ == "__main__":
#     logging.info("Process started.")
#
#     # directory_path = 'training_dataset/dataset_TF10MIN'  # Remplacez ceci par le chemin vers votre dossier contenant les fichiers CSV
#     # csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
#     #
#     # model = None
#     # scaler = None
#     #
#     # for file_path in csv_files:
#     #     logging.info(f"Processing file: {file_path}")
#     #
#     #     # 1. Charger et prétraiter les données
#     #     data = load_data(file_path)
#     #     X, y, file_scaler = preprocess_data(data)
#     #
#     #     # 2. Initialiser ou charger le modèle
#     #     if model is None:
#     #         model = build_model(input_shape=(X.shape[1], X.shape[2]))  # Créer le modèle avec la forme correcte
#     #     else:
#     #         scaler = file_scaler  # Utiliser le scaler du fichier actuel
#     #
#     #     # Diviser les données en ensembles de formation et de validation
#     #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
#     #
#     #     # 3. Entraîner le modèle
#     #     logging.info(f"Training model with file: {file_path}")
#     #     model, history = train_model(model, X_train, y_train, X_val, y_val)
#     #
#     #     # 4. Sauvegarder le modèle et le scaler
#     #     save_model(model, scaler)
#     #
#     #     logging.info(f"Model trained and saved for file: {file_path}")
#
#
#
#
#
#
# ############################## BACKTESTING ##############################
#     ############################################################
#     logging.info("Backtesting process started.")
#
#     # Chemin vers le modèle entraîné
#     model_path = 'training_models/lstm_model.h5'
#     scaler_path = 'training_models/scaler.pkl'
#
#     # Chargement du modèle et du scaler
#     model, scaler = load_trained_model(model_path=model_path, scaler_path=scaler_path)
#
#     # Charger les données historiques pour le backtest
#     test_file_path = 'training_dataset/dataset_prediction test/AUDNZD_PERIOD_M10_2024.05.22_to_2024.08.30.csv'  # Remplacez ceci par le chemin vers vos données de test
#     test_data = load_data(test_file_path)
#
#     # Exécuter le backtest
#     history, final_balance = backtest(model, scaler, test_data)
#
#     # Afficher les résultats du backtest
#     print_backtest_results(history, final_balance)
#
#     logging.info("Backtesting process finished.")
# ############################## BACKTESTING ##############################
#     ############################################################
#
#
#
#
#
#
#
#
#
#
#
#
#     # data = load_data('training_dataset/AUDCAD_PERIOD_M1_2019.08.29_to_2024.08.27.csv')
#     #
#     # # Exemple de prédiction avec des données récentes du dernier fichier traité
#     # model, scaler = load_model()
#     # recent_data = data.tail(10)  # Supposons que ce sont les dernières 10 minutes du dernier fichier
#     # predicted_price = predict_action(model, scaler, recent_data)
#     # current_price = recent_data['Close'].values[-1]
#     # action, stop_loss, take_profit = generate_trading_signal(current_price, predicted_price)
#     #
#     # logging.info(f"Action: {action}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
#     # logging.info("Process finished.")
