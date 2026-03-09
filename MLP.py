import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Загружаем данные
data = pd.read_csv("Data.csv")
X = data.drop(["objid", "ra", "dec", "z_spec"], axis=1).values
y = data["z_spec"].values

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создаём модель
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(92, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam",
              loss="mse",
              metrics=["mae", "mse"])

# Callback для подсчёта тестового loss на каждой эпохе
class TestLossCallback(Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.test_mae = []
        self.test_mse = []

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.test_mae.append(results[1])  # MAE
        self.test_mse.append(results[2])  # MSE

test_loss_cb = TestLossCallback(X_test, y_test)

# Обучение
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=70,
                    batch_size=32,
                    callbacks=[test_loss_cb])

# Предсказание и метрики
y_pred = model.predict(X_test)
print("R2: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Кривые обучения MSE
epochs = range(1, len(history.history['loss']) + 1)

# Кривые обучения MAE
plt.figure(figsize=(8, 5))
plt.plot(epochs, history.history['mae'], 'o-', label='Train MAE')
plt.plot(epochs, test_loss_cb.test_mae, 'o-', label='Test MAE')
#plt.plot(epochs, history.history['val_mae'], 'o--', label='Val MAE (для сравнения)')
plt.xlabel('Эпоха', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()
