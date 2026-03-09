import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("Data.csv")
X = data.drop("z_spec", axis=1)
y = data["z_spec"]

# 1. Сначала выделяем ФИНАЛЬНУЮ тестовую выборку (15-20%)
X_train_val, X_test_final, y_train_val, y_test_final = train_test_split(
    X, y, test_size=0.15, random_state=42  # 15% для финального теста
)

print("Размеры выборок:")
print(f"Тренировочная+Валидационная: {X_train_val.shape}")
print(f"Финальная тестовая: {X_test_final.shape}")

# 2. Обучаем модель на тренировочной+валидационной части
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, random_state=42)
model.fit(X_train_val, y_train_val)

# 3. Строим кривую обучения ТОЛЬКО на тренировочной+валидационной части
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_val, y_train_val,
    scoring="neg_mean_absolute_error",
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,  # 5-кратная кросс-валидация
    n_jobs=-1,
    random_state=42
)

# 4. Вычисляем средние значения MAE
train_mae = -train_scores.mean(axis=1)
val_mae = -val_scores.mean(axis=1)

# 5. Визуализация
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mae, 'o-', color='blue', linewidth=2, markersize=8, label='Обучающая выборка')
plt.plot(train_sizes, val_mae, 'o-', color='red', linewidth=2, markersize=8, label='Валидационная выборка (CV)')
plt.xlabel('Размер обучающей выборки', fontsize=20)
plt.ylabel('MAE', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Кривые обучения: Gradient Boosting для фото-z', fontsize=16)
plt.legend(loc='best', fontsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# 6. Финальная оценка
print("\n" + "="*50)
print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ:")
print("="*50)
print(f"MAE на тесте: {final_mae:.4f}")
print(f"RMSE на тесте: {metrics.root_mean_squared_error(y_test_final, y_pred_final):.4f}")
print(f"R² на тесте: {metrics.r2_score(y_test_final, y_pred_final):.4f}")