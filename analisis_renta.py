import pandas as pd
import os

directory = 'C:\\Users\\Santiago\\Desktop\\PruebasCienciaDeDatos\\Renta\\data set'
file_path = os.path.join(directory, 'House_Rent_Dataset.csv')
print(f"Trying to open file: {file_path}")

# Listar archivos en el directorio
print("Files in directory:")
for file in os.listdir(directory):
    print(file)

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    # Primeras filas
    print(data.head())
    # Tipos de datos y nulos
    print(data.info())
    # Estadísticas generales
    print(data.describe())
else:
    print(f"File not found: {file_path}")

import seaborn as sns
import matplotlib.pyplot as plt

# Seleccionar solo columnas numéricas
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Matriz de correlación
corr_matrix = numeric_data.corr()

# Heatmap de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de correlación")
plt.show()

# Selecciona columnas numéricas relevantes
columnas_relevantes = ['BHK', 'Rent', 'Size', 'Bathroom']
sns.pairplot(numeric_data[columnas_relevantes], diag_kind='kde', markers='+')
plt.show()

# Ajusta las columnas a analizar
columnas_a_graficar = ['BHK', 'Rent', 'Size', 'Bathroom']

for col in columnas_a_graficar:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, color='blue', bins=30)
    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.show()

# Boxplot por columna
for col in columnas_a_graficar:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col], color='green')
    plt.title(f"Boxplot de {col}")
    plt.show()

# Imputación de valores faltantes usando la mediana
data['Rent'] = data['Rent'].fillna(data['Rent'].median())

# Tratar outliers en la columna 'Rent'
Q1 = data['Rent'].quantile(0.25)
Q3 = data['Rent'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['Rent'] < (Q1 - 1.5 * IQR)) | (data['Rent'] > (Q3 + 1.5 * IQR)))]

# Crear nuevas características
data['Cost_per_Sqft'] = data['Rent'] / data['Size']

from sklearn.preprocessing import StandardScaler

# Si tienes variables categóricas, puedes realizar One-Hot Encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Normaliza los datos numéricos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

from sklearn.model_selection import train_test_split

X = data_encoded.drop('Rent', axis=1)  # Características
y = data_encoded['Rent']  # Etiqueta (target)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

# Crear y entrenar el modelo de Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_linear)}")
print(f"Linear Regression R2: {r2_score(y_test, y_pred_linear)}")

# Crear y entrenar el modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred_rf)}")
print(f"Random Forest R2: {r2_score(y_test, y_pred_rf)}")

# Crear y entrenar el modelo de Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print(f"Gradient Boosting MSE: {mean_squared_error(y_test, y_pred_gb)}")
print(f"Gradient Boosting R2: {r2_score(y_test, y_pred_gb)}")

# Crear y entrenar el modelo de XGBoost sin optimizar
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost sin optimizar MSE: {mean_squared_error(y_test, y_pred_xgb)}")
print(f"XGBoost sin optimizar R2: {r2_score(y_test, y_pred_xgb)}")

# Optimización de hiperparámetros para XGBoost
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_

# Validación cruzada con K-fold (n_splits=20)
kf = KFold(n_splits=20, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
print(f"XGBoost optimizado Cross-Validation MAE: {-cv_scores.mean()}")

# Evaluar el modelo de XGBoost optimizado
y_pred_xgb_opt = best_xgb_model.predict(X_test)
print(f"XGBoost optimizado MSE: {mean_squared_error(y_test, y_pred_xgb_opt)}")
print(f"XGBoost optimizado R2: {r2_score(y_test, y_pred_xgb_opt)}")
print(f"XGBoost optimizado MAE: {mean_absolute_error(y_test, y_pred_xgb_opt)}")




