import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cargar datos desde el archivo CSV
data = pd.read_csv(r'parkinsons.data')

# Separar características (excluyendo 'name' y 'status') y la variable objetivo 'status'
X = data.drop(columns=['name', 'status'])  # Características
y = data['status']                         # Variable objetivo

# Normalizamos las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir el modelo LassoCV para encontrar el mejor valor de alpha
lasso_cv = LassoCV(cv=5)

# Ajustar el modelo a las características y al objetivo
lasso_cv.fit(X_scaled, y)

# Obtener las características seleccionadas con coeficientes distintos de cero
selected_features = X.columns[(lasso_cv.coef_ != 0)]

# Crear una matriz de correlación reducida con las características seleccionadas
correlation_matrix = X[selected_features].corr()

print("Variables seleccionadas por Lasso:")
print(selected_features)
print("\nMatriz de correlación reducida:")
print(correlation_matrix)
