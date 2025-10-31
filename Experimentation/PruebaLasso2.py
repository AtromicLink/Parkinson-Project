import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import numpy as np

# Cargar datos y separar características y objetivo
data = pd.read_csv(r'parkinsons.data')
X = data.drop(columns=['name', 'status'])  # Características
y = data['status']                         # Variable objetivo

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Selección de características con regresión logística L1 ----
# Definir el modelo con regularización L1
log_reg = LogisticRegression(C=1, penalty='l1', solver='saga', max_iter=1000)

# Usar SelectFromModel para seleccionar características
selector = SelectFromModel(log_reg)
selector.fit(X_scaled, y)

# Obtener las características seleccionadas
selected_features = X.columns[(selector.get_support())]

# Imprimir resultados
print('Total de características:', X.shape[1])
print('Características seleccionadas:', len(selected_features))
print('Características seleccionadas por L1:')
print(selected_features)

# Contar cuántas características fueron descartadas (coeficientes en cero)
print('Características con coeficientes reducidos a cero:',
      np.sum(selector.estimator_.coef_ == 0))

# Mostrar las características que fueron descartadas
print('Características no seleccionadas:')
print(set(X.columns) - set(selected_features))
