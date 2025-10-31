    import pandas as pd
    from sklearn.linear_model import LassoCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Cargar datos y separar características y objetivo
    data = pd.read_csv(r'parkinsons.data')
    X = data.drop(columns=['name', 'status'])  # Características
    y = data['status']                         # Variable objetivo
    
    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ---- Selección de características con Lasso ----
    lasso_cv = LassoCV(cv=5)
    lasso_cv.fit(X_scaled, y)
    
    # Obtener las características seleccionadas con coeficientes distintos de cero
    selected_features_lasso = X.columns[(lasso_cv.coef_ != 0)]
    
    # ---- Selección de características con Softmax ----
    softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    softmax_model.fit(X_scaled, y)
    
    # Extraer coeficientes del modelo Softmax
    softmax_coefficients = np.abs(softmax_model.coef_).mean(axis=0)
    
    # Obtener características importantes de Softmax (con coeficientes más altos)
    selected_features_softmax = X.columns[softmax_coefficients > np.median(softmax_coefficients)]
    
    # Imprimir resultados de ambas selecciones
    print("Variables seleccionadas por Lasso:")
    print(selected_features_lasso)
    
    print("\nVariables seleccionadas por Softmax (coeficientes más altos):")
    print(selected_features_softmax)
