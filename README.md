# (Proyecto IA) Predicción de Parkinson mediante Análisis de Voz

<img src="https://i.imgur.com/83o0P68.png" width="100" height="100" alt="Logo de la Unimag">

Este repositorio contiene el proyecto final para la asignatura de Inteligencia Artificial de la Universidad del Magdalena. El objetivo es desarrollar y evaluar un modelo de Machine Learning capaz de predecir la presencia de la enfermedad de Parkinson basándose en grabaciones de señales de voz.

**Autores:**
* Jose Torreglosa
* Jennifer Roa
* Lauren Gonzales

---

### 💡 Sobre el Proyecto

Este proyecto utiliza el conjunto de datos "Parkinson's Disease Data Set" de la Universidad de Oxford. El dataset consiste en 195 grabaciones de voz de 31 individuos (23 con Parkinson y 8 sanos). Se extrajeron 22 características biomédicas de cada grabación (como *jitter*, *shimmer*, y frecuencias fundamentales).

El objetivo principal es clasificar correctamente a un paciente como "Sano" (0) o "Parkinson" (1) usando estas características.

### 챌 Desafío Principal: Desbalanceo de Clases

El desafío más significativo de este dataset es el **desbalanceo de clases**. De las 195 muestras:
* **147 (75%)** pertenecen a pacientes con Parkinson (Clase 1).
* **48 (25%)** pertenecen a pacientes sanos (Clase 0).

Un modelo simple entrenado con estos datos tenderá a predecir siempre "Parkinson", logrando un accuracy artificialmente alto pero siendo inútil para detectar pacientes sanos. Este notebook explora activamente soluciones a este problema.

---

### 🔬 Flujo de Trabajo (Workflow)

El notebook `PARKINSONll.ipynb` está estructurado de la siguiente manera:

1.  **Carga e Inspección:** Se cargan los datos y se verifica su integridad (sin valores nulos o duplicados).
2.  **Análisis Exploratorio (EDA):**
    * Visualización de las distribuciones de las características.
    * Análisis de la variable objetivo (`status`) para confirmar el desbalanceo.
    * Estudio de **multicolinealidad** mediante un *heatmap*, revelando alta correlación entre muchas características.
3.  **Preprocesamiento y Selección de Características:**
    * Debido a la alta multicolinealidad, se aplicó **Regresión Logística con regularización L1 (Lasso)**.
    * Esta técnica seleccionó las **13 características más relevantes** de las 22 originales, eliminando la redundancia.
    * Los datos fueron estandarizados usando `StandardScaler`.
4.  **Modelado y Experimentación:**
    * Se compararon múltiples estrategias para manejar el desbalanceo y maximizar el rendimiento.
    * **Método 1:** Regresión Logística + **SMOTE** (Accuracy: 76%).
    * **Método 2:** Random Forest + **Ponderación de Clases** (Accuracy: 93%).
    * **Método 3:** Optimización del Método 2 usando **GridSearchCV** (Accuracy: 93%).
    * **Método 4:** Random Forest (Baseline, *sin* balanceo) (Accuracy: 95%).
    * **Método 5 (CAMPEÓN):** Random Forest + **SMOTE** (Accuracy: 97%).

---

### 🏆 Resultados y Modelo Final

El modelo final, un **`RandomForestClassifier`** entrenado sobre datos aumentados con **SMOTE** (Synthetic Minority Over-sampling Technique), demostró ser el más robusto, alcanzando un **accuracy del 97%** en el conjunto de prueba (Celda 51).

Este modelo fue capaz de manejar exitosamente el desbalanceo, produciendo una matriz de confusión casi perfecta en el set de prueba, con solo 1 Falso Positivo y 1 Falso Negativo.

#### Comparativa de Modelos Finales

| Estrategia de Modelo (Random Forest) | Accuracy | Falsos Positivos | Falsos Negativos |
| :--- | :---: | :---: | :---: |
| **RF + SMOTE (Modelo Ganador)** | **97%** | **1** | **1** |
| RF (Sin técnica de balanceo) | 95% | 1 | 2 |
| RF + Ponderación de Clases | 93% | 3 | 1 |
| Regresión Logística + SMOTE | 76% | 12 | 2 |

---

### 🛠️ Cómo ejecutar este proyecto

1.  Clona el repositorio:
    ```sh
    git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
    ```
2.  Crea un entorno virtual e instala las dependencias (puedes crear un `requirements.txt`):
    ```
    pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter
    ```
3.  Abre el notebook `PARKINSONll.ipynb` usando Jupyter Lab o Jupyter Notebook.
4.  Asegúrate de que el archivo `parkinsons.data` esté en una carpeta `/data/` al mismo nivel que el notebook.

---

### 📩 

AtromicLink(LinkedGTF)
Jose Torreglosa.

![Análisis de Parkinson](https://private-user-images.githubusercontent.com/74038190/243328563-d0cfe7d1-0b8c-4e4a-9a66-875290ba6065.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjE5NDMyOTEsIm5iZiI6MTc2MTk0Mjk5MSwicGF0aCI6Ii83NDAzODE5MC8yNDMzMjg1NjMtZDBjZmU3ZDEtMGI4Yy00ZTRhLTlhNjYtODc1MjkwYmE2MDY1LmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDMxVDIwMzYzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTRmZTI5NGMyOGVjM2MxMjNlNjVhNTUyMmVlZjYwMTA5ZmE4YzgxZTEwMzAwODk3OTk3OTA4ZDhiOWRmM2Y3NmEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ctTkOLVGkTS-RMAiircyZPmlqRalwCkiqWqH_YhvuEs)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jtorreglosam/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:josddaniel1@gmail.com)

---
