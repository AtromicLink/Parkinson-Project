# (Proyecto IA) Predicción de la Enfermedad de Parkinson mediante Análisis de Voz

![Análisis de Parkinson](https://i.imgur.com/83o0P68.png)

Este repositorio contiene el proyecto final para la asignatura de Inteligencia Artificial de la Universidad del Magdalena (Noviembre 2024). El objetivo es desarrollar y evaluar un modelo de Machine Learning capaz de predecir la presencia de la enfermedad de Parkinson basándose en grabaciones de señales de voz.

**Autores:**
* Jose Torreglosa
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
    * Estudio de **multicolinealidad** mediante un *heatmap* y una función de detección, revelando alta correlación entre muchas características de voz.
3.  **Selección de Características:**
    * Debido a la alta multicolinealidad, se aplicó **Regresión Logística con regularización L1 (Lasso)**.
    * Esta técnica seleccionó las **13 características más relevantes** de las 22 originales, eliminando la redundancia.
4.  **Modelado y Balanceo de Clases:**
    * Se compararon múltiples estrategias para manejar el desbalanceo y maximizar el rendimiento.
    * **Método 1:** Regresión Logística + **SMOTE** (Accuracy: 76%).
    * **Método 2:** Random Forest + **Ponderación de Clases** (Accuracy: 93%).
    * **Método 3:** Optimización del Método 2 usando **GridSearchCV** (Accuracy: 93%).
    * **Método 4:** SMOTE + Random Forest (Accuracy: 92%).

---

### 🏆 Resultados y Conclusión

El modelo final, un **Random Forest Optimizado** utilizando la técnica de `class_weight='balanced_subsample'` (Método 3), demostró ser el más robusto, alcanzando un **accuracy del 93%** en el conjunto de prueba.

Este modelo fue capaz de manejar exitosamente el desbalanceo de clases, identificando correctamente a 12 de los 15 pacientes sanos y a 43 de los 44 pacientes con Parkinson en el set de prueba.

| Modelo | Accuracy | F1-Score (Sano) | F1-Score (Parkinson) |
| :--- | :---: | :---: | :---: |
| Regresión Logística + SMOTE | 0.76 | 0.65 | 0.82 |
| **Random Forest + Class Weight (Optimizado)** | **0.93** | **0.86** | **0.96** |
| Random Forest + SMOTE | 0.92 | 0.83 | 0.94 |

---

### 🛠️ Cómo ejecutar este proyecto

1.  Clona el repositorio:
    ```sh
    git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
    ```
2.  Crea un entorno virtual e instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```
    (Asegúrate de tener `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` y `imblearn`)
3.  Abre el notebook `PARKINSONll.ipynb` usando Jupyter Lab o Jupyter Notebook.
4.  Asegúrate de que el archivo `parkinsons.data` esté en una carpeta `/data/` al mismo nivel que el notebook.

---

### 📩 Contacto

¡Conéctate con nosotros!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jtorreglosam/)

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:josddaniel1@gmail.com)

---
