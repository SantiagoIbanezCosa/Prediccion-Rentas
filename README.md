Análisis de Datos de Renta de Viviendas y Modelado Predictivo
Este repositorio contiene un proyecto de análisis y modelado predictivo utilizando un conjunto de datos sobre alquiler de viviendas. El análisis incluye la visualización de los datos, la limpieza, la imputación de valores faltantes, la creación de nuevas características y la construcción de varios modelos de regresión.

Descripción del Proyecto
El proyecto realiza un análisis exploratorio de los datos y luego aplica técnicas de machine learning para predecir el alquiler de viviendas en función de varias características, como el número de habitaciones, el tamaño y los baños. Los siguientes modelos de regresión se entrenan y evalúan:

Regresión Lineal
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
Pasos realizados en el proyecto
Carga y Exploración de Datos:

Se cargan los datos desde un archivo CSV.
Se muestran las primeras filas, la información sobre los tipos de datos y los valores nulos.
Se muestran estadísticas generales y se examinan los archivos del directorio de trabajo.
Visualización de Datos:

Se realiza un análisis de correlación utilizando un heatmap.
Se muestran gráficas de dispersión para observar las relaciones entre variables numéricas.
Se crean histogramas y boxplots para explorar la distribución y los outliers en las columnas numéricas.
Preprocesamiento:

Se imputan valores faltantes en la columna Rent con la mediana.
Se manejan los outliers en la columna Rent utilizando el rango intercuartílico (IQR).
Se crea una nueva característica llamada Cost_per_Sqft, que representa el costo por pie cuadrado.
Transformación de Datos:

Se realiza la codificación One-Hot de las variables categóricas.
Se normalizan las características numéricas utilizando StandardScaler.
Entrenamiento de Modelos:

Se divide el conjunto de datos en entrenamiento y prueba.
Se entrenan modelos de regresión lineal, Random Forest, Gradient Boosting y XGBoost sin optimizar.
Se evalúan los modelos utilizando el error cuadrático medio (MSE), R2 y otros indicadores de rendimiento.
Optimización de XGBoost:

Se realiza una optimización de hiperparámetros utilizando Grid Search para XGBoost.
Se aplica validación cruzada con K-fold para evaluar el rendimiento del modelo optimizado.
Requisitos
Para ejecutar este proyecto, necesitarás tener instaladas las siguientes bibliotecas de Python:

pandas
seaborn
matplotlib
scikit-learn
xgboost
Puedes instalar las dependencias necesarias utilizando pip:

bash
Copiar código
pip install pandas seaborn matplotlib scikit-learn xgboost
Uso
Asegúrate de que el archivo House_Rent_Dataset.csv se encuentra en la carpeta data set del directorio especificado en el código (puedes cambiar la ruta en el código según sea necesario).
Ejecuta el código en un entorno de Python.
Los resultados incluyen estadísticas del conjunto de datos, gráficos de visualización, y las métricas de rendimiento de los modelos de machine learning.
Resultados
Los resultados de las métricas de cada modelo se mostrarán en la consola, incluyendo el MSE, R2, MAE, y el rendimiento del modelo XGBoost optimizado con Grid Search.

Contribuciones
Si deseas contribuir a este proyecto, por favor abre un issue o envía un pull request.
