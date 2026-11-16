# Clasificación de alojamientos de Airbnb en Madrid con aprendizaje automático

Proyecto de aprendizaje automático con [Python](https://www.python.org/) y [scikit-learn](https://scikit-learn.org/stable/index.html) para predecir el tipo de alojamiento de un anuncio de [Airbnb](https://www.airbnb.es/) en Madrid. A partir de características como el precio, el barrio, las reseñas y la disponibilidad, el modelo clasifica cada anuncio como vivienda completa, habitación privada o habitación compartida.

Para ello, se utiliza el archivo [`airbnb.csv`](airbnb.csv), que contiene 13.321 anuncios y 11 variables. Es una versión adaptada con fines educativos del listado de anuncios de Madrid de abril de 2017, recopilado por [Inside Airbnb](https://insideairbnb.com/) a partir de información pública de Airbnb.

A partir de estos datos, el notebook [clasificacion_airbnb_madrid.ipynb](./clasificacion_airbnb_madrid.ipynb) detalla qué se hace en cada etapa, por qué y qué resultados se obtienen.

1. **Análisis exploratorio de datos**. Análisis descriptivo y visualización de las variables, con especial atención a los valores atípicos, las correlaciones y el desbalanceo de clases entre los tipos de alojamiento.
2. **Preprocesamiento de datos**. Filtrado de valores atípicos, división estratificada en entrenamiento y test, codificación de variables cualitativas, imputación de valores ausentes, [feature engineering](https://aws.amazon.com/what-is/feature-engineering/) y normalización.
3. **Comparación de modelos**. Evaluación de Naive Bayes gaussiano, KNN y un árbol de decisión mediante [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) de diez rondas. Se compara la exactitud y se utilizan matrices de confusión para analizar los errores por clase.
4. **Ajuste de hiperparámetros**. Búsqueda de la mejor configuración de KNN y del árbol de decisión mediante [Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
5. **Evaluación final**. Comparación de los tres modelos con el conjunto de test, reservado hasta el final, y conclusiones sobre su rendimiento.

## Ejecución del proyecto

```bash
sudo apt update
sudo apt install python3
python -m pip install -r requirements.txt
```

Abre [`clasificacion_airbnb_madrid.ipynb`](./clasificacion_airbnb_madrid.ipynb) y ejecuta las celdas en orden. El archivo `airbnb.csv` debe estar en la misma carpeta que el notebook.

## Resultados

| Modelo | Exactitud en test |
| --- | ---: |
| Árbol de decisión | 89,68 % |
| KNN | 78,15 % |
| Naive Bayes gaussiano | 13,96 % |

El árbol de decisión obtuvo la mayor exactitud en test. Las matrices de confusión de la cross-validation muestran que las habitaciones compartidas (la clase menos frecuente) siguen siendo difíciles de identificar.
