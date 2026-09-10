# Clasificación de Sentimientos en Reseñas de Películas

Proyecto de **Procesamiento de Lenguaje Natural (NLP)** y **Machine Learning** desarrollado en Python para clasificar reseñas de películas de un dataset como **positivas** o **negativas**.

El proyecto implementa un flujo completo de Machine Learning que incluye carga y exploración de datos, creación de un dataset desbalanceado, balanceo mediante *Random Under-Sampling*, división de datos, transformación de texto mediante **TF-IDF**, entrenamiento de diferentes algoritmos de clasificación y evaluación de sus resultados.

---

## Objetivo del proyecto

El objetivo principal es construir y comparar diferentes modelos de Machine Learning capaces de determinar el sentimiento de una reseña.

El proyecto compara los siguientes algoritmos:

* **Support Vector Machine (SVM)**
* **Decision Tree**
* **Gaussian Naive Bayes**
* **Logistic Regression**

De esta manera, se puede observar qué modelo obtiene un mejor rendimiento para la clasificación de sentimientos utilizando representaciones de texto mediante TF-IDF.

---

## Tecnologías utilizadas

* **Python**
* **Pandas**
* **Scikit-learn**
* **imbalanced-learn**
* **TF-IDF**
* **Machine Learning**
* **Procesamiento de Lenguaje Natural (NLP)**

---

## Estructura del proyecto

```text
project/
│
├── data/
│   └── IMDB Dataset.csv
│
├── discovery_data.py
│
└── README-esp.md
```

### Dataset

El proyecto utiliza el archivo:

```text
data/IMDB Dataset.csv
```

Este dataset contiene reseñas de películas junto con su correspondiente sentimiento:

* `positive`
* `negative`

---

# Desarrollo del proyecto

## 1. Cargar el dataset

Primero se importan las librerías necesarias y se carga el dataset utilizando **Pandas**.

```python
data = pd.read_csv('data/IMDB Dataset.csv')
```

Esto permite almacenar las reseñas en un `DataFrame` para posteriormente realizar operaciones de filtrado, transformación y análisis.

---

## 1.1. Revisar la cantidad de datos por sentimiento

Se puede utilizar:

```python
data_count_balance = data.value_counts('sentiment')
print(data_count_balance)
```

para conocer cuántas reseñas existen para cada categoría.

Esto permite comprobar la distribución original de los sentimientos:

```text
positive
negative
```

Esta etapa es importante antes de entrenar los modelos porque permite conocer si las clases están equilibradas.

---

# 2. Crear un dataset desbalanceado

Para experimentar con técnicas de balanceo de datos, se crea intencionalmente un conjunto de datos desbalanceado.

Se seleccionan:

```python
df_postive = data[data['sentiment'] == 'positive'][:9000]
df_negative = data[data['sentiment'] == 'negative'][:1000]
```

Por lo tanto, se obtiene:

| Sentimiento |  Cantidad |
| ----------- | --------: |
| Positive    |      9000 |
| Negative    |      1000 |
| **Total**   | **10000** |

Esto genera una distribución de **90 % de reseñas positivas y 10 % de reseñas negativas**.

El propósito es simular un problema de clasificación donde una clase tiene mucha mayor representación que la otra.

---

## 2.1. Unir las reseñas

Se utilizan ambas categorías para construir un único `DataFrame`:

```python
df_review_des = pd.concat([df_postive, df_negative])
```

De esta manera se obtiene el dataset desbalanceado que será utilizado posteriormente.

---

## 2.2. Revisar el dataset desbalanceado

Para comprobar la distribución:

```python
data_count_des = df_review_des.value_counts('sentiment')
print(data_count_des)
```

Esto permite verificar que existen:

```text
positive    9000
negative    1000
```

---

#  3. Balancear el dataset

Debido al desbalance creado anteriormente, se utiliza:

```python
RandomUnderSampler
```

Esta técnica pertenece a la librería **imbalanced-learn**.

```python
rus = RandomUnderSampler()
df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(
    df_review_des[['review']], 
    df_review_des['sentiment']
)
```

### ¿Qué hace RandomUnderSampler?

La técnica **Random Under-Sampling** reduce aleatoriamente la cantidad de ejemplos de la clase mayoritaria para igualarla con la clase minoritaria.

En este caso:

```text
Antes:

Positive → 9000
Negative → 1000

Después:

Positive → 1000
Negative → 1000
```

El resultado es un dataset balanceado de:

```text
2000 reseñas
```

con una distribución equivalente entre ambas categorías.

---

# 4. División de los datos

Después de balancear el dataset, se divide en dos grupos:

* **Training Set:** datos utilizados para entrenar los modelos.
* **Test Set:** datos utilizados para evaluar los modelos.

Se utiliza:

```python
train, test = train_test_split(
    df_review_bal,
    test_size=0.33,
    random_state=42
)
```

### Parámetros

`test_size=0.33` significa que aproximadamente el **33 % de los datos** se utiliza para pruebas.

El `67 %` restante se utiliza para entrenamiento.

`random_state=42` permite obtener una división reproducible.

Posteriormente se separan las características y las etiquetas:

```python
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']
```

Donde:

* `train_x` → reseñas de entrenamiento.
* `train_y` → sentimientos de entrenamiento.
* `test_x` → reseñas de prueba.
* `test_y` → sentimientos reales de prueba.

---

# 5. Transformación del texto con TF-IDF

Los algoritmos de Machine Learning no trabajan directamente con texto.

Por esta razón se utiliza:

```python
TfidfVectorizer
```

```python
tfidf = TfidfVectorizer(stop_words='english')
```

TF-IDF convierte las palabras de las reseñas en representaciones numéricas.

### ¿Qué significa TF-IDF?

**TF-IDF** significa:

> Term Frequency - Inverse Document Frequency

Su objetivo es asignar un peso a las palabras dependiendo de qué tan importantes sean dentro de los documentos.

Las palabras frecuentes pero poco informativas reciben menor importancia, mientras que términos más representativos pueden obtener mayor peso.

---

## Entrenamiento del vectorizador

```python
train_x_vector = tfidf.fit_transform(train_x)
```

Aquí el vectorizador aprende el vocabulario utilizando únicamente los datos de entrenamiento y transforma las reseñas en vectores numéricos.

Para los datos de prueba:

```python
test_x_vector = tfidf.transform(test_x)
```

Se utiliza el mismo vocabulario aprendido durante el entrenamiento.

Esto evita que información del conjunto de prueba influya en el entrenamiento.

---

# 6. Support Vector Machine (SVM)

El primer modelo utilizado es **Support Vector Machine**.

```python
svc = SVC(kernel="linear")
svc.fit(train_x_vector, train_y)
```

Se utiliza un kernel lineal debido a que resulta adecuado para trabajar con representaciones de texto de alta dimensionalidad.

El modelo aprende a separar las reseñas positivas y negativas mediante una frontera de decisión.

También se puede realizar una predicción utilizando una reseña personalizada:

```python
svc.predict(
    tfidf.transform(["i don't like this movie"])
)
```

Esto permite probar el modelo con texto que no pertenece al dataset original.

---

# 7. Decision Tree

El segundo modelo utilizado es **Decision Tree**.

```python
des_tree = DecisionTreeClassifier()
des_tree.fit(train_x_vector, train_y)
```

Un árbol de decisión construye una estructura de decisiones basada en las características disponibles para determinar la clase correspondiente.

En este proyecto se utiliza la representación TF-IDF como entrada del modelo.

---

# 8. Gaussian Naive Bayes

El tercer algoritmo utilizado es **Gaussian Naive Bayes**.

```python
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)
```

A diferencia de algunos de los otros modelos utilizados, `GaussianNB` requiere trabajar con una matriz densa.

Por esta razón se utiliza:

```python
train_x_vector.toarray()
```

para convertir la matriz dispersa generada por TF-IDF en una matriz convencional.

---

# 9. Logistic Regression

El cuarto modelo utilizado es **Logistic Regression**.

```python
lr = LogisticRegression()
lr.fit(train_x_vector, train_y)
```

La regresión logística es un algoritmo ampliamente utilizado para problemas de clasificación binaria.

En este proyecto permite determinar si una reseña pertenece a:

```text
positive
```

o:

```text
negative
```

---

# 10. Evaluación de los modelos

Finalmente, se evalúa el rendimiento de cada modelo utilizando:

```python
model.score(test_x_vector, test_y)
```

El método `score()` devuelve la **exactitud (accuracy)** del modelo sobre los datos de prueba.

Resultados obtenidos:

| Modelo               |    Accuracy |
| -------------------- | ----------: |
| SVM                  | **83.79 %** |
| Decision Tree        | **71.67 %** |
| Gaussian Naive Bayes | **60.76 %** |
| Logistic Regression  | **84.55 %** |

---

# Comparación de resultados

De acuerdo con los resultados obtenidos, **Logistic Regression** presenta el mejor rendimiento:

```text
Logistic Regression → 84.55 %
```

seguido por:

```text
SVM → 83.79 %
```

Posteriormente:

```text
Decision Tree → 71.67 %
```

y finalmente:

```text
Gaussian Naive Bayes → 60.76 %
```

Por lo tanto, dentro de este experimento, **Logistic Regression fue el modelo con mayor accuracy** para clasificar las reseñas de películas.

Sin embargo, la diferencia entre Logistic Regression y SVM es pequeña, por lo que ambos modelos muestran un buen comportamiento para este problema.

---
