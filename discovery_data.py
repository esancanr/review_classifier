import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. CARGAR EL DATASET
# ============================================================
data = pd.read_csv('data/IMDB Dataset.csv')

# 1.1. REVISAR LA CANTIDAD DE DATOS POR CADA SENTIMIENTO
# Esta línea cuenta cuántas reseñas existen de cada categoría:
# "positive" y "negative".
# data_count_balance = data.value_counts('sentiment')
# print(data_count_balance)

# ============================================================
# 2. CREAR UN DATASET DESBALANCEADO
# ============================================================
df_postive = data[data['sentiment'] == 'positive'][:9000]
df_negative = data[data['sentiment'] == 'negative'][:1000]

# 2.1 Unir las reseñas positivas y negativas en un solo DataFrame.
df_review_des = pd.concat([df_postive, df_negative])

# 2.2. REVISAR EL DATASET DESBALANCEADO
# Estas líneas permiten comprobar cuántas reseñas hay
# de cada categoría después de crear el dataset desbalanceado.
# data_count_des = df_review_des.value_counts('sentiment')
# print(data_count_des)

# ============================================================
# 3. BALANCEAR EL DATASET
# ============================================================
rus = RandomUnderSampler()
df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(
    df_review_des[['review']], 
    df_review_des['sentiment']
    )

# ============================================================
# 6. DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA
# ============================================================
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# ============================================================
# 7. CONVERTIR EL TEXTO EN NÚMEROS
# ============================================================

# Los modelos de Machine Learning no pueden trabajar directamente
# con texto.
#
# Por eso utilizamos TF-IDF, que transforma las palabras de las
# reseñas en valores numéricos.

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

# ============================================================
# 8. SUPPORT VECTOR MACHINE (SVM)
# ============================================================
svc = SVC(kernel="linear")
svc.fit(train_x_vector, train_y)

#Testear la predicion de nuestro modelo en base a un review propio 
#print(svc.predict(tfidf.transform(["i don´t like this movie"])))

# ============================================================
# 9. DECISION TREE
# ============================================================
des_tree = DecisionTreeClassifier()
des_tree.fit(train_x_vector, train_y)

# ============================================================
# 10. NAIVE BAYES
# ============================================================
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# ============================================================
# 11. LOGISTIC REGRESSION
# ============================================================
lr = LogisticRegression()
lr.fit(train_x_vector, train_y)

# ============================================================
# 12. MODEL EVALUATION 
# ============================================================
print(svc.score(test_x_vector, test_y)) #0.8378787878787879
print(des_tree.score(test_x_vector, test_y)) #0.7166666666666667
print(gnb.score(test_x_vector.toarray(), test_y)) #0.6075757575757575
print(lr.score(test_x_vector, test_y)) #0.8454545454545455