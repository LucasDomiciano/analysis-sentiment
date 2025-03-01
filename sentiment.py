# sentiment.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregar um dataset simples (20 newsgroups, adaptado para exemplo)
from sklearn.datasets import fetch_20newsgroups

# Pegar dados de exemplo (positivos e negativos simulados)
categories = ['alt.atheism', 'soc.religion.christian']  # Exemplos de temas opostos
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
texts = data.data
labels = data.target  # 0 ou 1 (negativo/positivo)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Transformar texto em números (vetorização)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinar o modelo
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Função para prever novos textos
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Positivo" if prediction == 1 else "Negativo"

# Testar com exemplos
print(predict_sentiment("I love this product, it's amazing!"))
print(predict_sentiment("This is terrible, I hate it."))

# No final do sentiment.py
while True:
    user_input = input("Digite um texto (ou 'sair'): ")
    if user_input.lower() == 'sair':
        break
    print(f"Sentimento: {predict_sentiment(user_input)}")