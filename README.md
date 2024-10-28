# datasciencePROVA
Base para a Prova de Data Science.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from google.colab import files
import io

# Carregando a base de dados
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['imdb_labelled (3).txt']), sep='\t', names=['text', 'target'])

# Visualizando o balanceamento das classes
contagem_Classes = df['target'].value_counts(normalize=True) * 100
contagem_Classes.plot(kind='bar', title="Distribuição das Classes (%)")
plt.xlabel("Classes")
plt.ylabel("Porcentagem (%)")
plt.show()

# Separação das variáveis
X = df['text']
y = df['target']

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Extração de características usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Modelo 1: RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_tfidf, y_train)
y_pred_rf = model_rf.predict(X_test_tfidf)
y_pred_proba_rf = model_rf.predict_proba(X_test_tfidf)[:, 1]

# Modelo 2: LogisticRegression
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_tfidf, y_train)
y_pred_lr = model_lr.predict(X_test_tfidf)
y_pred_proba_lr = model_lr.predict_proba(X_test_tfidf)[:, 1]

# Função para calcular e exibir as métricas
def calcular_metricas(y_test, y_pred, y_pred_proba, modelo_nome):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recalls, precisions)
    
    print(f"Métricas para o modelo {modelo_nome}:")
    print("Acurácia:", accuracy)
    print("Precisão:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("AUC-ROC:", roc_auc)
    print("AUC-PR:", pr_auc)
    print("="*50)
    
    # Curva ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f}) - {modelo_nome}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {modelo_nome}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
    # Curva PR
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='purple', label=f'Curva PR (AUC = {pr_auc:.2f}) - {modelo_nome}')
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title(f'Curva de Precisão-Recall - {modelo_nome}')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

# Calculando métricas para ambos os modelos
calcular_metricas(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest")
calcular_metricas(y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression")

# Pontos de Corte para Análise Manual
# Definindo os pontos de corte
cut_off_negative = 0.4
cut_off_positive = 0.7

# Visualizando a distribuição acumulada de probabilidades
populacao = y_pred_proba_rf.tolist()
populacao_positivado = y_pred_proba_rf[y_test == 1].tolist()

bins = np.arange(0, 1.1, 0.1)
hist_populacao, _ = np.histogram(populacao, bins=bins)
hist_populacao_positivado, _ = np.histogram(populacao_positivado, bins=bins)

hist_populacao = hist_populacao / len(populacao) * 100
hist_populacao_positivado = hist_populacao_positivado / len(populacao_positivado) * 100

bar_width = 0.35
x_pos = np.arange(len(hist_populacao))

plt.figure(figsize=(10, 6))
plt.bar(x_pos, hist_populacao, width=bar_width, label='Toda a População', color='blue')
plt.bar(x_pos + bar_width, hist_populacao_positivado, width=bar_width, label='População Positivada', color='red')

plt.xticks(x_pos + bar_width / 2, [f'{int(b * 100)}%' for b in bins[:-1]])
plt.axvline(x=cut_off_negative * 10, color='green', linestyle='--', label='Corte Negativo')
plt.axvline(x=cut_off_positive * 10, color='orange', linestyle='--', label='Corte Positivo')
plt.axvline(x=0.5 * 10, color='purple', linestyle='--', label='Análise Manual')

plt.title('Distribuição Acumulada de Probabilidades')
plt.xlabel('Chance (%)')
plt.ylabel('Porcentagem da População')
plt.legend()
plt.show()

