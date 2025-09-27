import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar dataset limpo
df = pd.read_csv("data/carclaims_clean.csv")

# 2. Separar variáveis preditoras (X) e alvo (y)
X = df.drop("FraudFound", axis=1)
y = df["FraudFound"]

# 3. Dividir em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Criar e treinar o modelo
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 5. Fazer previsões
y_pred = model.predict(X_test)

# 6. Avaliar o modelo
print("\n--- Avaliação do Modelo ---")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# 7. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Não Fraude", "Fraude"], yticklabels=["Não Fraude", "Fraude"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()
