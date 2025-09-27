import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregar dataset limpo
df = pd.read_csv("data/carclaims_clean.csv")

# Separar features e target
X = df.drop("FraudFound", axis=1)
y = df["FraudFound"].map({"No": 0, "Yes": 1})

# Transformar variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalar dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar modelo
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Avaliar
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))

# -----------------------
# Visualização de métricas
# -----------------------

# Matriz de Confusão como Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud","Fraud"], yticklabels=["No Fraud","Fraud"])
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Matriz de Confusão")
plt.show()

# Distribuição de previsões
plt.figure(figsize=(5,3))
sns.countplot(x=y_pred)
plt.xticks([0,1], ["No Fraud","Fraud"])
plt.title("Distribuição das Previsões")
plt.show()

# Comparar fraude real x prevista
plt.figure(figsize=(6,4))
df_compare = pd.DataFrame({"Real": y_test, "Predito": y_pred})
sns.countplot(x="Real", hue="Predito", data=df_compare)
plt.xticks([0,1], ["No Fraud","Fraud"])
plt.title("Fraude Real vs Previsão")
plt.show()
