import pandas as pd

# Carregar dataset
df = pd.read_csv("data/carclaims.csv")

# Visualizar primeiras linhas
print(df.head())

# Info geral do dataset
print(df.info())

# Estatísticas descritivas
print(df.describe())


