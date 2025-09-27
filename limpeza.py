import pandas as pd

# 1. Carregar dataset
df = pd.read_csv("data/carclaims.csv")


# 2. Visão geral
print("Shape do dataset:", df.shape)
print("\nValores nulos por coluna:\n", df.isnull().sum())
print("\nDuplicados:", df.duplicated().sum())

# 3. Remover duplicados
df = df.drop_duplicates()

# 4. Tratar valores nulos
# Categorias → "Desconhecido", Numéricos → mediana
categorical_cols = df.select_dtypes(include=["object"]).columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna("Desconhecido")

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 5. Converter colunas categóricas em category
for col in categorical_cols:
    df[col] = df[col].astype("category")

# 6. Salvar dataset limpo
df.to_csv("data/carclaims_clean.csv", index=False)
print("\nDataset limpo salvo em 'data/carclaims_clean.csv'")

# 7. Conferir resultado final
print("\nShape final:", df.shape)
print("\nColunas finais:", df.columns.tolist())
