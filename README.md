# Análise de Fraude em Seguros de Automóveis

Este projeto explora dados de sinistros de seguros de automóveis e aplica modelos de machine learning para identificar possíveis casos de fraude. O objetivo é demonstrar o processo completo de análise de dados, desde a exploração inicial até a avaliação de modelos.

## Sobre o Dataset

O dataset contém informações detalhadas sobre sinistros, incluindo dados do veículo, do segurado e do acidente. A variável alvo é `FraudFound`, indicando se o sinistro foi considerado fraude ou não.

## Estrutura do Projeto

- `data/`  
  Contém o dataset original (`carclaims.csv`) e o dataset limpo (`carclaims_clean.csv`).

- `analiseinicial.py`  
  Carrega o dataset e realiza análise exploratória inicial com estatísticas descritivas.

- `limpeza.py`  
  Trata valores nulos, remove duplicados e converte variáveis categóricas.

- `modelagem.py`  
  Aplica um modelo de Random Forest para classificação de fraudes, avaliando desempenho com métricas e matrizes de confusão.

- `regressao.py`  
  Implementa regressão logística para comparação de resultados e visualização de previsões versus realidade.

- `analisemlseguros.ipynb`  
  Notebook Jupyter integrando análise, visualizações e explicações em Markdown.

## Como Executar

1. Certifique-se de ter o Python 3 instalado.  
2. Instale as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

3. Execute os scripts em sequência para reproduzir a análise:

analiseinicial.py → limpeza.py → modelagem.py → regressao.py

Ou abra o notebook analisemlseguros.ipynb no VS Code ou Jupyter.

Resultados

O projeto gera:

Estatísticas descritivas e análise exploratória.

Métricas de desempenho dos modelos (acurácia, precisão, recall e F1-score).

Matrizes de confusão e gráficos comparando previsões e realidade.

Esses resultados ajudam a entender como os modelos identificam fraudes e quais variáveis são mais relevantes.

Aprendizados

Durante o desenvolvimento, foi possível:

Praticar o fluxo completo de um projeto de machine learning: limpeza, pré-processamento, modelagem e avaliação.

Observar o impacto do desbalanceamento das classes nas métricas do modelo.

Comparar diferentes algoritmos (Random Forest e Regressão Logística) e visualizar suas performances.

Próximos Passos

Experimentar técnicas de balanceamento de dados (SMOTE, oversampling, undersampling).

Testar outros algoritmos de classificação e ajustar hiperparâmetros.

Criar uma interface simples para enviar novos dados e gerar previsões em tempo real.
