# Análise de Sentimentos com Python
Um projeto simples de Machine Learning para classificar textos como positivos ou negativos.

## Como funciona
- Usa Scikit-learn para treinar um modelo de regressão logística.
- Transforma texto em números com TF-IDF.
- Dataset: Subconjunto do 20 Newsgroups (adaptado).

## Rodando o projeto
1. Clone o repositório: `git clone [seu-link]`
2. Instale dependências: `pip install -r requirements.txt`
3. Execute: `python sentiment.py`

## Exemplo de saída
- "I love this product, it's amazing!" → Positivo
- "This is terrible, I hate it." → Negativo

## Próximos passos
- Testar com datasets maiores (ex.: IMDB Reviews).
- Adicionar suporte a "neutro".