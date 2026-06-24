import pandas as pd
import joblib
import re

print("📂 1. Carregando os dados brutos e o Modelo de Produção...")
df = pd.read_parquet('amostra_estratificada.parquet').dropna(subset=['paragrafo', 'classe_alvo'])

clf = joblib.load('xgboost_producao.pkl')
vectorizer = joblib.load('tfidf_producao.pkl')
encoder = joblib.load('encoder_producao.pkl')

print("🔍 2. O Modelo está lendo os diários (Inferência)...")
X_tfidf = vectorizer.transform(df['paragrafo'])
df['classe_prevista'] = encoder.inverse_transform(clf.predict(X_tfidf))

print("🗑️ 3. Filtrando o ruído...")
df_util = df[df['classe_prevista'].isin(['Licitação', 'Aditivo'])].copy()
print(f"   -> O modelo filtrou {len(df_util)} parágrafos importantes.")

print("🎣 4. Aplicando a RegEx Inteligente (Heurística de Negócio)...")

def extrair_valor_inteligente(texto, classe):
    # Encontra TODOS os valores monetários no parágrafo
    matches = re.findall(r'R\$\s*([\d\.]*,\d{2})', texto, re.IGNORECASE)
    if not matches:
        return None
        
    valores = []
    for m in matches:
        # Limpa os pontos e arruma a vírgula decimal
        val_str = m.replace('.', '').replace(',', '.')
        try:
            valores.append(float(val_str))
        except ValueError:
            continue
            
    if not valores:
        return None
        
    # A Lógica de Ouro
    if classe == 'Aditivo':
        # Aditivos costumam colocar o valor acrescido no final da frase
        return valores[-1] 
    else:
        # Licitações costumam ter o valor global como o maior número do texto
        return max(valores)

def extrair_processo_robusto(texto):
    # Procura a palavra 'Processo', ignora o que tiver no meio (como 'nº' ou 'Licitatório')
    # e captura rigidamente o padrão numérico NNN/YYYY
    match = re.search(r'processo.*?\b(\d{2,6}\s*[-/]\s*(?:20\d{2}|\d{2}))\b', texto, re.IGNORECASE)
    if match:
        return re.sub(r'\s+', '', match.group(1)).replace('-', '/')
    
    # Plano B: Se não achar a palavra 'Processo', tenta 'Contrato'
    match_contrato = re.search(r'contrato.*?\b(\d{2,6}\s*[-/]\s*(?:20\d{2}|\d{2}))\b', texto, re.IGNORECASE)
    if match_contrato:
        return re.sub(r'\s+', '', match_contrato.group(1)).replace('-', '/')
        
    return "Não Identificado"

# Aplicando as novas funções
df_util['valor_R$'] = df_util.apply(lambda row: extrair_valor_inteligente(row['paragrafo'], row['classe_prevista']), axis=1)
df_util['id_contrato'] = df_util['paragrafo'].apply(extrair_processo_robusto)

# Salvando a base final
nome_arquivo = 'dados_estruturados.csv'
df_util.to_csv(nome_arquivo, index=False, sep=';', decimal=',')
print(f"\n✅ Magia concluída! Arquivo '{nome_arquivo}' super atualizado.")