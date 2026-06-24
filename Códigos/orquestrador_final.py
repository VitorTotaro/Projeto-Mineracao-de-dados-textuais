import pandas as pd
import joblib
import re
import gc  # Garbage Collector para não estourar a memória RAM

print("🚀 INICIANDO O ORQUESTRADOR DE PRODUÇÃO...")

print("🧠 1. Carregando a 'Trindade' de Machine Learning...")
clf = joblib.load('xgboost_producao.pkl')
vectorizer = joblib.load('tfidf_producao.pkl')
encoder = joblib.load('encoder_producao.pkl')

def extrair_valor_inteligente(texto, classe):
    # Primeiro, pega todos os valores formatados como R$ do texto
    matches = re.findall(r'R\$\s*([\d\.]*,\d{2})', texto, re.IGNORECASE)
    if not matches: return None
    
    valores = []
    for m in matches:
        val_str = m.replace('.', '').replace(',', '.')
        try: valores.append(float(val_str))
        except ValueError: continue
    if not valores: return None

    # Lógica para Licitação (Mantém o maior valor, que costuma ser o teto da compra)
    if classe == 'Licitação':
        return max(valores)
        
    # Lógica Turbinada para Aditivo
    if classe == 'Aditivo':
        # Tentativa 1: Busca o valor imediatamente após palavras de acréscimo
        match_acrescimo = re.search(r'(?:acresce|acréscimo|adita|adicional|mais).*?R\$\s*([\d\.]*,\d{2})', texto, re.IGNORECASE)
        if match_acrescimo:
            return float(match_acrescimo.group(1).replace('.', '').replace(',', '.'))
            
        # Tentativa 2: Se tiver mais de um valor citado, descarta o maior (que geralmente é o Global) 
        # e pega o maior valor dos que sobraram.
        if len(valores) > 1:
            valores.remove(max(valores))
            return max(valores)
        
        # Fallback: Se só citaram 1 valor no texto inteiro, tem que ser ele.
        return valores[0]

def extrair_processo_robusto(texto):
    # O segredo aqui é o '20\d{2}', que obriga o ID a terminar com um ano (ex: 2024, 2025)
    # Isso elimina instantaneamente qualquer data como 28/02 ou 30/10
    match = re.search(r'processo.*?\b(\d{2,6}\s*[-/]\s*20\d{2})\b', texto, re.IGNORECASE)
    if match: return re.sub(r'\s+', '', match.group(1)).replace('-', '/')
    
    match_contrato = re.search(r'contrato.*?\b(\d{2,6}\s*[-/]\s*20\d{2})\b', texto, re.IGNORECASE)
    if match_contrato: return re.sub(r'\s+', '', match_contrato.group(1)).replace('-', '/')
    
    return "Não Identificado"

# Caminhos dos arquivos (Ajuste 'base_completa.parquet' para o nome do seu arquivo gigante)
arquivo_entrada = 'base_completa_parquet/' 
arquivo_saida = 'dados_finais_powerbi.csv'

print(f"📦 2. Processando o arquivo gigante: {arquivo_entrada}")

# Lendo o arquivo em blocos para salvar a memória (se for parquet, o pyarrow cuida disso internamente)
df_gigante = pd.read_parquet(arquivo_entrada).dropna(subset=['paragrafo'])

# Criando o arquivo CSV vazio apenas com os cabeçalhos para ir adicionando os dados aos poucos
cabecalhos = ['municipio', 'date', 'classe_alvo', 'Chave_Unica', 'valor_R$']
pd.DataFrame(columns=cabecalhos).to_csv(arquivo_saida, index=False, sep=';', decimal=',')

# Definindo o tamanho do lote (processa 10.000 parágrafos por vez)
tamanho_lote = 10000
total_linhas = len(df_gigante)
linhas_salvas = 0

print(f"⚙️ Iniciando processamento em lotes de {tamanho_lote}...")

for inicio in range(0, total_linhas, tamanho_lote):
    fim = inicio + tamanho_lote
    lote = df_gigante.iloc[inicio:fim].copy()
    
    # Passo A: Inferência do Machine Learning
    X_tfidf = vectorizer.transform(lote['paragrafo'])
    lote['classe_prevista'] = encoder.inverse_transform(clf.predict(X_tfidf))
    
    # Passo B: Filtro de Ruído (Deixa só Licitação e Aditivo)
    lote = lote[lote['classe_prevista'].isin(['Licitação', 'Aditivo'])]
    
    # Passo C: O FILTRO DE NEGÓCIO (Excluindo SRP e Consórcios)
    termos_proibidos = r'registro de preç|srp|consórcio intermunicipal|ata de registro|cimasp'
    lote = lote[~lote['paragrafo'].str.contains(termos_proibidos, case=False, regex=True, na=False)]
    
    # Se o lote ficou vazio após os filtros, pula para o próximo
    if lote.empty:
        continue
        
    # Passo D: A Extração de RegEx
    lote['valor_R$'] = lote.apply(lambda row: extrair_valor_inteligente(row['paragrafo'], row['classe_prevista']), axis=1)
    lote['id_contrato'] = lote['paragrafo'].apply(extrair_processo_robusto)
    
    # Removendo quem não teve valor extraído ou contrato identificado
    lote = lote.dropna(subset=['valor_R$'])
    lote = lote[lote['id_contrato'] != "Não Identificado"]
    
    # Passo E: Criando a Chave Composta diretamente no Python (Poupa trabalho no Power BI)
    lote['Chave_Unica'] = lote['municipio'] + " | " + lote['id_contrato']
    
    # Passo F: Salvando o lote no CSV de forma incremental (append)
    lote_final = lote[['municipio', 'date', 'classe_prevista', 'Chave_Unica', 'valor_R$']]
    lote_final.to_csv(arquivo_saida, mode='a', index=False, header=False, sep=';', decimal=',')
    
    linhas_salvas += len(lote_final)
    print(f"   -> Lote processado. Linhas úteis encontradas e salvas: {len(lote_final)}")
    
    # Limpando a memória na marra antes do próximo ciclo
    del lote
    del X_tfidf
    gc.collect()

print(f"\n🎉 GOLAÇO! Processamento completo. {linhas_salvas} licitações e aditivos reais extraídos!")
print(f"Arquivo '{arquivo_saida}' está pronto para ser injetado no Power BI.")