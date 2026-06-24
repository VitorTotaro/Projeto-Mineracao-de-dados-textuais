import pandas as pd
import requests
import re
from io import BytesIO
from urllib.parse import urljoin
import urllib3
from pdfminer.high_level import extract_text
import os

urllib3.disable_warnings()

# 1. A Lista de Ouro 
cidades_amostra = [
    'mg_belo_horizonte', 'ba_salvador', 'mg_contagem', 
    'sp_jundiai', 'sp_marilia', 'mg_betim', 'ba_lajedao'
]

pasta_origem = 'dados_brutos'
limite_por_cidade = 60 # Pega 60 diários de cada cidade para a base de treino
dados_finais = []

# Nossa função blindada de extração (agora otimizada e sem prints excessivos)
def extrair_texto_pdf(url):
    if not url: return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
                   'Accept': 'application/pdf,text/html;q=0.9'}
        
        resp = requests.get(url, headers=headers, timeout=20, verify=False)
        tipo = resp.headers.get('Content-Type', '').lower()
        
        # Lida com o falso redirecionamento
        if 'html' in tipo and 'meta http-equiv="refresh"' in resp.text.lower():
            match = re.search(r'url=([^"\']+)', resp.text, re.IGNORECASE)
            if match:
                url_verdadeira = urljoin(url, match.group(1))
                resp = requests.get(url_verdadeira, headers=headers, timeout=20, verify=False)
                tipo = resp.headers.get('Content-Type', '').lower()
        
        if 'pdf' in tipo:
            return extract_text(BytesIO(resp.content))
        return None
    except:
        return None

# 2. O Loop de Extração
print("🚀 Iniciando a Fábrica de Treino para o modelo...\n")

for cidade in cidades_amostra:
    caminho = f"{pasta_origem}/{cidade}.json"
    
    if not os.path.exists(caminho):
        print(f"⚠️ Arquivo não encontrado: {caminho}. Pulando...")
        continue
        
    print(f"📌 Processando amostra de: {cidade.upper()}")
    df_cidade = pd.read_json(caminho)
    
    # Pega uma amostra aleatória para não viciar em um único mês
    df_amostra = df_cidade.sample(n=min(limite_por_cidade, len(df_cidade)), random_state=42).copy()
    
    df_amostra['url_limpa'] = df_amostra['file_urls'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    
    # Barra de progresso visual simples no terminal
    textos = []
    for i, url in enumerate(df_amostra['url_limpa']):
        print(f"   Lendo PDF {i+1}/{len(df_amostra)}...", end='\r')
        textos.append(extrair_texto_pdf(url))
        
    df_amostra['texto_bruto'] = textos
    df_amostra['municipio'] = cidade # Guarda a origem para análises futuras
    
    # O .copy() evita alertas de memória do Pandas
    df_valido = df_amostra.dropna(subset=['texto_bruto']).copy()
    
    if not df_valido.empty:
        # Se o robô da cidade esqueceu alguma coluna, nós criamos ela vazia
        if 'edition_number' not in df_valido.columns:
            df_valido['edition_number'] = None
        if 'date' not in df_valido.columns:
            df_valido['date'] = None
            
        # Agora podemos anexar sem medo do script quebrar
        dados_finais.append(df_valido[['date', 'edition_number', 'municipio', 'texto_bruto']])
    print(f"\n   ✅ {len(df_valido)} Diários extraídos com sucesso.\n")

# 3. Consolidação e Fatiamento
print("🔨 Consolidando os dados de todas as cidades...")
df_consolidado = pd.concat(dados_finais, ignore_index=True)

print("🔪 Fatiando os textos gigantes em parágrafos independentes...")
df_consolidado['paragrafo'] = df_consolidado['texto_bruto'].apply(lambda x: str(x).split('\n\n'))
df_fatiado = df_consolidado.explode('paragrafo')

# Limpeza e filtro
df_fatiado['paragrafo_limpo'] = df_fatiado['paragrafo'].str.strip().str.lower()
df_fatiado = df_fatiado[df_fatiado['paragrafo_limpo'].str.len() > 50]

# 4. O Rotulador Heurístico
def classificar_texto(texto):
    if re.search(r'(aviso de licita[çc][ãa]o|tomada de pre[çc]os|preg[ãa]o eletr[ôo]nico)', texto):
        return 'Licitação'
    elif re.search(r'(termo aditivo|aditamento ao contrato)', texto):
        return 'Aditivo'
    else:
        return 'Outros'

print("🏷️ Aplicando regras de rotulação (Ground Truth)...")
df_fatiado['classe_alvo'] = df_fatiado['paragrafo_limpo'].apply(classificar_texto)

resumo = df_fatiado['classe_alvo'].value_counts()
print("\n📊 --- BALANÇO FINAL DA AMOSTRA ---")
print(resumo)

# 5. Salvamento Otimizado em Parquet
colunas_finais = ['date', 'municipio', 'edition_number', 'paragrafo', 'classe_alvo']
df_final = df_fatiado[colunas_finais]

nome_arquivo = 'amostra_treinamento.parquet'
df_final.to_parquet(nome_arquivo, index=False)

print(f"\n🎉 PRONTO! Base salva como '{nome_arquivo}'.")
print("O minério foi refinado. Estamos prontos para a fase de Machine Learning!")