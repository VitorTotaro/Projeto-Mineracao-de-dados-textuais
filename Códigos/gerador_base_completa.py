import os
import glob
import pandas as pd
import gc
import re
import requests
from io import BytesIO
from urllib.parse import urljoin
from pdfminer.high_level import extract_text
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("🚜 INICIANDO A EXTRAÇÃO DA BASE COMPLETA (ON-THE-FLY)...")

pasta_saida = 'base_completa_parquet'
pasta_jsons = 'dados_brutos' 
os.makedirs(pasta_saida, exist_ok=True)

# --- 1. LENDO TODOS OS ARQUIVOS JSON E INJETANDO O MUNICÍPIO ---
print(f"📂 Procurando arquivos .json na pasta '{pasta_jsons}'...")
arquivos_json = glob.glob(os.path.join(pasta_jsons, "*.json"))

if not arquivos_json:
    print(f"❌ Nenhum arquivo .json encontrado na pasta '{pasta_jsons}'.")
    exit()

lista_dataframes = []
for arquivo in arquivos_json:
    try:
        df_temp = pd.read_json(arquivo)
        
        # MÁGICA 1: Pega o nome do arquivo (ex: 'ba_angical.json') e vira o nome do município
        nome_municipio = os.path.basename(arquivo).replace('.json', '')
        df_temp['municipio'] = nome_municipio
        
        lista_dataframes.append(df_temp)
    except Exception as e:
        print(f"⚠️ Erro ao ler o arquivo {arquivo}: {e}")

df_scrapy = pd.concat(lista_dataframes, ignore_index=True)
print(f"✅ Base bruta unificada! {len(df_scrapy)} Diários Oficiais encontrados.")

# --- 2. FUNÇÃO DE EXTRAÇÃO DO PDF ---
def extrair_paragrafos_do_pdf(url):
    if not url: return []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/pdf,text/html;q=0.9'
        }
        
        resp = requests.get(url, headers=headers, timeout=20, verify=False)
        tipo = resp.headers.get('Content-Type', '').lower()
        
        if 'html' in tipo and 'meta http-equiv="refresh"' in resp.text.lower():
            match = re.search(r'url=([^"\']+)', resp.text, re.IGNORECASE)
            if match:
                url_verdadeira = urljoin(url, match.group(1))
                resp = requests.get(url_verdadeira, headers=headers, timeout=20, verify=False)
                tipo = resp.headers.get('Content-Type', '').lower()
        
        if 'pdf' in tipo:
            texto_bruto = extract_text(BytesIO(resp.content))
            paragrafos_crus = re.split(r'\n\s*\n', texto_bruto)
            return [re.sub(r'\s+', ' ', p).strip() for p in paragrafos_crus]
            
        return []
    except Exception:
        return []

# --- 3. PROCESSAMENTO EM LOTES ---
tamanho_lote = 50 
total_arquivos = len(df_scrapy)
lote_atual = []
contador_particoes = 1

print(f"\n⚙️ Iniciando o download e fatiamento dos PDFs...")

for index, row in df_scrapy.iterrows():
    municipio = row.get('municipio', 'Desconhecido')
    data_pub = row.get('date', 'Desconhecida')
    
    # MÁGICA 2: Acessando a lista dentro do JSON para extrair a URL limpa
    lista_urls = row.get('file_urls')
    
    # Valida se realmente é uma lista e se não está vazia antes de pegar o índice [0]
    if isinstance(lista_urls, list) and len(lista_urls) > 0:
        caminho_pdf = lista_urls[0]
    else:
        continue # Se o diário não tiver link, pula pro próximo
    
    paragrafos = extrair_paragrafos_do_pdf(caminho_pdf)
    
    for p in paragrafos:
        if len(p.strip()) > 20:
            lote_atual.append({
                'municipio': municipio,
                'date': data_pub,
                'paragrafo': p
            })
            
    if (index + 1) % tamanho_lote == 0 or (index + 1) == total_arquivos:
        if lote_atual:
            df_lote = pd.DataFrame(lote_atual)
            nome_arquivo = f"{pasta_saida}/particao_{contador_particoes:04d}.parquet"
            df_lote.to_parquet(nome_arquivo, index=False)
            
            print(f"💾 Lote {contador_particoes} salvo com {len(df_lote)} parágrafos. ({index + 1}/{total_arquivos} PDFs processados)")
            
            contador_particoes += 1
            lote_atual = [] 
            
            del df_lote
            gc.collect()

print("\n✅ EXTRAÇÃO COMPLETA CONCLUÍDA COM SUCESSO!")