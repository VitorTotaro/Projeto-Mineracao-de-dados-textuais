import pandas as pd
import requests
import re
from io import BytesIO
from urllib.parse import urljoin
import urllib3
from pdfminer.high_level import extract_text
import os
import glob
import math

urllib3.disable_warnings()

# ================= CONFIGURAÇÕES =================
PASTA_ORIGEM = 'dados_brutos'
ORCAMENTO_GLOBAL_PDFS = 5000 
# =================================================

def extrair_texto_pdf(url):
    if not url: return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
        resp = requests.get(url, headers=headers, timeout=20, verify=False)
        tipo = resp.headers.get('Content-Type', '').lower()
        
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

# Movendo o rotulador para cima para usarmos durante o download
def classificar_texto(texto):
    if re.search(r'(aviso de licita[çc][ãa]o|tomada de pre[çc]os|preg[ãa]o eletr[ôo]nico)', texto):
        return 'Licitação'
    elif re.search(r'(termo aditivo|aditamento ao contrato)', texto):
        return 'Aditivo'
    else:
        return 'Outros'

print("🔍 Fase 1: Mapeando o universo de dados...")
arquivos_json = glob.glob(f"{PASTA_ORIGEM}/*.json")

dados_cidades = {}
total_diarios_universo = 0

for arq in arquivos_json:
    try:
        df_temp = pd.read_json(arq)
        qtd = len(df_temp)
        if qtd > 0:
            nome_cidade = os.path.basename(arq).replace('.json', '')
            dados_cidades[nome_cidade] = {'df': df_temp, 'qtd_total': qtd}
            total_diarios_universo += qtd
    except ValueError:
        continue

print(f"🌍 Universo mapeado: {total_diarios_universo} diários em {len(dados_cidades)} cidades.")
print("\n⚖️ Fase 2 e 3 Integradas: Processamento On-the-Fly (Anti-Estouro de Memória)...")

dados_finais = []
pdfs_baixados = 0

for cidade, infos in dados_cidades.items():
    proporcao = infos['qtd_total'] / total_diarios_universo
    cota = math.ceil(proporcao * ORCAMENTO_GLOBAL_PDFS)
    cota = min(cota, infos['qtd_total'])
    
    if cota <= 0: continue
    print(f"\n📌 {cidade.upper()} | Peso: {proporcao:.1%} | Cota: {cota} PDFs")
    
    df_amostra = infos['df'].sample(n=cota, random_state=42)
    
    # Processa linha por linha
    for index, row in df_amostra.iterrows():
        url = row.get('file_urls', [None])
        url_limpa = url[0] if isinstance(url, list) and len(url) > 0 else None
        
        texto_pdf = extrair_texto_pdf(url_limpa)
        pdfs_baixados += 1
        print(f"   Progresso Global: {pdfs_baixados}/{ORCAMENTO_GLOBAL_PDFS} PDFs baixados...", end='\r')
        
        if not texto_pdf:
            continue
            
        # O SEGREDO: Fatiar e classificar imediatamente antes de jogar no Pandas
        paragrafos = str(texto_pdf).split('\n\n')
        
        for p in paragrafos:
            p_limpo = p.strip().lower()
            # Filtro de tamanho para jogar lixo fora cedo
            if len(p_limpo) > 50:
                classe = classificar_texto(p_limpo)
                
                # Monta a linha pronta
                dados_finais.append({
                    'date': row.get('date', None),
                    'municipio': cidade,
                    'edition_number': row.get('edition_number', None),
                    'paragrafo': p,
                    'classe_alvo': classe
                })

print("\n\n🔨 Fase Final: Salvando os dados...")
# Transforma a lista limpa diretamente no DataFrame final (sem explode)
df_final = pd.DataFrame(dados_finais)

print("\n📊 --- BALANÇO DA AMOSTRA ESTRATIFICADA ---")
print(df_final['classe_alvo'].value_counts())

#A CONVERSÃO DE TIPO (A vacina contra o erro do Parquet)
df_final['edition_number'] = df_final['edition_number'].astype(str)

df_final.to_parquet('amostra_estratificada.parquet', index=False)
print("\n🎉 PRONTO! Base estratificada salva como 'amostra_estratificada.parquet'.")