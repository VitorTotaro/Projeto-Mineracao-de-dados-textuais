import pandas as pd
import requests
import re
from urllib.parse import urljoin
from io import BytesIO
from pdfminer.high_level import extract_text

# 1. Carregar o JSON do município (ajuste o nome do arquivo se necessário)
caminho_arquivo = 'dados_brutos/mg_contagem.json'
df = pd.read_json(caminho_arquivo)

print(f"Total de diários encontrados: {len(df)}")

# 2. VAMOS TESTAR PEQUENO: Pegar apenas as 3 primeiras linhas
df_teste = df.head(3).copy()

# A coluna 'file_urls' vem como uma lista no JSON. Vamos pegar o primeiro link:
df_teste['url_limpa'] = df_teste['file_urls'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

def ler_pdf_da_nuvem(url):
    if not url: return None
    
    try:
        print(f"⏳ Acessando: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8'
        }
        
        import urllib3
        urllib3.disable_warnings()
        
        # Primeira requisição (vai bater na página de redirecionamento)
        resposta = requests.get(url, headers=headers, timeout=30, verify=False)
        tipo_arquivo = resposta.headers.get('Content-Type', '').lower()
        
        # Tática de Interceptação: Se for HTML e tiver a tag de refresh...
        if 'html' in tipo_arquivo and 'meta http-equiv="refresh"' in resposta.text.lower():
            print("   -> 🔄 Redirecionamento HTML detectado! Extraindo link real...")
            
            # Caça o link usando Regex
            match = re.search(r'url=([^"\']+)', resposta.text, re.IGNORECASE)
            
            if match:
                url_escondida = match.group(1)
                # O urljoin junta o domínio base "portal.contagem..." com o "/uploads/..."
                url_verdadeira = urljoin(url, url_escondida)
                print(f"   -> Novo alvo: {url_verdadeira}")
                
                # Faz o segundo ataque, agora no link direto do PDF
                resposta = requests.get(url_verdadeira, headers=headers, timeout=30, verify=False)
                tipo_arquivo = resposta.headers.get('Content-Type', '').lower()
            else:
                print("   -> ❌ Link escondido não encontrado no HTML.")
                return None
        
        # Validação Final
        if 'pdf' not in tipo_arquivo:
            print("   -> ❌ FALSO PDF! O servidor continuou bloqueando a entrega.")
            return None
            
        # Sucesso! Joga o PDF na RAM e extrai o texto
        print("   -> 📄 PDF alcançado! Extraindo textos...")
        arquivo_pdf = BytesIO(resposta.content)
        texto = extract_text(arquivo_pdf)
        return texto
    
    except Exception as e:
        print(f"❌ Erro ao ler {url}: {e}")
        return None

# 3. Aplica a função para criar a nossa tão aguardada coluna de texto bruto
print("-" * 40)
df_teste['texto_bruto'] = df_teste['url_limpa'].apply(ler_pdf_da_nuvem)
print("-" * 40)

# 4. Mostrando o resultado final
print("\nPrévia dos dados extraídos:")
print(df_teste[['date', 'edition_number', 'texto_bruto']])

# Salvar o teste num CSV para você visualizar depois
df_teste.to_csv('teste_textos_contagem.csv', index=False, sep=';')
print("\n✅ Arquivo teste_textos_contagem.csv salvo com sucesso!")