import subprocess
import os

# 1. Defina os estados que você quer (use os prefixos oficiais da lista)
estados_alvo = ['mg_', 'sp_', 'ba_']

# Parâmetros de data
data_inicio = '2025-01-01'
data_fim = '2025-12-31'

# Cria a pasta para salvar os arquivos, se não existir
pasta_saida = 'dados_brutos'
os.makedirs(pasta_saida, exist_ok=True)

print("Consultando a lista de robôs disponíveis...")
# 2. Roda o 'scrapy list' silenciosamente para pegar todos os nomes
resultado = subprocess.run(['scrapy', 'list'], capture_output=True, text=True, cwd='data_collection/gazette')

if resultado.returncode != 0:
    print("Erro ao listar os robôs. Verifique se o ambiente virtual está ativado.")
    exit()

todos_robos = resultado.stdout.splitlines()

# 3. Filtra apenas os robôs dos estados que queremos
robos_selecionados = [
    robo for robo in todos_robos 
    if any(robo.startswith(estado) for estado in estados_alvo)
]

print(f"🎯 Encontrados {len(robos_selecionados)} municípios para extração.\n")
print("-" * 40)

# 4. Inicia a linha de produção
for robo in robos_selecionados:
    arquivo_saida = f"../../{pasta_saida}/{robo}.json"
    
    print(f"⏳ Iniciando raspagem: {robo}...")
    
    comando = [
        'scrapy', 'crawl', robo,
        '-a', f'start={data_inicio}',
        '-a', f'end={data_fim}',
        '-o', arquivo_saida
    ]
    
    # Executa o comando esperando ele terminar antes de ir para o próximo
    subprocess.run(comando, cwd='data_collection/gazette')
    
    print(f"✅ {robo} finalizado e salvo!\n")

print("🎉 Extração em lote concluída com sucesso!")