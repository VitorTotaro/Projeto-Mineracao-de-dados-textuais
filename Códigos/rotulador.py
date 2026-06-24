import pandas as pd
import re

print("Carregando os dados brutos...")
# Lê o CSV que você acabou de gerar
df = pd.read_csv('teste_textos_contagem.csv', sep=';')

# Joga fora linhas que não conseguiram baixar o texto
df = df.dropna(subset=['texto_bruto'])

print("Fatiando os textos gigantes em parágrafos...")
# 1. Divide o texto gigante em uma lista de parágrafos usando a quebra de linha dupla
df['paragrafo'] = df['texto_bruto'].apply(lambda x: str(x).split('\n\n'))

# 2. EXPLODE! Transforma cada item da lista em uma nova linha do DataFrame
df_fatiado = df.explode('paragrafo')

# Limpeza básica: remove espaços em branco inúteis e joga tudo para minúsculo para facilitar a busca
df_fatiado['paragrafo_limpo'] = df_fatiado['paragrafo'].str.strip().str.lower()

# Remove parágrafos muito curtos (como "Página 1", "Contagem", etc)
df_fatiado = df_fatiado[df_fatiado['paragrafo_limpo'].str.len() > 50]

print(f"Total de parágrafos isolados: {len(df_fatiado)}")

# 3. O ROTULADOR HEURÍSTICO (O embrião do seu modelo!)
def classificar_texto(texto):
    # Se encontrar palavras fortes de licitação, ganha o rótulo
    if re.search(r'(aviso de licita[çc][ãa]o|tomada de pre[çc]os|preg[ãa]o eletr[ôo]nico)', texto):
        return 'Licitação'
    # Se encontrar palavras de aditivo
    elif re.search(r'(termo aditivo|aditamento ao contrato)', texto):
        return 'Aditivo'
    else:
        return 'Outros'

print("Aplicando as regras de rotulação...")
df_fatiado['classe_alvo'] = df_fatiado['paragrafo_limpo'].apply(classificar_texto)

# 4. Estatísticas do que encontramos
resumo = df_fatiado['classe_alvo'].value_counts()
print("\n--- Resultado da Rotulação ---")
print(resumo)

# Salva a base pronta para Machine Learning!
df_final = df_fatiado[['date', 'edition_number', 'paragrafo', 'classe_alvo']]
df_final.to_csv('base_treino_contagem.csv', index=False, sep=';')
print("\n✅ Base de treino gerada com sucesso!")