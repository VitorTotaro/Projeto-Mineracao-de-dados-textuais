import pandas as pd

print("🔎 Procurando a anomalia de Itajubá na base original...\n")

# Carrega o parquet original
df = pd.read_parquet('amostra_estratificada.parquet').dropna(subset=['paragrafo'])

# Filtra exatamente a cidade e o número do contrato que deu problema
filtro = (df['municipio'] == 'mg_itajuba') & (df['paragrafo'].str.contains('051/2025', na=False))
textos_suspeitos = df[filtro]['paragrafo'].tolist()

print(f"🚨 Encontramos {len(textos_suspeitos)} parágrafo(s). Lendo o conteúdo:\n")

for i, texto in enumerate(textos_suspeitos):
    print(f"--- PARÁGRAFO {i+1} ---")
    print(texto)
    print("-" * 50)