import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("📂 1. Carregando o minério refinado...")
df = pd.read_parquet('amostra_estratificada.parquet')

# Filtra linhas vazias por segurança
df = df.dropna(subset=['paragrafo', 'classe_alvo'])

print("🔀 2. Separando Treino (80%) e Teste (20%)...")
X = df['paragrafo']
y = df['classe_alvo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🧮 3. Vetorização (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, lowercase=True) 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("🧠 4. Treinando o modelo Baseline (Random Forest)...")
clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf.fit(X_train_tfidf, y_train)

print("🎯 5. Avaliando o modelo e gerando as imagens de documentação...\n")
y_pred = clf.predict(X_test_tfidf)
classes = clf.classes_

# --- GERANDO A IMAGEM DA MATRIZ DE CONFUSÃO ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))

# Usamos o Seaborn para criar um mapa de calor (heatmap) elegante
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            cbar=False, annot_kws={"size": 12})

plt.ylabel('Classe Real', fontsize=12, fontweight='bold')
plt.xlabel('Classe Prevista', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusão - Modelo Baseline', fontsize=14, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_confusao.png', dpi=300)
plt.close()
print("📸 Imagem 'matriz_confusao.png' salva!")

# --- GERANDO A IMAGEM DO RELATÓRIO DE CLASSIFICAÇÃO ---
# Pegamos o relatório em formato de dicionário para converter em DataFrame
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)

plt.figure(figsize=(10, 4))
plt.axis('off') # Esconde os eixos do gráfico, queremos só a tabela

# Desenhamos o DataFrame como uma tabela visual
tabela = plt.table(cellText=report_df.values,
                   rowLabels=report_df.index,
                   colLabels=report_df.columns,
                   cellLoc='center', 
                   loc='center',
                   bbox=[0, 0, 1, 1])

tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.2, 1.2)
plt.title('Relatório de Classificação - Modelo Baseline', fontsize=14, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('relatorio_classificacao.png', dpi=300)
plt.close()
print("📸 Imagem 'relatorio_classificacao.png' salva!")

# Salva também em CSV
report_df.to_csv('relatorio_classificacao.csv', sep=';', decimal=',')

print("\n✅ Entrega 1 concluída! Arquivos de imagem prontos na sua pasta.")