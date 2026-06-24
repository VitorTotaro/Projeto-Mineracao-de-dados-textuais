import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("📂 1. Carregando os dados e a 'Trindade' congelada...")
df = pd.read_parquet('amostra_estratificada.parquet').dropna(subset=['paragrafo', 'classe_alvo'])

# Lendo os arquivos que acabamos de salvar no HD
clf = joblib.load('xgboost_producao.pkl')
vectorizer = joblib.load('tfidf_producao.pkl')
encoder = joblib.load('encoder_producao.pkl')

print("🔀 2. Recriando o ambiente de Teste exato...")
y = encoder.transform(df['classe_alvo'])
X = df['paragrafo']

# O random_state=42 garante que a fatia de teste será exatamente a mesma do treinamento
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🧮 3. Transformando o texto e fazendo as predições...")
X_test_tfidf = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_tfidf)

nomes_classes = encoder.inverse_transform([0, 1, 2])

print("📸 4. Gerando as imagens em alta resolução para o artigo...")

# --- GERANDO A IMAGEM DA MATRIZ DE CONFUSÃO ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=nomes_classes, yticklabels=nomes_classes,
            cbar=False, annot_kws={"size": 12})

plt.ylabel('Classe Real', fontsize=12, fontweight='bold')
plt.xlabel('Classe Prevista', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusão - XGBoost + SMOTE', fontsize=14, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_confusao_xgboost.png', dpi=300)
plt.close()
print("   -> Imagem 'matriz_confusao_xgboost.png' salva!")

# --- GERANDO A IMAGEM DO RELATÓRIO DE CLASSIFICAÇÃO ---
report_dict = classification_report(y_test, y_pred, target_names=nomes_classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)

plt.figure(figsize=(10, 4))
plt.axis('off')

tabela = plt.table(cellText=report_df.values,
                   rowLabels=report_df.index,
                   colLabels=report_df.columns,
                   cellLoc='center', 
                   loc='center',
                   bbox=[0, 0, 1, 1])

tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.2, 1.2)
plt.title('Relatório de Classificação - XGBoost + SMOTE', fontsize=14, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('relatorio_classificacao_xgboost.png', dpi=300)
plt.close()
print("   -> Imagem 'relatorio_classificacao_xgboost.png' salva!")

print("\n✅ Gráficos gerados! Seu modelo em produção está pronto para documentação.")