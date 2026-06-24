import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("🚀 1. Iniciando o Pipeline de Produção...")
df = pd.read_parquet('amostra_estratificada.parquet').dropna(subset=['paragrafo', 'classe_alvo'])

print("🔠 2. Preparando os Tradutores (LabelEncoder)...")
encoder = LabelEncoder()
y = encoder.fit_transform(df['classe_alvo'])
X = df['paragrafo']

print("🔀 3. Separando os dados (Treino/Teste)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🧮 4. Transformando Texto em Matemática (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("⚖️ 5. Aplicando o Antídoto do Desbalanceamento (SMOTE)...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

print("🧠 6. Treinando o Campeão Oficial (XGBoost)...")
clf = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
clf.fit(X_train_res, y_train_res)

print("📊 7. Validação Final de Qualidade...\n")
y_pred = clf.predict(X_test_tfidf)

# Revertendo os números para os nomes das classes para o relatório ficar legível
nomes_classes = encoder.inverse_transform([0, 1, 2])
print("--- RELATÓRIO DE CLASSIFICAÇÃO OFICIAL ---")
print(classification_report(y_test, y_pred, target_names=nomes_classes))

print("💾 8. Exportando a 'Trindade' para o disco...")
joblib.dump(clf, 'xgboost_producao.pkl')
joblib.dump(vectorizer, 'tfidf_producao.pkl')
joblib.dump(encoder, 'encoder_producao.pkl')

print("\n✅ SUCESSO! A Entrega 2 está oficialmente concluída e o modelo está congelado.")