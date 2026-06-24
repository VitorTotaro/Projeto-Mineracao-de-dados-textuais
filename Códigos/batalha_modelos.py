import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Balanceadores
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

print("📂 1. Carregando os dados...")
df = pd.read_parquet('amostra_estratificada.parquet').dropna(subset=['paragrafo', 'classe_alvo'])

# O XGBoost exige que o alvo seja numérico (0, 1, 2)
encoder = LabelEncoder()
df['target_num'] = encoder.fit_transform(df['classe_alvo'])
mapa_classes = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

X = df['paragrafo']
y = df['target_num']

print("🔀 2. Separando Treino e Teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🧮 3. Aplicando TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Definindo os competidores
balanceadores = {
    "Nenhum (Baseline)": None,
    "Undersampling": RandomUnderSampler(random_state=42),
    "SMOTE (Oversampling)": SMOTE(random_state=42),
}

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
}

print("\n⚔️ 4. INICIANDO A BATALHA DE MODELOS E BALANCEAMENTOS...\n")

resultados = []

for nome_bal, balanceador in balanceadores.items():
    print(f"--- Aplicando Balanceamento: {nome_bal} ---")
    
    if balanceador is None:
        X_res, y_res = X_train_tfidf, y_train
    else:
        # Algoritmos de balanceamento demoram um pouco, paciência aqui
        X_res, y_res = balanceador.fit_resample(X_train_tfidf, y_train)
        
    for nome_mod, modelo in modelos.items():
        print(f"Treinando {nome_mod}...")
        modelo.fit(X_res, y_res)
        
        y_pred = modelo.predict(X_test_tfidf)
        relatorio = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Coletando o Recall específico da classe Aditivo (a mais difícil)
        # O código descobre qual é o número correspondente ao 'Aditivo'
        idx_aditivo = str(encoder.transform(['Aditivo'])[0])
        recall_aditivo = relatorio[idx_aditivo]['recall']
        
        resultados.append({
            'Balanceamento': nome_bal,
            'Modelo': nome_mod,
            'Acurácia Global': relatorio['accuracy'],
            'Recall Aditivo': recall_aditivo,
            'F1-Score Licitação': relatorio[str(encoder.transform(['Licitação'])[0])]['f1-score']
        })

print("\n🏆 --- RESULTADO DA ARENA ---")
df_resultados = pd.DataFrame(resultados).sort_values(by='Recall Aditivo', ascending=False)
print(df_resultados.to_string(index=False))