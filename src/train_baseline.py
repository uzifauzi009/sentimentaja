import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os  

# ==========================================
# 1. LOAD DATA
# ==========================================
print("📁 Loading data...")
train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')

# --- TAMBAHKAN KODE INI BUAT BERSIHIN DATA KOSONG (NaN) ---
train = train.dropna(subset=['text'])
val = val.dropna(subset=['text'])
# ----------------------------------------------------------

X_train, y_train = train['text'], train['label']
X_val, y_val = val['text'], val['label']

print(f"Train: {len(X_train)} | Val: {len(X_val)}")
# ============================================
# 2. PIPELINE
# ============================================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# ============================================
# 3. GRID SEARCH
# ============================================
param_grid = {
    'tfidf__max_features': [3000, 5000, 7000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10]
}

print("🔍 Grid searching...")
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

# ============================================
# 4. MLflow TRACKING
# ============================================
mlflow.set_experiment("sentiment-baseline")

with mlflow.start_run(run_name="logistic-regression-tfidf"):
    # Training
    grid.fit(X_train, y_train)
    
    # Best model
    best_model = grid.best_estimator_
    
    # Prediksi
    y_pred = best_model.predict(X_val)
    
    # Metrics
    f1 = f1_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"\n✅ Val F1: {f1:.4f}")
    print(f"✅ Val Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Negatif', 'Positif']))
    
    # Log ke MLflow
    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({
        "val_f1": f1,
        "val_accuracy": acc
    })
    mlflow.sklearn.log_model(best_model, "model")
    
    # Simpan model lokal
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/baseline_model.pkl')
    print("\n💾 Model saved to models/baseline_model.pkl")
    print(f"📁 Run ID: {mlflow.active_run().info.run_id}")

print("\n🎉 Done!")