import pandas as pd
import os
import joblib
import warnings
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# ====== KONFIGURASI PATH (DISESUAIKAN UNTUK MLPROJECT) ======
# Path data kini RELATIF dari direktori MLProject/
DATA_DIR = "dataset_preprosessing" 
TRAIN_PATH = os.path.join(DATA_DIR, "data_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "data_test.csv")

# MLflow akan menyimpan model sebagai ARTIFAK di MLflow Tracking, 
# jadi tidak perlu menyimpan file model secara manual ke MODEL_DIR
MODEL_NAME = "model_best_logistic_regression.pkl"
SCALER_NAME = "scaler.pkl"
# ============================================================

# ====== KONFIGURASI MLFLOW ======
# MLflow akan melacak run di folder 'mlruns' lokal / atau server tracking
MLFLOW_TRACKING_URI = "file:./mlruns" # Tetap menggunakan lokal untuk CI di GitHub
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("CI_Diabetes_Prediction_Tuning")
# ================================

def load_and_scale_data():
    """Memuat data training dan testing serta melakukan scaling."""
    print("Loading dataset...")
    try:
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        print("Dataset training dan testing berhasil dimuat.")
    except FileNotFoundError as e:
        print(f"ERROR: File tidak ditemukan. Detail: {e}")
        exit()

    X_train = train.drop("Outcome", axis=1)
    y_train = train["Outcome"]
    X_test = test.drop("Outcome", axis=1)
    y_test = test["Outcome"]

    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

def tune_and_log_model(X_train_scaled, y_train, X_test_scaled, y_test, scaler):
    """Melakukan hyperparameter tuning dan logging ke MLflow."""

    params = {
        "C": [0.01, 0.1, 1, 10, 100], 
        "solver": ["liblinear", "lbfgs"], 
        "max_iter": [100, 300, 500],
        "class_weight": [None, "balanced"]
    }

    model = LogisticRegression(random_state=42)

    print("Running GridSearchCV (Hyperparameter Tuning)...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=3, 
        scoring="accuracy",
        verbose=1, 
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    # Hitung metrik lengkap
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # --- MANUAL LOGGING DENGAN MLFLOW ---
    # Tambahkan nested=True jika Anda ingin membuat run di dalam run (Kriteria MLflow Project)
with mlflow.start_run(run_name="CI_Tuned_Logistic_Regression", nested=True) as run:
        
        # 1. Logging Parameter Terbaik
        mlflow.log_params(grid.best_params_)
        
        # 2. Logging Metrik
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("cv_best_score", grid.best_score_)
        
        print("\n✅ Hasil Tuning Berhasil Dicatat di MLflow Tracking.")

        # 3. Logging Artefak (Model dan Scaler)
        # Simpan model sebagai artefak ke MLflow. Ini yang akan kita ambil di CI.
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="TunedLogRegModel"
        )
        
        # Simpan scaler
        scaler_temp_path = "scaler.pkl"
        joblib.dump(scaler, scaler_temp_path)
        mlflow.log_artifact(scaler_temp_path)
        os.remove(scaler_temp_path) # Hapus file sementara

        print(f"✅ Model dan scaler telah disimpan di MLflow Run ID: {run.info.run_id}")
        
    print(f"Accuracy pada Data Uji: {acc:.4f}")

# ====== EKSEKUSI UTAMA ======
if __name__ == "__main__":
    X_train_scaled, y_train, X_test_scaled, y_test, scaler = load_and_scale_data()
    tune_and_log_model(X_train_scaled, y_train, X_test_scaled, y_test, scaler)
    print("\nCI Model Training Selesai 🚀")
