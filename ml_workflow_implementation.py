import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. SIMULASI DATA COLLECTION & PREPROCESSING
def prepare_data():
    print("[1/4] Mengumpulkan dan Memproses Data (Data Preprocessing)...")
    np.random.seed(42)
    
    # Simulasi dataset fitur (misal: jam belajar, skor sebelumnya, kehadiran)
    X = np.random.rand(200, 3) * [10, 100, 100] 
    
    # Simulasi label target (0 = Tidak Lulus/Gagal, 1 = Lulus) berdasarkan aturan sederhana
    y = np.array([1 if (row[0]*0.4 + row[1]*0.4 + row[2]*0.2) > 50 else 0 for row in X])
    
    # Normalisasi data numerik (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y


# 2. DATA SPLITTING (Training & Testing Set)

def split_dataset(X, y):
    print("[2/4] Membagi Dataset menjadi Training (80%) dan Testing (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 3. MODEL TRAINING (Supervised Learning - Klasifikasi)

def train_model(X_train, y_train):
    print("[3/4] Melatih Model Machine Learning (Model Training)...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. MODEL EVALUATION
def evaluate_model(model, X_test, y_test):
    print("[4/4] Mengevaluasi Performa Model (Model Evaluation)...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- HASIL EVALUASI MODEL ---")
    print(f"Akurasi Model: {acc * 100:.2f}%")
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    print(" MEMULAI ALUR KERJA MACHINE LEARNING ")
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    trained_model = train_model(X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)
    print(" ALUR KERJA MACHINE LEARNING SELESAI ")