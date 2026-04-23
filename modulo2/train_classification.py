import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv

def main():
    # 1. Carregar dados
    df = pd.read_csv('data/processed_lisboa_porto_air_quality.csv', sep=';')
    
    # 2. Pré-processamento
    # Assumindo que queremos prever 'air_quality_good' usando as restantes numéricas
    X = df.drop(columns=['air_quality_good', 'datetime', 'city'], errors='ignore')
    y = df['air_quality_good']
    # Add this after defining 'y'
    print("Class distribution in 'y':")
    print(y.value_counts())
    
    # Imputação de valores nulos (estratégia: mediana)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Split Treino/Teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Treinar Modelos
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random_Forest_Clf': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    metrics_list = []
    
    for name, model in models.items():
        # Treino
        model.fit(X_train_scaled, y_train)
        
        # Previsão
        y_pred = model.predict(X_test_scaled)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics_list.append({
            'Model': name,
            'Task': 'Classification',
            'Target': 'air_quality_good',
            'Accuracy_R2': acc, # Usamos a mesma coluna no CSV para Acc(clf) e R2(reg)
            'Precision_MAE': prec,
            'Recall_RMSE': rec,
            'F1_Score': f1
        })
        
        # Guardar o modelo em .pkl
        joblib.dump(model, f"{name}.pkl")
        print(f"[{name}] Accuracy: {acc:.4f} | F1: {f1:.4f} - Modelo guardado.")

    # Guardar Scaler e Imputer para uso futuro
    joblib.dump(scaler, "scaler_clf.pkl")
    joblib.dump(imputer, "imputer_clf.pkl")
    
    # Escrever métricas parciais num CSV temporário (ou enviar para o final)
    pd.DataFrame(metrics_list).to_csv('metrics_clf.csv', index=False)
    print("Métricas de classificação guardadas em 'metrics_clf.csv'.")

    loaded_model = joblib.load('logistic_regression_model.pkl')
    loaded_scaler = joblib.load('scaler_clf.pkl')
    
if __name__ == "__main__":
    main()