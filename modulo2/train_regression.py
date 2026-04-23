import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    # 1. Carregar dados
    df = pd.read_csv('data/processed_lisboa_porto_air_quality_filtered.csv')
    
    # 2. Pré-processamento
    # Target: no2_concentration (assumindo este nome)
    # Removemos a label de classificação para não causar data leakage
    X = df.drop(columns=['NO2', 'air_quality_good', 'datetime', 'city'], errors='ignore')
    y = df['NO2']
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Split Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Treinar Modelos
    models = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest_Reg': RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    metrics_list = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Métricas
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics_list.append({
            'Model': name,
            'Task': 'Regression',
            'Target': 'NO2_concentration',
            'Accuracy_R2': r2, 
            'Precision_MAE': mae,
            'Recall_RMSE': rmse,
            'F1_Score': np.nan # N/A para regressão
        })
        
        joblib.dump(model, f"{name}.pkl")
        print(f"[{name}] R2: {r2:.4f} | MAE: {mae:.4f} - Modelo guardado.")

    joblib.dump(scaler, "scaler_reg.pkl")
    joblib.dump(imputer, "imputer_reg.pkl")
    
    # Juntar com as métricas de classificação
    try:
        df_clf = pd.read_csv('metrics_clf.csv')
        df_final = pd.concat([df_clf, pd.DataFrame(metrics_list)], ignore_index=True)
    except FileNotFoundError:
        df_final = pd.DataFrame(metrics_list)
        
    # Salvar ficheiro final
    df_final.to_csv('metrics.csv', index=False)
    print("Ficheiro final 'metrics.csv' gerado com sucesso!")

if __name__ == "__main__":
    main()