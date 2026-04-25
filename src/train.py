import pandas as pd
import numpy as np
import joblib
import wandb  
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# 1. INICIALIZAR MONITOR (Weights & Biases)
wandb.init(project="prediccion-eolica")

def train():
    print("Iniciando proceso de entrenamiento...")

    # Buscamos la carpeta 'data' en la raíz del proyecto, sin importar dónde estemos
    # Esto funcionará en GitHub Actions, Local y Docker
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'Data_Eolica.csv')
    model_output = os.path.join(base_dir, 'data', 'modelo_random_forest.joblib')
    
    print(f"Buscando datos en: {data_path}") # Esto te ayudará a debuguear en GitHub

    if not os.path.exists(data_path):
        print(f"ERROR: No se encuentra el archivo en {data_path}")
        # Listamos los archivos para ver qué ve GitHub
        print(f"Contenido de la carpeta actual: {os.listdir('.')}")
        return

    # 1. Carga y Limpieza
    df = pd.read_csv(data_path, delimiter=';')

    df['LV ActivePower'] = df['LV ActivePower'].astype(float)
    df['Wind Speed'] = df['Wind Speed'].astype(float)
    df['Theoretical_Power_Curve'] = df['Theoretical_Power_Curve'].astype(float)
    df['Wind Direction'] = df['Wind Direction'].astype(float)

    features = ['Wind Speed', 'Theoretical_Power_Curve', 'Wind Direction'] 
    target = 'LV ActivePower'

    X = df[features]
    y = df[target]

    # 3. DIVISIÓN DE DATOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. CONFIGURACIÓN DEL MODELO Y GRIDSEARCH
    print("Buscando los mejores parámetros...")
    rf = RandomForestRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_leaf': [1, 2]
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_RF = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kfold,
        n_jobs=-1
    )

    grid_RF.fit(X_train, y_train)

    # 5. MEJOR MODELO Y EVALUACIÓN
    best_rf = grid_RF.best_estimator_
    y_pred = best_rf.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    
    # Imprimir métricas para que aparezcan en el log de GitHub Actions
    print(f"--- METRICAS DE ENTRENAMIENTO ---")
    print(f"mse_test: {mse_test}")
    print(f"best_n_estimators: {grid_RF.best_params_['n_estimators']}")
    print(f"best_max_depth: {grid_RF.best_params_['max_depth']}")
    print(f"---------------------------------")

    # 6. REGISTRO EN W&B
    wandb.log({
        "mse_test": mse_test,
        "best_n_estimators": grid_RF.best_params_['n_estimators'],
        "best_max_depth": grid_RF.best_params_['max_depth']
    })

    print(f"Entrenamiento completado. MSE Test: {mse_test}")

    # 7. GUARDAR EL MODELO (Aquí corregí 'model' por 'best_rf')
    joblib.dump(best_rf, model_output)
    print(f"Modelo guardado exitosamente en: {model_output}")

if __name__ == "__main__":
    train()