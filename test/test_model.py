import pytest
import joblib
import numpy as np
import os

# Definimos la ruta del modelo (ajustada a la estructura de tu repo)
MODEL_PATH = 'data/modelo_random_forest.joblib'

def test_model_exists():
    """Verifica que el archivo del modelo exista"""
    assert os.path.exists(MODEL_PATH), f"No se encontró el modelo en {MODEL_PATH}"

def test_model_loading():
    """Verifica que el modelo .joblib cargue correctamente"""
    model = joblib.load(MODEL_PATH)
    assert model is not None, "El modelo cargado es None"
    # Verificamos que sea un RandomForest (o el estimador que uses)
    assert hasattr(model, 'predict'), "El objeto cargado no tiene el método predict"

def test_model_prediction():
    """Verifica que la función de predicción devuelva un número y no un error"""
    model = joblib.load(MODEL_PATH)
    # Creamos un dato de prueba: [Wind Speed, Theoretical_Power_Curve, Wind Direction]
    entrada_ejemplo = np.array([[12.5, 1500.0, 200.0]])
    
    try:
        prediccion = model.predict(entrada_ejemplo)
        # Verificamos que el resultado sea un número (float o int)
        assert isinstance(float(prediccion[0]), float), "La predicción no es un valor numérico"
    except Exception as e:
        pytest.fail(f"La predicción falló con el error: {e}")