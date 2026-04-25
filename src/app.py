from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import threading
import json
import time
import warnings
from kafka import KafkaConsumer
from flask import Response

# Librerías para Prometheus
from prometheus_client import start_http_server, Counter, Gauge, generate_latest

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, template_folder='../templates')

# 1. MÉTRICAS DE PROMETHEUS
# Counter: Incrementa (útil para total de predicciones)
PREDICCIONES_TOTALES = Counter('predicciones_totales', 'Número total de predicciones realizadas por el modelo')
# Gauge: Sube y baja (útil para valores instantáneos como la potencia)
POTENCIA_ACTUAL = Gauge('potencia_eolica_kw', 'Última potencia predicha en kW')
VIENTO_ACTUAL = Gauge('velocidad_viento_ms', 'Última velocidad de viento recibida')

# 2. CARGA DEL MODELO
MODEL_PATH = '/app/data/modelo_random_forest.joblib'
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error cargando el modelo: {e}")

# Variable global para guardar el último dato recibido por Kafka
ultimo_dato_kafka = {"viento": 0, "curva": 0, "direccion": 0, "prediccion": 0}

# 3. FUNCIÓN DEL CONSUMIDOR KAFKA
def kafka_consumer_thread():
    global ultimo_dato_kafka
    print("Hilo de Kafka iniciado...")
    
    # Obtener el broker desde variable de entorno o usar default (local cluster)
    kafka_broker = os.environ.get('KAFKA_BROKER', 'kafka-service:9092')
    
    consumer = None
    while consumer is None:
        try:
            consumer = KafkaConsumer(
                'datos_eolicos',
                bootstrap_servers=[kafka_broker],
                auto_offset_reset='earliest',  # Cambiar a 'earliest' para consumir desde el principio
                enable_auto_commit=True,
                group_id='eolica-group-v2',  # Cambiar group_id para evitar metadatos anteriores
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=1000
            )
            print(f"Conectado exitosamente a Kafka en {kafka_broker}")
        except Exception as e:
            print(f"Esperando a Kafka ({kafka_broker})... reintentando en 5s. Error: {e}")
            time.sleep(5)

    while True:
        try:
            msg_pack = consumer.poll(timeout_ms=500)
            
            for tp, messages in msg_pack.items():
                for message in messages:
                    data = message.value
                    entrada = np.array([[data['viento'], data['curva'], data['direccion']]])
                    prediccion = model.predict(entrada)[0]     # Predicciòn del modelo para el dato recibido
                    
                    # --- ACTUALIZACIÓN DE MÉTRICAS ---
                    PREDICCIONES_TOTALES.inc() # Incrementa contador
                    POTENCIA_ACTUAL.set(float(prediccion)) # Actualiza valor actual
                    VIENTO_ACTUAL.set(float(data['viento'])) # Actualiza viento actual
                    
                    ultimo_dato_kafka = {
                        "viento": float(data['viento']),
                        "curva": float(data['curva']),
                        "direccion": float(data['direccion']),
                        "prediccion": round(float(prediccion), 2)
                    }
                    print(f"Nuevo dato: {ultimo_dato_kafka['prediccion']} kW")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error en el bucle de consumo: {e}")
            time.sleep(2)

# 4. RUTAS DE FLASK
@app.route('/')
def index():
    return render_template('index.html', data_streaming=ultimo_dato_kafka)

@app.route('/api/streaming')
def get_streaming_data():
    return jsonify(ultimo_dato_kafka)

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        viento = float(request.form['viento'])
        curva = float(request.form['curva'])
        direccion = float(request.form['direccion'])
        datos_entrada = np.array([[viento, curva, direccion]])
        prediccion = model.predict(datos_entrada)[0]
        
        # También registramos predicciones manuales en las métricas
        PREDICCIONES_TOTALES.inc()
        POTENCIA_ACTUAL.set(float(prediccion))
        
        return render_template('index.html', 
                               prediction=round(float(prediccion), 2),
                               viento=viento)
    except Exception as e:
        return f"Error en la predicción manual: {e}", 400

if __name__ == "__main__":
    # Iniciamos el hilo de Kafka
    hilo = threading.Thread(target=kafka_consumer_thread, daemon=True)
    hilo.start()
    
    # Arrancamos Flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)