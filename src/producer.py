from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers=['127.0.0.1:9092'], # Usa la IP local explícita
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    api_version=(2, 0, 2), # Fuerza la versión de la API para evitar el autodiscovery
    request_timeout_ms=10000
)

print("Productor iniciado. Enviando datos de turbina...")

try:
    while True:
        # Simulamos datos reales
        data = {
            "viento": round(random.uniform(5, 25), 2),
            "curva": round(random.uniform(1500, 3500), 2),
            "direccion": round(random.uniform(0, 360), 2)
        }
        
        producer.send('datos_eolicos', data)
        print(f"Enviado: {data}")
        time.sleep(2)
except KeyboardInterrupt:
    print("Producción detenida.")