from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'], # Puerto externo
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
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