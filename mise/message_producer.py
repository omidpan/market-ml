import json
import time
from kafka import KafkaProducer

# Configuration - update with your actual server configuration address if needed
KAFKA_SERVER = 'localhost:9092'
TOPIC_NAME = 'ionq_raw_candles'

# Initialize the producer with built-in JSON mapping/serialization
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all'
)

# Your target stock candle dataset with valid epoch integers
candle_data = [
    {"datetime": 1785384726, "open": 193.64, "high": 195.67, "low": 192.64, "close": 194.67, "volume":40.0}
]

print(f"Sending {len(candle_data)} candles to topic '{TOPIC_NAME}'...")

for candle in candle_data:
    # Send each payload straight into the broker
    producer.send(TOPIC_NAME, value=candle)
    print(f"Sent: {candle}")

# Flush ensures everything currently waiting in memory is pushed before closing
producer.flush()
print("All messages successfully transmitted.")
