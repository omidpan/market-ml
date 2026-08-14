# filename: strategy.py

"""
Trading Strategy Engine

Consumes LSTM predictions from Kafka and
generates trading decisions.

Current version:
- No direct IB order execution yet
- Maintains internal position state
- Prevents repeated entries
- Provides execution-ready order instructions
"""


import json
from enum import Enum

from kafka import KafkaConsumer



# --------------------------------------------------
# Kafka Configuration
# --------------------------------------------------

KAFKA_SERVER = "localhost:9092"

PREDICTION_TOPIC = "ionq_predictions"



# --------------------------------------------------
# Strategy Parameters
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.005


STOP_LOSS_PCT = 0.015

TAKE_PROFIT_PCT = 0.03



BASE_POSITION_SIZE = 100

MAX_POSITION_SIZE = 1000



# --------------------------------------------------
# Position State
# --------------------------------------------------

class Position(Enum):

    FLAT = 0

    LONG = 1

    SHORT = -1



# --------------------------------------------------
# Strategy Engine
# --------------------------------------------------

class StrategyEngine:


    def __init__(self):

        self.position = Position.FLAT

        self.position_size = 0

        self.entry_price = None



    # ----------------------------------------------
    # Position sizing
    # ----------------------------------------------

    def calculate_position_size(
        self,
        predicted_return
    ):


        size = int(

            BASE_POSITION_SIZE *

            (
                abs(predicted_return)
                /
                CONFIDENCE_THRESHOLD
            )

        )


        return min(
            size,
            MAX_POSITION_SIZE
        )



    # ----------------------------------------------
    # Evaluate prediction
    # ----------------------------------------------

    def evaluate(
        self,
        prediction_data
    ):


        predicted_return = prediction_data.get(
            "predicted_return",
            0.0
        )


        current_price = prediction_data.get(
            "current_close",
            0.0
        )



        # ------------------------------------------
        # No trade zone
        # ------------------------------------------

        if abs(predicted_return) < CONFIDENCE_THRESHOLD:


            return {

                "action": "HOLD",

                "reason":
                    "Prediction below confidence threshold",

                "position":
                    self.position.name

            }



        # ------------------------------------------
        # Determine target position
        # ------------------------------------------

        if predicted_return > 0:

            target_position = Position.LONG

        else:

            target_position = Position.SHORT



        # ------------------------------------------
        # Same position
        # ------------------------------------------

        if target_position == self.position:


            return {

                "action": "HOLD",

                "reason":
                    "Already holding same position",

                "position":
                    self.position.name

            }



        # ------------------------------------------
        # New position size
        # ------------------------------------------

        size = self.calculate_position_size(
            predicted_return
        )



        previous_position = self.position



        self.position = target_position

        self.position_size = size

        self.entry_price = current_price



        # ------------------------------------------
        # Determine transition
        # ------------------------------------------

        if previous_position == Position.FLAT:

            action = "OPEN"


        else:

            action = "REVERSE"



        # ------------------------------------------
        # Risk parameters
        # ------------------------------------------

        if self.position == Position.LONG:


            stop_loss = (

                current_price *

                (1 - STOP_LOSS_PCT)

            )


            take_profit = (

                current_price *

                (1 + TAKE_PROFIT_PCT)

            )


            side = "BUY"



        else:


            stop_loss = (

                current_price *

                (1 + STOP_LOSS_PCT)

            )


            take_profit = (

                current_price *

                (1 - TAKE_PROFIT_PCT)

            )


            side = "SELL"




        # ------------------------------------------
        # Execution decision
        # ------------------------------------------

        return {


            "action": action,


            "side": side,


            "quantity": size,


            "entry_price": current_price,


            "stop_loss": stop_loss,


            "take_profit": take_profit,


            "position": self.position.name,


            "previous_position":
                previous_position.name


        }




# --------------------------------------------------
# Kafka Consumer
# --------------------------------------------------

def main():


    strategy = StrategyEngine()



    consumer = KafkaConsumer(PREDICTION_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        auto_offset_reset="latest",
        group_id="trading_strategy_group",
        enable_auto_commit=True,
        value_deserializer=lambda x:
            json.loads(
                x.decode("utf-8")
            )

    )



    print(
        f"Strategy engine listening... Waiting for predictions from Kafka topic: {PREDICTION_TOPIC}"
    )



    for message in consumer:


        prediction = message.value


        decision = strategy.evaluate(
            prediction
        )


        print(
            "=" * 60
        )


        print(
            "STRATEGY DECISION"
        )


        for key, value in decision.items():

            print(
                f"{key}: {value}"
            )



        print(
            "=" * 60
        )




if __name__ == "__main__":

    main()