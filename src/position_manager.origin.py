# filename: position_manager.py

"""
Position Manager

Responsible for:
- Maintaining trading position state
- Recovering state after restart
- Tracking broker positions
- Synchronizing strategy and IB

Flow:

IB Gateway
     |
     v
Position Manager
     |
     v
Strategy Engine


Current version:
- JSON persistence
- Single symbol management
- Ready for database replacement
"""


import json
import os
from datetime import datetime



POSITION_FILE = "position_state.json"





class PositionManager:



    def __init__(self, symbol):


        self.symbol = symbol


        self.position = {

            "symbol": symbol,

            "side": "FLAT",

            "quantity": 0,

            "average_price": 0.0,

            "entry_time": None,

            "last_action": None

        }


        self.load_state()




    # ------------------------------------------------
    # Load previous state
    # ------------------------------------------------

    def load_state(self):


        if not os.path.exists(
            POSITION_FILE
        ):

            return



        with open(
            POSITION_FILE,
            "r"
        ) as file:


            saved = json.load(
                file
            )



        if saved.get(
            "symbol"
        ) == self.symbol:


            self.position = saved



        print(

            "Position restored:",

            self.position

        )





    # ------------------------------------------------
    # Save state
    # ------------------------------------------------

    def save_state(self):


        with open(
            POSITION_FILE,
            "w"
        ) as file:


            json.dump(

                self.position,

                file,

                indent=4

            )





    # ------------------------------------------------
    # Current position
    # ------------------------------------------------

    def get_position(self):


        return self.position





    # ------------------------------------------------
    # Check if holding
    # ------------------------------------------------

    def is_flat(self):


        return (

            self.position["quantity"] == 0

        )





    def is_long(self):


        return (

            self.position["side"]
            ==
            "LONG"

        )





    def is_short(self):


        return (

            self.position["side"]
            ==
            "SHORT"

        )





    # ------------------------------------------------
    # Open position
    # ------------------------------------------------

    def open_position(

        self,

        side,

        quantity,

        price

    ):


        self.position = {


            "symbol":
                self.symbol,


            "side":
                side,


            "quantity":
                quantity,


            "average_price":
                price,


            "entry_time":
                datetime.utcnow()
                .isoformat(),


            "last_action":
                "OPEN"


        }



        self.save_state()



    # ------------------------------------------------
    # Close position
    # ------------------------------------------------

    def close_position(self):


        self.position = {


            "symbol":
                self.symbol,


            "side":
                "FLAT",


            "quantity":
                0,


            "average_price":
                0.0,


            "entry_time":
                None,


            "last_action":
                "CLOSE"


        }



        self.save_state()




    # ------------------------------------------------
    # Reverse position
    # ------------------------------------------------

    def reverse_position(

        self,

        new_side,

        quantity,

        price

    ):


        self.position = {


            "symbol":
                self.symbol,


            "side":
                new_side,


            "quantity":
                quantity,


            "average_price":
                price,


            "entry_time":
                datetime.utcnow()
                .isoformat(),


            "last_action":
                "REVERSE"


        }



        self.save_state()





    # ------------------------------------------------
    # Synchronize with IB
    # ------------------------------------------------

    def update_from_ib(

        self,

        ib_position

    ):


        """
        Example input:

        {
            "symbol":"IONQ",
            "position":100,
            "avgCost":31.25
        }

        """



        qty = ib_position.get(
            "position",
            0
        )


        avg = ib_position.get(
            "avgCost",
            0
        )



        if qty > 0:


            side = "LONG"



        elif qty < 0:


            side = "SHORT"



        else:


            side = "FLAT"




        self.position = {


            "symbol":
                self.symbol,


            "side":
                side,


            "quantity":
                abs(qty),


            "average_price":
                avg,


            "entry_time":
                self.position.get(
                    "entry_time"
                ),


            "last_action":
                "IB_SYNC"


        }



        self.save_state()





    # ------------------------------------------------
    # Compare strategy signal
    # ------------------------------------------------

    def needs_action(

        self,

        target_side

    ):


        current = self.position["side"]



        if current == target_side:


            return False



        return True





    # ------------------------------------------------
    # Debug
    # ------------------------------------------------

    def print_status(self):


        print(
            """
------------------------
POSITION STATUS
------------------------
"""
        )


        for k,v in self.position.items():

            print(
                f"{k}: {v}"
            )


        print(
            "------------------------"
        )






# ------------------------------------------------
# Test
# ------------------------------------------------

if __name__ == "__main__":


    manager = PositionManager(
        "IONQ"
    )


    manager.print_status()



    manager.open_position(

        side="LONG",

        quantity=100,

        price=31.50

    )



    manager.print_status()