from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import string
import jsonpickle
import numpy as np
import math


class Product:
    ORCHIDS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.ORCHIDS:{
        "make_edge": 2,
        "make_min_edge": 1,
        "make_probability": 0.566,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size":0.5
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.ORCHIDS: 70}

    def orchids_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1, observation.askPrice + observation.importTariff + observation.transportFees

    def orchids_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict
    ) -> float: 
        if timestamp == 0:
            traderObject["ORCHIDS"]["curr_edge"] = self.params[Product.ORCHIDS]["init_make_edge"]
            return self.params[Product.ORCHIDS]["init_make_edge"]

        traderObject["ORCHIDS"]["volume_history"].append(abs(position))
        if len(traderObject["ORCHIDS"]["volume_history"]) > self.params[Product.ORCHIDS]["volume_avg_timestamp"]:
            traderObject["ORCHIDS"]["volume_history"].pop(0)

        if len(traderObject["ORCHIDS"]["volume_history"]) < self.params[Product.ORCHIDS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["ORCHIDS"]["optimized"]:
            volume_avg = np.mean(traderObject["ORCHIDS"]["volume_history"])

            # Bump up edge if consistently getting lifted full size
            if volume_avg >= self.params[Product.ORCHIDS]["volume_bar"]:
                traderObject["ORCHIDS"]["volume_history"] = [] # clear volume history if edge changed
                traderObject["ORCHIDS"]["curr_edge"] = curr_edge + self.params[Product.ORCHIDS]["step_size"]
                return curr_edge + self.params[Product.ORCHIDS]["step_size"]

            # Decrement edge if more cash with less edge, included discount
            elif self.params[Product.ORCHIDS]["dec_edge_discount"] * self.params[Product.ORCHIDS]["volume_bar"] * (curr_edge - self.params[Product.ORCHIDS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.ORCHIDS]["step_size"] > self.params[Product.ORCHIDS]["min_edge"]:
                    traderObject["ORCHIDS"]["volume_history"] = [] # clear volume history if edge changed
                    traderObject["ORCHIDS"]["curr_edge"] = curr_edge - self.params[Product.ORCHIDS]["step_size"]
                    traderObject["ORCHIDS"]["optimized"] = True
                    return curr_edge - self.params[Product.ORCHIDS]["step_size"]
                else:
                    traderObject["ORCHIDS"]["curr_edge"] = self.params[Product.ORCHIDS]["min_edge"]
                    return self.params[Product.ORCHIDS]["min_edge"]

        traderObject["ORCHIDS"]["curr_edge"] = curr_edge
        return curr_edge

    def orchids_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)                                                                                                                                                                    

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.ORCHIDS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity) # max amount to buy
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity) # max amount to sell
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def orchids_arb_clear(
        self,
        position: int
    ) -> int:
        conversions = -position
        return conversions

    def orchids_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]

        # Implied Bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        # Implied Ask = observation.askPrice + observation.importTariff + observation.transportFees
        implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge

        # ask = foreign_mid - 1.6 best performance so far
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6 # Aggressive ask

        # don't lose money
        if aggressive_ask >= implied_ask + self.params[Product.ORCHIDS]['min_edge']:
            ask = aggressive_ask
            print("AGGRESSIVE")
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")
        else:
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 40]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 25]

        # If we're not best level, penny until min edge
        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and  bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        print(f"IMPLIED_BID: {implied_bid}")
        print(f"IMPLIED_ASK: {implied_ask}")
        print(f"FOREIGN ASK: {observation.askPrice}")
        print(f"FOREIGN BID: {observation.bidPrice}")

        best_bid = min(order_depth.buy_orders.keys())
        best_ask = max(order_depth.sell_orders.keys())

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(ask), -sell_quantity))  # Sell order

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.ORCHIDS in self.params and Product.ORCHIDS in state.order_depths:
            if "ORCHIDS" not in traderObject:
                traderObject["ORCHIDS"] = {"curr_edge": self.params[Product.ORCHIDS]["init_make_edge"], "volume_history": [], "optimized": False}
            orchids_position = (
                state.position[Product.ORCHIDS]
                if Product.ORCHIDS in state.position
                else 0
            )
            print(f"ORCHIDS POSITION: {orchids_position}")

            conversions = self.orchids_arb_clear(
                orchids_position
            )

            adap_edge = self.orchids_adap_edge(
                state.timestamp,
                traderObject["ORCHIDS"]["curr_edge"],
                orchids_position,
                traderObject,
            )

            orchids_position = 0

            orchids_take_orders, buy_order_volume, sell_order_volume = self.orchids_arb_take(
                state.order_depths[Product.ORCHIDS],
                state.observations.conversionObservations[Product.ORCHIDS],
                adap_edge,
                orchids_position,
            )

            orchids_make_orders, _, _ = self.orchids_arb_make(
                state.order_depths[Product.ORCHIDS],
                state.observations.conversionObservations[Product.ORCHIDS],
                orchids_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.ORCHIDS] = (
                orchids_take_orders + orchids_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
