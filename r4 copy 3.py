import json
import numpy as np
from math import sqrt, log
from statistics import NormalDist
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any
from copy import deepcopy
from datamodel import Order, TradingState, ConversionObservation
import math
import random


class Macaron:
    def __init__(self, symbol, limit, conversion_limit):
        self.symbol = symbol
        self.limit = limit
        self.conversion_limit = conversion_limit

    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        obs = state.observations.conversionObservations[self.symbol]

        far_bid = obs.bidPrice - obs.exportTariff - obs.transportFees
        far_ask = obs.askPrice + obs.importTariff + obs.transportFees
        orders = []

        order_depth = state.order_depths[self.symbol]
        buy_price, buy_volume = sorted(order_depth.sell_orders.items())[0]
        sell_price, sell_volume = sorted(order_depth.buy_orders.items(), reverse=True)[0]

        export_profit = far_bid - buy_price
        import_profit = sell_price - far_ask
        
        position = state.position.get(self.symbol, 0)
        conversions = -position

        if export_profit > 0:
            volume = min(abs(buy_volume), self.conversion_limit)
            orders.append(Order(self.symbol, buy_price, volume))
            conversions = -(position + volume)
        elif import_profit > 0:
            volume = min(abs(sell_volume), self.conversion_limit)
            orders.append(Order(self.symbol, sell_price, -volume))
            conversions = -(position - volume)

        print(f"Orders: {orders}")
        print(f"Conversions: {conversions}")
        print(f"Export_profit: {export_profit}")
        print(f"Import_profit: {import_profit}")

        return orders, conversions


class Trader:
    def __init__(self):
        self.print_log = False
        self.macaron_symbol = ["MAGNIFICENT_MACARONS"]
        self.macaron_limit = 70
        self.macaron_conversion_limit = 10

    def macaron(self, state: TradingState, symbol: str) -> Tuple[List[Order], int]:
        macaron_trader = Macaron(symbol, self.macaron_limit, self.macaron_conversion_limit)
        orders, conversions = macaron_trader.run(state)
        return orders, conversions

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        print(f"[{state.timestamp}] POSITIONS → " + ", ".join(
            f"{symbol}: {state.position.get(symbol, 0)}"
            for symbol in self.macaron_symbol
        ))

        all_orders: Dict[str, List[Order]] = {}
        total_conversions = 0

        for symbol in self.macaron_symbol:
            orders, conversions = self.macaron(state, symbol)
            all_orders[symbol] = orders
            total_conversions += conversions

        return all_orders, total_conversions, ""
