import math
import json
import copy
import random
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple
from datamodel import (Order, OrderDepth, TradingState, Listing, Observation, ProsperityEncoder, Symbol, Trade)


def get_mid_price(state: TradingState, symbol: str, method: str, min_vol: int) -> float:
    order_depth = state.order_depths[symbol]

    total_ask = 0
    quantity_ask = 0
    total_bid = 0
    quantity_bid = 0

    for price, quantity in order_depth.sell_orders.items():
        total_ask -= price * quantity
        quantity_ask -= quantity
    for price, quantity in order_depth.buy_orders.items():
        total_bid += price * quantity
        quantity_bid += quantity

    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0.0

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    best_bid_quantity = order_depth.buy_orders[best_bid]
    best_ask_quantity = -order_depth.sell_orders[best_ask]

    popular_bid = max(order_depth.buy_orders.items(), key=lambda tup: tup[1])[0] if order_depth.buy_orders else best_bid
    popular_ask = min(order_depth.sell_orders.items(), key=lambda tup: tup[1])[0] if order_depth.sell_orders else best_ask
    popular_price = (popular_bid + popular_ask) / 2

    filtered_best_ask = min([price for price in order_depth.sell_orders if -order_depth.sell_orders[price] >= min_vol], default=0)
    filtered_best_bid = max([price for price in order_depth.buy_orders if order_depth.buy_orders[price] >= min_vol], default=0)

    mid_price = (best_bid + best_ask) / 2
    weighted_average = (total_ask + total_bid) / (quantity_ask + quantity_bid)
    weighted_mid_price = (best_bid_quantity * best_bid + best_ask_quantity * best_ask) / (best_ask_quantity + best_bid_quantity)

    if filtered_best_ask != 0 and filtered_best_bid != 0:
        filtered_mid_price = (filtered_best_ask + filtered_best_bid) / 2
    else:
        filtered_mid_price = mid_price

    if method == 'mid_price':
        return mid_price
    elif method == 'popular_price':
        return popular_price
    elif method == 'filtered_mid_price':
        return filtered_mid_price
    elif method == 'weighted_mid_price':
        return weighted_mid_price
    elif method == 'weighted_average':
        return weighted_average
    elif method == 'ensemble':
        return (weighted_average + filtered_mid_price) / 2
    else:
        raise ValueError(f"Unknown method: {method}")


def aggressive_take_orders(symbol, state, true_value, remaining_buy, remaining_sell, take_width, prevent_adverse=False, adverse_volume=0):
    order_depth = state.order_depths[symbol]
    orders = []

    buy_price_take = math.floor(true_value - take_width)
    sell_price_take = math.ceil(true_value + take_width)

    # if symbol == "SQUID_INK":
    #     sell_prices = sorted(order_depth.sell_orders.keys())
    #     if len(sell_prices) >= 2:
    #         second_ask_price = sell_prices[1]
    #         second_ask_volume = -order_depth.sell_orders[second_ask_price]
    #         if second_ask_volume > 10:
    #             for price, volume in order_depth.buy_orders.items():
    #                 orders.append(Order(symbol, price, -volume))
    #     buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
    #     if len(buy_prices) >= 2:
    #         second_bid_price = buy_prices[1]
    #         second_bid_volume = order_depth.buy_orders[second_bid_price]
    #         if second_bid_volume > 10:
    #             for price, volume in order_depth.sell_orders.items():
    #                 orders.append(Order(symbol, price, -volume))

    for price, volume in sorted(order_depth.sell_orders.items()):
        if remaining_buy > 0 and price <= buy_price_take and (not prevent_adverse or abs(volume) <= adverse_volume):
            quantity = min(remaining_buy, -volume)
            orders.append(Order(symbol, price, quantity))
            remaining_buy -= quantity

    for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
        if remaining_sell > 0 and price >= sell_price_take and (not prevent_adverse or volume <= adverse_volume):
            quantity = min(remaining_sell, volume)
            orders.append(Order(symbol, price, -quantity))
            remaining_sell -= quantity

    return orders, remaining_buy, remaining_sell


def clear_position_liquidation(symbol, state, true_value, position, remaining_buy, remaining_sell, liquidate_width):
    orders = []
    order_depth = state.order_depths[symbol]
    to_clear = position - remaining_buy + remaining_sell
    buy_price_liquidate = round(true_value - liquidate_width)
    sell_price_liquidate = round(true_value + liquidate_width)
    if to_clear > 0:
        clear_quantity = sum(
            volume for price, volume in order_depth.buy_orders.items()
            if price >= sell_price_liquidate
        )
        sent = min(to_clear, clear_quantity)
        if sent > 0:
            orders.append(Order(symbol, sell_price_liquidate, -sent))
            remaining_sell -= sent
    elif to_clear < 0:
        clear_quantity = sum(
            -volume for price, volume in order_depth.sell_orders.items()
            if price <= buy_price_liquidate
        )
        sent = min(-to_clear, clear_quantity)
        if sent > 0:
            orders.append(Order(symbol, buy_price_liquidate, sent))
            remaining_buy -= sent
    return orders, remaining_buy, remaining_sell


def default_market_make(symbol, state, true_value, remaining_buy, remaining_sell, market_make_spread, price_filter_width, fallback_offset):
    orders = []
    order_depth = state.order_depths[symbol]
    sell_prices = [price for price in order_depth.sell_orders if price > true_value + price_filter_width]
    buy_prices = [price for price in order_depth.buy_orders if price < true_value - price_filter_width]

    sell_base = min(sell_prices) if sell_prices else true_value + fallback_offset
    buy_base = max(buy_prices) if buy_prices else true_value - fallback_offset

    buy_price_make = round(buy_base - market_make_spread)
    sell_price_make = round(sell_base + market_make_spread)

    if symbol == 'SQUID_INK':
        if len(sell_prices) >= 2:
            second_ask_price = sell_prices[1]
            second_ask_volume = -order_depth.sell_orders[second_ask_price]
        else:
            second_ask_volume = 0
        if len(buy_prices) >= 2:
            second_bid_price = buy_prices_prices[1]
            second_bid_volume = order_depth.buy_orders[second_bid_price]
        else:
            second_bid_volume = 0
        if second_ask_volume>15 or second_bid_volume>15:
            buy_price_make = buy_price_make - 2
            sell_price_make = sell_price_make + 2

    if remaining_buy > 0:
        orders.append(Order(symbol, buy_price_make, remaining_buy))
    if remaining_sell > 0:
        orders.append(Order(symbol, sell_price_make, -remaining_sell))
    return orders


class MarketMakingStrategy:
    def __init__(self, symbol, limit, print_log=0, take_order_fn=None, take_order_args=None, liquidation_fn=None, liquidation_args=None, market_make_fn=None, market_make_args=None, true_value_args=None):
        self.symbol = symbol
        self.limit = limit
        self.print_log = print_log
        self.take_order_fn = take_order_fn
        self.take_order_args = take_order_args or {}
        self.liquidation_fn = liquidation_fn
        self.liquidation_args = liquidation_args or {}
        self.market_make_fn = market_make_fn
        self.market_make_args = market_make_args or {}
        self.true_value_args = true_value_args or {}
        self.remaining_buy = 0
        self.remaining_sell = 0
        self.orders = []
        self.conversions = 0

    def run(self, state):
        self.orders = []
        self.conversions = 0
        true_value = self.get_true_value(state)
        self.act(state, true_value)
        return self.orders, self.conversions

    def act(self, state, true_value):
        position = state.position.get(self.symbol, 0)
        self.remaining_buy = self.limit - position
        self.remaining_sell = self.limit + position

        if self.take_order_fn:
            orders, self.remaining_buy, self.remaining_sell = self.take_order_fn(
                self.symbol, state, true_value, self.remaining_buy, self.remaining_sell, **self.take_order_args
            )
            self.orders.extend(orders)

        if self.liquidation_fn:
            orders, self.remaining_buy, self.remaining_sell = self.liquidation_fn(
                self.symbol, state, true_value, position, self.remaining_buy, self.remaining_sell, **self.liquidation_args
            )
            self.orders.extend(orders)

        if self.market_make_fn:
            orders = self.market_make_fn(
                self.symbol, state, true_value, self.remaining_buy, self.remaining_sell, **self.market_make_args
            )
            self.orders.extend(orders)

    def get_true_value(self, state):
        if self.true_value_args.get("use_fixed", False):
            return self.true_value_args.get("fixed_value", 0)
        return get_mid_price(state, self.symbol, method="weighted_average", min_vol=15)


class Trader:
    def __init__(self):
        self.print_log = 1
        self.symbols = ["KELP", "RAINFOREST_RESIN", "SQUID_INK"]
        self.strategy_config = {
            "KELP": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": True, "adverse_volume": 15, "take_width": 1},
                "liquidation_args": {"liquidate_width": 1},
                "market_make_args": {"market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 0.5},
                "true_value_args": {"use_fixed": False}
            },
            "RAINFOREST_RESIN": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": False, "adverse_volume": 0, "take_width": 1},
                "liquidation_args": {"liquidate_width": 1},
                "market_make_args": {"market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 0.5},
                "true_value_args": {"use_fixed": True, "fixed_value": 10000}
            },
            "SQUID_INK": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": False, "adverse_volume": 0, "take_width": 1},
                "liquidation_args": {"liquidate_width": 1},
                "market_make_args": {"market_make_spread": 1, "price_filter_width": 1, "fallback_offset": 2},
                "true_value_args": {"use_fixed": False}
            }
        }
        self.cashs = defaultdict(float)
        self.pnl = defaultdict(float)
        self.pnl_history = defaultdict(lambda: deque(maxlen=20))
        self.last_timestamp = -100

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        all_orders = {}
        conversions = 0

        for symbol in self.symbols:
            for trade in state.own_trades.get(symbol, []):
                if trade.timestamp == self.last_timestamp:
                    if trade.buyer == "SUBMISSION":
                        self.cashs[symbol] -= trade.price * trade.quantity
                    elif trade.seller == "SUBMISSION":
                        self.cashs[symbol] += trade.price * trade.quantity

        for symbol in self.symbols:
            cfg = self.strategy_config[symbol]
            strat = MarketMakingStrategy(
                symbol=symbol,
                limit=cfg["limit"],
                print_log=self.print_log,
                take_order_fn=aggressive_take_orders,
                take_order_args=cfg["take_order_args"],
                liquidation_fn=clear_position_liquidation,
                liquidation_args=cfg["liquidation_args"],
                market_make_fn=default_market_make,
                market_make_args=cfg["market_make_args"],
                true_value_args=cfg["true_value_args"]
            )

            orders, conv = strat.run(state)
            tv = strat.get_true_value(state)
            pos = state.position.get(symbol, 0)
            pnl_now = self.cashs[symbol] + pos * tv
            self.pnl[symbol] = pnl_now
            self.pnl_history[symbol].append(pnl_now)

            all_orders[symbol] = orders
            conversions += conv

            if self.print_log:
                print(f"[{symbol}] PnL: {pnl_now:.2f}, Cash: {self.cashs[symbol]:.2f}, Pos: {pos}, TV: {tv:.2f}")

        self.last_timestamp = state.timestamp
        return all_orders, conversions, "{}"