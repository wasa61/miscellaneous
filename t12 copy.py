import json
import numpy as np
from math import sqrt, log
from statistics import NormalDist
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any
from copy import deepcopy
from datamodel import Order, TradingState
import math
import random


def get_mid_price(state: TradingState, symbol: str, method: str, min_vol=0) -> float:
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
        buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        sell_prices = sorted(order_depth.sell_orders.keys())
        if len(sell_prices) >= 2:
            second_ask_price = sell_prices[1]
            second_ask_volume = -order_depth.sell_orders[second_ask_price]
        else:
            second_ask_volume = 0
        if len(buy_prices) >= 2:
            second_bid_price = buy_prices[1]
            second_bid_volume = order_depth.buy_orders[second_bid_price]
        else:
            second_bid_volume = 0
        if second_ask_volume > 15 or second_bid_volume > 15:
            return []

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

        print('startstart') if self.print_log == 1 else None
        print(f"to_buy{self.remaining_buy}...") if self.print_log == 1 else None
        print(f"to_sell{self.remaining_sell}...") if self.print_log == 1 else None
        print(f"position{position}...") if self.print_log == 1 else None

        if self.take_order_fn:
            orders, self.remaining_buy, self.remaining_sell = self.take_order_fn(
                self.symbol, state, true_value, self.remaining_buy, self.remaining_sell, **self.take_order_args
            )
            self.orders.extend(orders)
            if self.print_log == 1:
                for o in orders:
                    print(f"[{self.symbol}] TAKE → {'BUY' if o.quantity > 0 else 'SELL'} {abs(o.quantity)} @ {o.price}") if self.print_log == 1 else None

        if self.liquidation_fn:
            orders, self.remaining_buy, self.remaining_sell = self.liquidation_fn(
                self.symbol, state, true_value, position, self.remaining_buy, self.remaining_sell, **self.liquidation_args
            )
            self.orders.extend(orders)
            if self.print_log == 1:
                for o in orders:
                    print(f"[{self.symbol}] LIQUIDATE → {'BUY' if o.quantity > 0 else 'SELL'} {abs(o.quantity)} @ {o.price}") if self.print_log == 1 else None
        if self.market_make_fn:
            orders = self.market_make_fn(
                self.symbol, state, true_value, self.remaining_buy, self.remaining_sell, **self.market_make_args
            )
            self.orders.extend(orders)
            if self.print_log == 1:
                for o in orders:
                    print(f"[{self.symbol}] MAKE → {'BUY' if o.quantity > 0 else 'SELL'} {abs(o.quantity)} @ {o.price}") if self.print_log == 1 else None
        print('endend') if self.print_log == 1 else None

    def get_true_value(self, state):
        if self.true_value_args.get("use_fixed", False):
            return self.true_value_args.get("fixed_value", 0)
        return get_mid_price(state, self.symbol, method="weighted_average", min_vol=15)


class VolSmileArbTrader:
    def __init__(self, option_strikes: Dict[str, int], limits: Dict[str, int], z_threshold: float, history_window: int, best_only_entry: bool, best_only_exit: bool):
        self.option_strikes = option_strikes
        self.limits = limits
        self.underlying_symbol = "VOLCANIC_ROCK"
        self.outlier_threshold = 0.04
        self.entry_threshold = z_threshold
        self.history_window = history_window
        self.best_only_entry = best_only_entry
        self.best_only_exit = best_only_exit

    def black_scholes_call(self, S, K, T, sigma):
        d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * NormalDist().cdf(d1) - K * NormalDist().cdf(d2)

    def implied_volatility(self, price, S, K, T, tol=1e-8, max_iter=100):
        low, high = 0.01, 2.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            guess = self.black_scholes_call(S, K, T, mid)
            if abs(guess - price) < tol:
                return mid
            if guess > price:
                high = mid
            else:
                low = mid
        return mid

    def delta(self, S, K, T, sigma):
        d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
        return NormalDist().cdf(d1)

    def get_orderbook(self, symbol: str, is_buy: bool) -> Dict[int, int]:
        od = self.order_depths[symbol]
        return od.sell_orders if is_buy else od.buy_orders

    def get_volume_at_best(self, symbol: str, is_buy: bool) -> int:
        book = self.get_orderbook(symbol, is_buy)
        if not book:
            return 0
        best_price = min(book) if is_buy else max(book)
        return -book[best_price] if is_buy else book[best_price]

    def consume_best(self, symbol: str, size: int, is_buy: bool, orders: Dict[str, List[Order]]) -> int:
        book = self.get_orderbook(symbol, is_buy)
        if not book:
            return 0
        price = min(book) if is_buy else max(book)
        available = -book[price] if is_buy else book[price]
        volume = min(size, available)
        if volume <= 0:
            return 0
        orders[symbol].append(Order(symbol, price, volume if is_buy else -volume))
        book[price] += volume if is_buy else -volume
        self.position[symbol] += volume if is_buy else -volume
        return volume

    def get_max_position_volume(self, symbol: str, is_buy: bool) -> int:
        current_pos = self.position[symbol]
        limit = self.limits[symbol]
        return max(0, (limit - current_pos) if is_buy else (limit + current_pos))

    def generate_orders(self, state: TradingState, trader_data: dict) -> Dict[str, List[Order]]:
        self.position = defaultdict(int, state.position)
        self.order_depths = deepcopy(state.order_depths)
        orders = defaultdict(list)

        underlying = self.underlying_symbol
        spot = get_mid_price(state, underlying, 'mid_price', 0)
        TTE = max(0.01, (5 - state.timestamp / 1e6) / 250)

        trader_data.setdefault("coeff_history", [])
        temp = []
        m_list, v_list, meta = [], [], []
        for symbol, K in self.option_strikes.items():
            if symbol not in self.order_depths:
                continue
            option_mid = get_mid_price(state, symbol, 'mid_price', 0)
            if option_mid <= 0:
                continue
            iv = self.implied_volatility(option_mid, spot, K, TTE)
            m = log(K / spot) / sqrt(TTE)
            m_list.append(m)
            v_list.append(iv)
            meta.append((symbol, K, iv, m, option_mid))

        if len(m_list) < 3:
            return {}

        if len(trader_data["coeff_history"]) < self.history_window:
            coeffs = np.polyfit(m_list, v_list, deg=2)
            trader_data["coeff_history"].append(coeffs.tolist())
            return {}

        coeff_array = np.array(trader_data["coeff_history"][-self.history_window:])
        avg_coeffs = np.mean(coeff_array, axis=0)
        fit_avg = np.poly1d(avg_coeffs)

        filtered_m, filtered_v = [], []
        for _, _, iv, m, _ in meta:
            if abs(iv - fit_avg(m)) < self.outlier_threshold:
                filtered_m.append(m)
                filtered_v.append(iv)
        if len(filtered_m) >= 3:
            trader_data["coeff_history"].append(np.polyfit(filtered_m, filtered_v, 2).tolist())

        for symbol, K, iv, m, option_mid in meta:
            fitted_iv = fit_avg(m)
            rel_diff = iv - fitted_iv
            temp.append(rel_diff)
            abs_rel_diff = abs(rel_diff)

            delta_val = self.delta(spot, K, TTE, iv)
            pos = self.position[symbol]

            # print(f"[{symbol}] IV: {iv:.4f}, Fit: {fitted_iv:.4f}, Diff: {rel_diff:.4f} → {'ENTRY' if abs_rel_diff >= self.entry_threshold else ('OUTLIER' if abs_rel_diff >= self.outlier_threshold else 'NORMAL')}")

            if abs_rel_diff < self.outlier_threshold:
                if pos == 0:
                    continue
                is_long = pos > 0
                option_is_buy = not is_long
                hedge_is_buy = is_long

                option_volume = self.get_volume_at_best(symbol, option_is_buy)
                hedge_volume = self.get_volume_at_best(underlying, hedge_is_buy)

                max_close = min(
                    abs(pos),
                    option_volume,
                    int(hedge_volume / abs(delta_val)),
                    self.get_max_position_volume(underlying, hedge_is_buy)
                )
                if max_close > 0:
                    self.consume_best(symbol, max_close, option_is_buy, orders)
                    self.consume_best(underlying, round(delta_val * max_close), hedge_is_buy, orders)

            elif abs_rel_diff >= self.entry_threshold:
                is_underpriced = rel_diff < 0
                option_is_buy = is_underpriced
                hedge_is_buy = not is_underpriced

                option_volume = self.get_volume_at_best(symbol, option_is_buy)
                hedge_volume = self.get_volume_at_best(underlying, hedge_is_buy)

                max_trade = min(
                    option_volume,
                    int(hedge_volume / abs(delta_val)),
                    self.get_max_position_volume(symbol, option_is_buy),
                    self.get_max_position_volume(underlying, hedge_is_buy)
                )
                if max_trade > 0:
                    self.consume_best(symbol, max_trade, option_is_buy, orders)
                    self.consume_best(underlying, round(delta_val * max_trade), hedge_is_buy, orders)
            else:
                continue
        print(f"Temp: {temp}")
        return dict(orders)


class SyntheticOnlyBasketStrategy:
    def __init__(self, combos, limits, rules, print_log, mid_method):
        self.combos = combos
        self.limits = limits
        self.rules = rules
        self.print_log = print_log
        self.mid_method = mid_method
        self.orders = {}
        self.history = {basket: deque() for basket in combos}
        self.stats = {basket: {"avg": 0.0, "std": 0.0, "spread": 0.0} for basket in combos}

    def load_history(self, trader_data: dict):
        history_data = trader_data.get("basket_history", {})
        for basket in self.combos:
            self.history[basket] = deque(history_data.get(basket, []))

    def save_history(self, trader_data: dict):
        trader_data["basket_history"] = {
            basket: list(deq) for basket, deq in self.history.items()
        }

    def get_auto_side(self, components, spread):
        target_asset = max(
            (item for item in components.items() if item[1] < 0),
            key=lambda x: abs(x[1]),
            default=(None, 0)
        )[0]
        side = "sell" if spread > 0 else "buy"
        return target_asset, side

    def run(self, state, trader_data: dict):
        self.orders.clear()
        self.load_history(trader_data)
        basket_signals = []
        for basket, components in self.combos.items():
            spread = sum(w * get_mid_price(state, s, method=self.mid_method, min_vol=0) for s, w in components.items())
            self.history[basket].append(spread)
            if len(self.history[basket]) < 10:
                continue
            history_list = list(self.history[basket])
            avg = np.mean(history_list)
            std = np.std(history_list[-30:])
            z = (spread - avg) / std if std > 1e-6 else 0.0

            if self.print_log:
                print(f"[{basket}] SPREAD: {spread:.2f}, Z: {z:.2f}")

            self.stats[basket].update({"avg": avg, "std": std, "spread": spread})
            if z > 7:
                basket_signals.append((abs(z), z, basket, components, "open", "sell", True))
            elif z < -7:
                basket_signals.append((abs(z), z, basket, components, "open", "buy", True))

        # for basket, components in self.combos.items():
        #     spread = sum(w * get_mid_price(state, s, method=self.mid_method, min_vol=0) for s, w in components.items())
        #     self.history[basket].append(spread)
        #     if len(self.history[basket]) < 10:
        #         continue
        #     history_list = list(self.history[basket])
        #     avg = np.mean(history_list)
        #     std = np.std(history_list[-50:])
        #     z = (spread - avg) / std if std > 1e-6 else 0.0

        #     if self.print_log == 1:
        #         print(f"Mean: {avg}")
        #         print(f"[{basket}] SPREAD: {spread:.2f}, Z: {z:.2f}")

        #     self.stats[basket].update({"avg": avg, "std": std, "spread": spread})

        #     for rule in self.rules.get(basket, []):
        #         side = None
        #         if rule["action"] == "open" and abs(z) >= rule["z_min"]:
        #             _, side = self.get_auto_side(components, spread)
        #             basket_signals.append((abs(z), z, basket, components, "open", side, rule["all_levels"]))
        #         elif rule["action"] == "close" and rule["z_min"] <= z <= rule["z_max"]:
        #             current_position = state.position.get(basket, 0)
        #             if current_position == 0:
        #                 continue

                    # entry_spread = self.history[basket][0]
                    # pnl = (spread - entry_spread) * current_position
                    # pnl_ratio = pnl / (abs(entry_spread) + 1e-6)

                    # if self.print_log == 1:
                    #     print(f"[{basket}] PnL: {pnl:.2f}, PnL%: {pnl_ratio:.2%}")

                    # if pnl_ratio >= self.take_profit_threshold or pnl_ratio <= self.stop_loss_threshold:
                    #     basket_signals.append((abs(z), z, basket, components, "close", side, rule["all_levels"]))
                    #     self.history[basket].clear()

        basket_signals.sort(reverse=True, key=lambda x: x[0])
        random.shuffle(basket_signals)

        for _, z, basket, components, action, side, all_levels in basket_signals:
            units = self.get_executable_units(state, components, side, all_levels, float('inf'))
            if self.print_log:
                print(f"[{basket}] Z: {z:.2f} → {action.upper()} {side or '→'} Units: {units}")
            if units > 0:
                self.execute_synthetic_trade(state, components, side, units, all_levels)
        self.save_history(trader_data)
        return self.orders

    def get_executable_units(self, state, components, side, all_levels, max_unit=float('inf')):
        def get_qty(od, side):
            if not od.sell_orders and not od.buy_orders:
                return 0
            levels = sorted(od.sell_orders.items()) if side == "buy" else sorted(od.buy_orders.items(), reverse=True)
            qty = sum(-v if side == "buy" else v for _, v in levels) if all_levels else abs(levels[0][1])
            return qty
        pos = state.position
        units = float('inf')
        for s, w in components.items():
            if w == 0:
                continue
            actual_qty = w if side == "buy" else -w
            component_side = "buy" if actual_qty > 0 else "sell"
            od = state.order_depths[s]
            component_depth_qty = get_qty(od, component_side)
            current_pos = pos.get(s, 0)
            if component_side == "buy":
                max_by_pos = max(0, self.limits[s] - current_pos)
            else:
                max_by_pos = max(0, self.limits[s] + current_pos)
            if abs(w) < 1e-6:
                continue
            units_by_depth = component_depth_qty // abs(w)
            units_by_pos = max_by_pos // abs(w)
            units = min(units, units_by_depth, units_by_pos, max_unit)
        return max(0, int(units))

    def execute_synthetic_trade(self, state, components, side, units, all_levels):
        for s, w in components.items():
            actual_qty = units * w if side == "buy" else -units * w
            trade_side = "buy" if actual_qty > 0 else "sell"
            self.place_order(state, s, trade_side, abs(actual_qty), all_levels)

    def place_order(self, state, symbol, side, total_qty, all_levels):
        od = state.order_depths[symbol]
        book = od.sell_orders if side == "buy" else od.buy_orders
        sorted_book = sorted(book.items()) if side == "buy" else sorted(book.items(), reverse=True)
        remaining = total_qty
        current_pos = state.position.get(symbol, 0)
        limit = self.limits[symbol]
        for price, vol in sorted_book:
            available = -vol if side == "buy" else vol
            if available <= 0:
                continue
            max_allowed = limit - current_pos if side == "buy" else limit + current_pos
            qty = min(remaining, available, max_allowed)
            if qty <= 0:
                break
            self.orders.setdefault(symbol, []).append(
                Order(symbol, price, qty if side == "buy" else -qty)
            )
            remaining -= qty
            state.position[symbol] = current_pos + (qty if side == "buy" else -qty)
            current_pos = state.position[symbol]
            if not all_levels or remaining <= 0:
                break


class Trader:
    def __init__(self):
        self.print_log = False

        self.underlying_symbol = "VOLCANIC_ROCK"
        self.option_symbols = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]

        self.option_strikes = {
            symbol: int(symbol.split("_")[-1])
            for symbol in self.option_symbols
        }

        self.option_limits = {
            self.underlying_symbol: 400
        }
        for symbol in self.option_symbols:
            self.option_limits[symbol] = 200

        self.option_trader = VolSmileArbTrader(
            option_strikes=self.option_strikes,
            limits=self.option_limits,
            z_threshold=0.1,
            history_window=50,
            best_only_entry=True,
            best_only_exit=False
        )

        self.symbols = ["KELP", "RAINFOREST_RESIN", "SQUID_INK"]
        self.strategy_config = {
            "KELP": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": True, "adverse_volume": 15, "take_width": 1},
                "liquidation_args": {"liquidate_width": 0.5},
                "market_make_args": {"market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 1},
                "true_value_args": {"use_fixed": False}
            },
            "RAINFOREST_RESIN": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": True, "adverse_volume": 15, "take_width": 1},
                "liquidation_args": {"liquidate_width": 0.5},
                "market_make_args": {"market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 1},
                "true_value_args": {"use_fixed": False}
            },
            "SQUID_INK": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": True, "adverse_volume": 15, "take_width": 1},
                "liquidation_args": {"liquidate_width": 1},
                "market_make_args": {"market_make_spread": 1, "price_filter_width": 1, "fallback_offset": 2},
                "true_value_args": {"use_fixed": False}
            }
        }

        # Initialize market makers for different symbols
        self.market_makers = []
        for symbol in self.symbols:
            config = self.strategy_config[symbol]
            mm = MarketMakingStrategy(
                symbol=symbol,
                limit=config["limit"],
                print_log=self.print_log,
                take_order_fn=aggressive_take_orders,
                take_order_args=config.get("take_order_args"),
                liquidation_fn=clear_position_liquidation,
                liquidation_args=config.get("liquidation_args"),
                market_make_fn=default_market_make,
                market_make_args=config.get("market_make_args"),
                true_value_args=config.get("true_value_args")
            )
            self.market_makers.append(mm)

        self.shared_limits = {
            'CROISSANTS': 250,
            'JAMS': 350,
            'DJEMBES': 60,
            'PICNIC_BASKET1': 60,
            'PICNIC_BASKET2': 100
        }
        self.strategy_rules = {
            "DJEMBES": [
                {"z_min": 5, "action": "open", "all_levels": False},
                {"z_min": -0.1, "z_max": 0.1, "action": "close", "all_levels": False},
            ],
            "PICNIC_BASKET1": [
                {"z_min": 5, "action": "open", "all_levels": False},
                {"z_min": -0.1, "z_max": 0.1, "action": "close", "all_levels": False},
            ],
            "PICNIC_BASKET2": [
                {"z_min": 5, "action": "open", "all_levels": False},
                {"z_min": -0.1, "z_max": 0.1, "action": "close", "all_levels": False},
            ]
        }

        self.combo_strategy = SyntheticOnlyBasketStrategy(
            combos={  # Define basket combinations with weights
                "DJEMBES": {"PICNIC_BASKET1": 2, "PICNIC_BASKET2": -3, 'DJEMBES': -2},
                "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1, "PICNIC_BASKET1": -1},
                "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2, "PICNIC_BASKET2": -1},
            },
            limits=self.shared_limits,
            rules=self.strategy_rules,
            print_log=self.print_log,
            mid_method="weighted_average",  # Method to calculate the mid-price
        )

    def run_market_makers(self, state):
        # Run the market-making strategies and generate orders
        mm_orders = {}
        for mm in self.market_makers:
            orders, _ = mm.run(state)
            for order in orders:
                mm_orders.setdefault(order.symbol, []).append(order)
        return mm_orders
    
    def run_basket_strategy(self, state, trader_data: dict):
        return self.combo_strategy.run(state, trader_data)
               
    def load_trader_data(self, trader_data_str: str) -> Dict[str, Any]:
        try:
            return json.loads(trader_data_str) if trader_data_str else {}
        except Exception as e:
            print(f"[Trader] Failed to load traderData: {e}")
            return {}

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        trader_data = self.load_trader_data(state.traderData)
        all_orders: Dict[str, List[Order]] = {}

        if True:
            print(f"[{state.timestamp}] POSITIONS → " + ", ".join(
                f"{symbol}: {state.position.get(symbol, 0)}"
                for symbol in self.option_limits
            ))

        # mm_orders = self.run_market_makers(state)
        # for symbol, orders in mm_orders.items():
        #     all_orders.setdefault(symbol, []).extend(orders)

        # combo_orders = self.run_basket_strategy(state, trader_data)
        # for symbol, orders in combo_orders.items():
        #     all_orders.setdefault(symbol, []).extend(orders)

        option_orders = self.option_trader.generate_orders(state, trader_data)
        for symbol, orders in option_orders.items():
            all_orders.setdefault(symbol, []).extend(orders)

        return all_orders, 0, json.dumps(trader_data)