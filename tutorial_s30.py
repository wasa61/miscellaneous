import math
import json
import random
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional
from datamodel import Order, TradingState

# === Utility Functions ===

def get_mid_price(state: TradingState, symbol: str, method: str = 'weighted_average', min_vol: int = 15) -> float:
    order_depth = state.order_depths[symbol]
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0.0

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    total_ask = sum(-price * qty for price, qty in order_depth.sell_orders.items())
    total_bid = sum(price * qty for price, qty in order_depth.buy_orders.items())
    qty_ask = sum(-qty for qty in order_depth.sell_orders.values())
    qty_bid = sum(order_depth.buy_orders.values())

    filtered_best_ask = min([p for p, q in order_depth.sell_orders.items() if -q >= min_vol], default=best_ask)
    filtered_best_bid = max([p for p, q in order_depth.buy_orders.items() if q >= min_vol], default=best_bid)

    mid = (best_bid + best_ask) / 2
    weighted_avg = (total_ask + total_bid) / (qty_ask + qty_bid)
    filtered_mid = (filtered_best_ask + filtered_best_bid) / 2 if filtered_best_ask and filtered_best_bid else mid

    return {
        'mid_price': mid,
        'weighted_average': weighted_avg,
        'filtered_mid_price': filtered_mid,
        'ensemble': (weighted_avg + filtered_mid) / 2
    }.get(method, mid)

def get_moving_average(history: deque, symbol: str, method: str, window_size: int, min_vol: int) -> Optional[float]:
    if len(history) < window_size:
        return None
    values = [
        get_mid_price(state, symbol, method, min_vol)
        for state in list(history)[-window_size:]
    ]
    return sum(values) / window_size


def aggressive_take_orders(symbol, state, true_value, remaining_buy, remaining_sell,
                           take_width, prevent_adverse=False, adverse_volume=0):
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
    buy_price_liquidate = round(true_value - liquidate_width)
    sell_price_liquidate = round(true_value + liquidate_width)

    to_clear = position - remaining_buy + remaining_sell

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

def no_liquidation(symbol, state, true_value, position, remaining_buy, remaining_sell, liquidate_width):
    return [], remaining_buy, remaining_sell

def default_market_make(symbol, state, true_value, remaining_buy, remaining_sell,
                        market_make_spread, price_filter_width, fallback_offset):
    orders = []
    order_depth = state.order_depths[symbol]
    sell_prices = [price for price in order_depth.sell_orders if price > true_value + price_filter_width]
    buy_prices = [price for price in order_depth.buy_orders if price < true_value - price_filter_width]
    sell_base = min(sell_prices) if sell_prices else true_value + fallback_offset
    buy_base = max(buy_prices) if buy_prices else true_value - fallback_offset
    buy_price_make = round(buy_base - market_make_spread)
    sell_price_make = round(sell_base + market_make_spread)
    if remaining_buy > 0:
        orders.append(Order(symbol, buy_price_make, remaining_buy))
    if remaining_sell > 0:
        orders.append(Order(symbol, sell_price_make, -remaining_sell))
    return orders


class QLearningController:
    def __init__(self, actions, alpha=0.5, gamma=0.95, epsilon=0.3):
        self.actions = actions
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action_with_index(self, state):
        if random.random() < self.epsilon:
            idx = random.randint(0, len(self.actions) - 1)
        else:
            idx = int(np.argmax(self.Q[state]))
        return idx, self.actions[idx]

    def update(self, last_state, last_action, reward, next_state):
        if last_state is None or last_action is None:
            return
        if last_action >= len(self.actions):
            return
        q_old = self.Q[last_state][last_action]
        q_max_next = np.max(self.Q[next_state])
        self.Q[last_state][last_action] = q_old + self.alpha * (reward + self.gamma * q_max_next - q_old)

    def save(self):
        return {json.dumps(k): list(v) for k, v in self.Q.items()}

    def load(self, q_table):
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))
        for k, v in q_table.items():
            self.Q[tuple(json.loads(k))] = np.array(v)


class MarketMakingStrategy:
    def __init__(self, symbol: str, limit: int, print_log: int = 0,
                 take_order_fn=None, take_order_args=None,
                 liquidation_fn=None, liquidation_args=None,
                 market_make_fn=None, market_make_args=None,
                 true_value_args=None):
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

        # self.history = deque(maxlen=10)
        self.true_value_history = deque(maxlen=30)
        self.window = deque()
        self.window_size = 3

        self.remaining_buy = 0
        self.remaining_sell = 0

        self.orders = []
        self.conversions = 0

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        # self.history.append(copy.deepcopy(state))
        true_value = self.get_true_value(state)
        self.act(state, true_value)
        return self.orders, self.conversions

    def act(self, state: TradingState, true_value: float):
        position = state.position.get(self.symbol, 0)
        self.remaining_buy = self.limit - position
        self.remaining_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        print('startstart') if self.print_log == 1 else None
        print(f"to_buy{self.remaining_buy}...") if self.print_log == 1 else None
        print(f"to_sell{self.remaining_sell}...") if self.print_log == 1 else None
        print(f"position{position}...") if self.print_log == 1 else None

        if self.take_order_fn:
            orders, self.remaining_buy, self.remaining_sell = self.take_order_fn(
                self.symbol, state, true_value,
                self.remaining_buy, self.remaining_sell,
                **self.take_order_args
            )
            self.orders.extend(orders)
            if self.print_log:
                for o in orders:
                    print(f"[{self.symbol}] TAKE → {'BUY' if o.quantity > 0 else 'SELL'} {abs(o.quantity)} @ {o.price}") if self.print_log == 1 else None

        if self.liquidation_fn:
            orders, self.remaining_buy, self.remaining_sell = self.liquidation_fn(
                self.symbol, state, true_value,
                position,
                self.remaining_buy, self.remaining_sell,
                **self.liquidation_args
            )
            self.orders.extend(orders)
            if self.print_log:
                for o in orders:
                    print(f"[{self.symbol}] LIQUIDATE → {'BUY' if o.quantity > 0 else 'SELL'} {abs(o.quantity)} @ {o.price}") if self.print_log == 1 else None

        if self.market_make_fn:
            orders = self.market_make_fn(
                self.symbol, state, true_value,
                self.remaining_buy, self.remaining_sell,
                **self.market_make_args
            )
            self.orders.extend(orders)
            if self.print_log:
                for o in orders:
                    print(f"[{self.symbol}] MAKE → {'BUY' if o.quantity > 0 else 'SELL'} {abs(o.quantity)} @ {o.price}") if self.print_log == 1 else None

        print('endend') if self.print_log == 1 else None

    def get_true_value(self, state: TradingState) -> float:
        use_fixed = self.true_value_args.get("use_fixed", False)
        fixed_value = self.true_value_args.get("fixed_value", None)
        if use_fixed and fixed_value is not None:
            return fixed_value
        true_value = get_mid_price(state, self.symbol, method="weighted_average", min_vol=15)
        return true_value
        # self.true_value_history.append(true_value)
        # if self.symbol == "KELP":
        #     if len(self.true_value_history) > 10:
        #         return np.median(self.true_value_history)
        #     else:
        #         return true_value

    def save(self):
        return list(self.window)

    def load(self, data):
        self.window = deque(data) if data else deque()


class Trader:
    def __init__(self):
        self.print_log = 1
        self.symbols = ["KELP", "RAINFOREST_RESIN"]

        self.q_controllers = {
            "KELP": QLearningController(actions=[
                {"take_width": 1, "market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 1.5, "liquidate_width": 1},
                {"take_width": 1, "market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 1, "liquidate_width": 1},
                {"take_width": 1, "market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 2, "liquidate_width": 1}
            ]),
            "RAINFOREST_RESIN": QLearningController(actions=[
                {"take_width": 1, "market_make_spread": -1, "price_filter_width": 1, "fallback_offset": 2, "liquidate_width": 1}
            ])
        }

        self.strategy_config = {
            "KELP": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": True, "adverse_volume": 15},
                "liquidation_args": {},
                "market_make_args": {},
                "true_value_args": {"use_fixed": False}
            },
            "RAINFOREST_RESIN": {
                "limit": 50,
                "take_order_args": {"prevent_adverse": False, "adverse_volume": 0},
                "liquidation_args": {},
                "market_make_args": {},
                "true_value_args": {"use_fixed": True, "fixed_value": 10000}
            }
        }

        self.cashs = defaultdict(float)
        self.pnl = defaultdict(float)
        self.pnl_history = defaultdict(lambda: deque(maxlen=20))
        self.last_timestamp = -100
        self.last_states = {}
        self.last_actions = {}

    def get_state(self, symbol: str, state: TradingState) -> Tuple:
        position = state.position.get(symbol, 0)
        best_bid = max(state.order_depths[symbol].buy_orders.keys(), default=0)
        best_ask = min(state.order_depths[symbol].sell_orders.keys(), default=0)
        spread = best_ask - best_bid if best_bid and best_ask else 1
        if self.strategy_config[symbol]["true_value_args"].get("use_fixed", False):
            true_value = self.strategy_config[symbol]["true_value_args"]["fixed_value"]
        else:
            true_value = get_mid_price(state, symbol, method="weighted_average", min_vol=15)
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else true_value
        tv_diff = round(true_value - mid_price)
        clipped_position = max(-5, min(5, int(position / 10)))
        return (clipped_position, int(spread), tv_diff)

    def save_state(self) -> str:
        return json.dumps({
            "cashs": dict(self.cashs),
            "pnl": dict(self.pnl),
            "pnl_history": {k: list(v) for k, v in self.pnl_history.items()},
            "last_timestamp": self.last_timestamp,
            "q_tables": {
                sym: self.q_controllers[sym].save()
                for sym in self.symbols
            }
        })

    def load_state(self, data: str):
        if not data:
            return
        try:
            d = json.loads(data)
            self.cashs = defaultdict(float, d.get("cashs", {}))
            self.pnl = defaultdict(float, d.get("pnl", {}))
            self.last_timestamp = d.get("last_timestamp", -100)
            self.pnl_history = defaultdict(lambda: deque(maxlen=20), {
                k: deque(v, maxlen=20) for k, v in d.get("pnl_history", {}).items()
            })
            for sym in self.symbols:
                q_table = d.get("q_tables", {}).get(sym, {})
                self.q_controllers[sym].load(q_table)
        except Exception as e:
            print("[LOAD ERROR]", e)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if state.traderData:
            self.load_state(state.traderData)

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
            q_ctrl = self.q_controllers[symbol]
            state_tuple = self.get_state(symbol, state)

            action_idx, params = q_ctrl.select_action_with_index(state_tuple)
            self.last_states[symbol] = state_tuple
            self.last_actions[symbol] = action_idx

            if self.print_log:
                print(f"[{symbol}] Action Index: {action_idx}")

            strat = MarketMakingStrategy(
                symbol=symbol,
                limit=cfg["limit"],
                print_log=self.print_log,
                take_order_fn=aggressive_take_orders,
                take_order_args={**cfg["take_order_args"], "take_width": params["take_width"]},
                liquidation_fn=clear_position_liquidation,
                liquidation_args={**cfg["liquidation_args"], "liquidate_width": params["liquidate_width"]},
                market_make_fn=default_market_make,
                market_make_args={
                    **cfg["market_make_args"],
                    "market_make_spread": params["market_make_spread"],
                    "price_filter_width": params["price_filter_width"],
                    "fallback_offset": params["fallback_offset"]
                },
                true_value_args=cfg["true_value_args"]
            )

            orders, conv = strat.run(state)
            tv = strat.get_true_value(state)
            pos = state.position.get(symbol, 0)
            pnl_now = self.cashs[symbol] + pos * tv
            self.pnl[symbol] = pnl_now
            self.pnl_history[symbol].append(pnl_now)

            reward = 0
            if len(self.pnl_history[symbol]) >= 15:
                reward = self.pnl_history[symbol][-1] - self.pnl_history[symbol][0]

            q_ctrl.update(self.last_states[symbol], self.last_actions[symbol], reward, self.get_state(symbol, state))

            all_orders[symbol] = orders
            conversions += conv

            if self.print_log:
                print(f"[{symbol}] PnL: {pnl_now:.2f}, Cash: {self.cashs[symbol]:.2f}, Pos: {pos}, TV: {tv:.2f}, Reward: {reward:.2f}")

        self.last_timestamp = state.timestamp
        return all_orders, conversions, self.save_state()
