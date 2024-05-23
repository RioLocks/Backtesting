import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from ta.custom_indicators import SuperTrend
import warnings
from itertools import product
from joblib import Parallel, delayed
import os

# Filtrer les avertissements pour les ignorer
warnings.filterwarnings("ignore")


class BTParallel:
    def __init__(self) -> None:
        self.symbol = "BNBUSDT"
        self.dfs = {}
        self.trades = {}
        self.days = {}

    def get_data(self):
        binance_folder = r"./database/Binance"
        for subdir, _, files in os.walk(binance_folder):
            for file in files:
                if file.endswith(".csv") and self.symbol in file:
                    timeframe = os.path.basename(subdir).split('_')[-1]
                    df = pd.read_csv(os.path.join(subdir, file), parse_dates=['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df.apply(pd.to_numeric)
                    self.dfs[timeframe] = df

    def calculate_indicators(self, short_ema_window, long_ema_window, ecartement_bandes, atr_window, atr_multi, bb_window, bb_window_dev):
        for timeframe, df in self.dfs.items():
            warnings.filterwarnings("ignore")
            bb = BollingerBands(close=df['close'], window=bb_window, window_dev=bb_window_dev)
            super_trend = SuperTrend(df["high"], df["low"], df["close"], atr_window=atr_window, atr_multi=atr_multi)
            short_ema = EMAIndicator(close=df["close"], window=short_ema_window).ema_indicator()
            long_ema = EMAIndicator(close=df["close"], window=long_ema_window).ema_indicator()

            upper_band = bb.bollinger_hband()
            lower_band = bb.bollinger_lband()

            band_width = upper_band - lower_band

            df['bandes_ecartees'] = (band_width > band_width.rolling(window=ecartement_bandes).mean()) & (band_width.shift(1) <= band_width.rolling(window=ecartement_bandes).mean().shift(1))
            df['super_trend_direction'] = super_trend.super_trend_direction()
            df['short_ema'] = short_ema
            df['long_ema'] = long_ema

            condition_achat = (df['bandes_ecartees'] == True) & (df['super_trend_direction'] == True)
            condition_vente = (df['short_ema'] < df['long_ema']) & (df['short_ema'].shift(1) >= df['long_ema'].shift(1))
            df['buy_signal'] = condition_achat
            df['sell_signal'] = condition_vente

        return self.dfs

    def run_backtest(self):
        for timeframe, df in self.dfs.items():
            balance = 1000
            position = None
            fees = 0.0007
            trades = []
            days = []
            previous_day = -1

            for index, row in df.iterrows():
                current_day = index.day
                if previous_day != current_day:
                    temp_balance = balance
                    if position:
                        close_price = row["close"]
                        trade_result = (close_price - position["open_price"]) / position["open_price"]
                        close_size = position["open_size"] + position["open_size"] * trade_result
                        fee = close_size * fees
                        close_size -= fee
                        temp_balance = temp_balance + close_size - position["open_size"]
                    days.append({
                        "day": index.date(),
                        "balance": temp_balance,
                        "price": row['close']
                    })
                previous_day = current_day

                if position is None and row["buy_signal"]:
                    open_price = row["close"]
                    open_size = balance
                    fee = open_size * fees
                    open_size -= fee
                    balance -= fee
                    stop_loss = open_price - (open_price * 0.1)
                    position = {
                        "open_price": open_price,
                        "open_size": balance,
                        "open_date": index,
                        "open_fee": fee,
                        "open_reason": "Market Buy",
                        "open_balance": balance,
                        "stop_loss": stop_loss,
                    }

                elif position and row["sell_signal"]:
                    close_price = row["close"]
                    trade_result = (close_price - position["open_price"]) / position["open_price"]
                    close_size = position["open_size"] + position["open_size"] * trade_result
                    fee = close_size * fees
                    close_size -= fee
                    balance += close_size - position["open_size"]
                    trades.append(
                        {
                            "open_date": position["open_date"],
                            "close_date": index,
                            "open_price": position["open_price"],
                            "close_price": close_price,
                            "open_size": position["open_size"],
                            "close_size": close_size,
                            "open_fee": position["open_fee"],
                            "close_fee": fee,
                            "open_reason": position["open_reason"],
                            "close_reason": "Market Sell",
                            "open_balance": position["open_balance"],
                            "close_balance": balance,
                        }
                    )
                    position = None
            self.trades[timeframe] = trades
            self.days[timeframe] = days

    def backtest_analysis(self):
        for timeframe, trades in self.trades.items():
            df_trades = pd.DataFrame(trades)
            df_days = pd.DataFrame(self.days[timeframe])

            if df_trades.empty:
                raise Exception("No trades found")
            if df_days.empty:
                raise Exception("No days found")

            df_days['evolution'] = df_days['balance'].diff()
            df_days['daily_return'] = df_days['evolution'] / df_days['balance'].shift(1)

            df_trades["trade_result"] = df_trades["close_size"] - df_trades["open_size"]
            df_trades["trade_result_pct"] = df_trades["trade_result"] / df_trades["open_size"]
            df_trades["trades_duration"] = df_trades["close_date"] - df_trades["open_date"]

            df_days["balance_ath"] = df_days["balance"].cummax()
            df_days["drawdown"] = df_days["balance_ath"] - df_days["balance"]
            df_days["drawdown_pct"] = df_days["drawdown"] / df_days["balance_ath"]

            total_trades = len(df_trades)
            total_days = len(df_days)

            good_trades = df_trades.loc[df_trades["trade_result"] > 0]
            total_good_trades = len(good_trades)
            avg_profit_good_trades = good_trades["trade_result_pct"].mean()
            mean_good_trades_duration = good_trades["trades_duration"].mean()
            global_win_rate = total_good_trades / total_trades

            bad_trades = df_trades.loc[df_trades["trade_result"] < 0]
            total_bad_trades = len(bad_trades)
            avg_profit_bad_trades = bad_trades["trade_result_pct"].mean()
            mean_bad_trades_duration = bad_trades["trades_duration"].mean()

            max_days_drawdown = df_days["drawdown_pct"].max()
            initial_balance = df_days.iloc[0]["balance"]
            final_balance = df_days.iloc[-1]["balance"]
            balance_evolution = (final_balance - initial_balance) / initial_balance
            mean_trades_duration = df_trades["trades_duration"].mean()
            avg_profit = df_trades["trade_result_pct"].mean()
            mean_trades_duration = df_trades['trades_duration'].mean()
            mean_trades_per_days = total_trades / total_days

            best_trade = df_trades.loc[df_trades["trade_result_pct"].idxmax()]
            worst_trade = df_trades.loc[df_trades["trade_result_pct"].idxmin()]

            sharpe_ratio = (365 ** (0.5) * df_days['daily_return'].mean()) / df_days['daily_return'].std()

            self.days[timeframe] = df_days
            self.trades[timeframe] = df_trades
            self.trades[timeframe]['sharpe_ratio'] = sharpe_ratio


def run_backtest_with_params(short_ema_window, long_ema_window, ecartement_bandes, atr_window, atr_multi, bb_window, bb_window_dev):
    bt = BTParallel()
    bt.get_data()
    bt.calculate_indicators(short_ema_window=short_ema_window, long_ema_window=long_ema_window, ecartement_bandes=ecartement_bandes,
                            atr_window=atr_window, atr_multi=atr_multi, bb_window=bb_window,
                            bb_window_dev=bb_window_dev)
    bt.run_backtest()
    bt.backtest_analysis()
    final_balance = bt.days[list(bt.days.keys())[0]].iloc[-1]["balance"]
    max_days_drawdown = bt.days[list(bt.days.keys())[0]]["drawdown_pct"].max()
    sharpe_ratio = bt.trades[list(bt.trades.keys())[0]]['sharpe_ratio'].max()
    return {
        'short_ema_window': short_ema_window,
        'long_ema_window': long_ema_window,
        'ecartement_bandes': ecartement_bandes,
        'atr_window': atr_window,
        'atr_multi': atr_multi,
        'bb_window': bb_window,
        'bb_window_dev': bb_window_dev,
        'final_balance': final_balance,
        'Drawdown': max_days_drawdown,
        'sharpe_ratio': sharpe_ratio
    }


# Définir les plages de valeurs des paramètres à tester
short_ema_windows = [5, 10, 15, 20]
long_ema_windows = [60, 80, 100]
ecartements_bandes = [2, 4, 6]
atr_windows = [5, 10]
atr_multi = [2, 3, 4]
bb_windows = [10, 20]
bb_windows_dev = [2, 3]

print("Debut de l'analyse...")
# Utiliser Parallel pour exécuter les boucles en parallèle
results = Parallel(n_jobs=-1)(delayed(run_backtest_with_params)(short_ema_window, long_ema_window, ecartement_bandes, bb_window, bb_window_dev, atr_window, atr_multi)
                              for short_ema_window, long_ema_window, ecartement_bandes, bb_window, bb_window_dev, atr_window, atr_multi
                              in product(short_ema_windows, long_ema_windows, ecartements_bandes, bb_windows, bb_windows_dev, atr_windows, atr_multi))

valid_results = [result for result in results if result is not None]
print("Recherche des meilleurs resultats...")
# Calcul de la métrique composite pour les résultats valides
# Afficher les paramètres en cours de test
for params in product(short_ema_windows, long_ema_windows, ecartements_bandes, bb_windows, bb_windows_dev, atr_windows, atr_multi):
    print("Testing parameters:", params)

# Trouver les meilleurs paramètres basés sur la métrique choisie
best_balance = max(valid_results, key=lambda x: x['final_balance'])
print("\nMeilleure balance:")
print("Short EMA window:", best_balance['short_ema_window'])
print("Long EMA window:", best_balance['long_ema_window'])
print("Ecartement bandes:", best_balance['ecartement_bandes'])
print("ATR window:", best_balance['atr_window'])
print("ATR multiplier:", best_balance['atr_multi'])
print("BB window:", best_balance['bb_window'])
print("BB window deviation:", best_balance['bb_window_dev'])
print("Final balance:", best_balance['final_balance'])
print("Sharpe ratio:", best_balance['sharpe_ratio'])

# Trouver les meilleurs paramètres basés sur la métrique choisie
best_sharp_ratio = max(valid_results, key=lambda x: x['sharpe_ratio'])
print("\nMeilleur sharp ratio:")
print("Short EMA window:", best_sharp_ratio['short_ema_window'])
print("Long EMA window:", best_sharp_ratio['long_ema_window'])
print("Ecartement bandes:", best_sharp_ratio['ecartement_bandes'])
print("ATR window:", best_sharp_ratio['atr_window'])
print("ATR multiplier:", best_sharp_ratio['atr_multi'])
print("BB window:", best_sharp_ratio['bb_window'])
print("BB window deviation:", best_sharp_ratio['bb_window_dev'])
print("Final balance:", best_sharp_ratio['final_balance'])
print("Sharpe ratio:", best_sharp_ratio['sharpe_ratio'])

# Trouver les meilleurs paramètres basés sur la métrique choisie
less_drawdown = min(valid_results, key=lambda x: x['Drawdown'])
print("\nMoins de drawdown:")
print("Short EMA window:", less_drawdown['short_ema_window'])
print("Long EMA window:", less_drawdown['long_ema_window'])
print("Ecartement bandes:", less_drawdown['ecartement_bandes'])
print("ATR window:", less_drawdown['atr_window'])
print("ATR multiplier:", less_drawdown['atr_multi'])
print("BB window:", less_drawdown['bb_window'])
print("BB window deviation:", less_drawdown['bb_window_dev'])
print("Final balance:", less_drawdown['final_balance'])
print("Sharpe ratio:", less_drawdown['sharpe_ratio'])


# Calcul de la métrique composite pour les résultats valides
for result in valid_results:
    # Calculez la métrique composite en utilisant la nouvelle formule
    result['composite_metric'] = (1 / (result['Drawdown'] + 1)) * result['sharpe_ratio'] * result['final_balance']


# Trouvez la meilleure combinaison de paramètres basée sur la métrique composite
best_result = max(valid_results, key=lambda x: x['composite_metric'])

# Affichez les paramètres de la meilleure combinaison
print("Meilleure combinaison de paramètres basée sur la métrique composite:")
print("Short EMA window:", best_result['short_ema_window'])
print("Long EMA window:", best_result['long_ema_window'])
print("Ecartement bandes:", best_result['ecartement_bandes'])
print("ATR window:", best_result['atr_window'])
print("ATR multiplier:", best_result['atr_multi'])
print("BB window:", best_result['bb_window'])
print("BB window deviation:", best_result['bb_window_dev'])
print("Final balance:", best_result['final_balance'])
print("Sharpe ratio:", best_result['sharpe_ratio'])
print("Drawdown:", best_result['Drawdown'])
print("Composite metric:", best_result['composite_metric'])












# for result in valid_results:
#     result['composite_metric'] = (1 / (result['Drawdown'] + 1)) * result['sharpe_ratio'] * result['final_balance']

# # Initialisation des meilleurs résultats pour chaque métrique
# best_balance = {'final_balance': float('-inf')}
# best_sharpe_ratio = {'sharpe_ratio': float('-inf')}
# less_drawdown = {'Drawdown': float('inf')}
# best_composite_metric = {'composite_metric': float('-inf')}

# # Identification des meilleurs paramètres pour chaque métrique
# for result in valid_results:
#     if result['final_balance'] > best_balance['final_balance']:
#         best_balance = result
#     if result['sharpe_ratio'] > best_sharpe_ratio['sharpe_ratio']:
#         best_sharpe_ratio = result
#     if result['Drawdown'] < less_drawdown['Drawdown']:
#         less_drawdown = result
#     if result['composite_metric'] > best_composite_metric['composite_metric']:
#         best_composite_metric = result

# # Affichage des meilleurs paramètres pour chaque métrique
# print("\nMeilleure balance:")
# print(best_balance)
# print("\nMeilleur Sharpe ratio:")
# print(best_sharpe_ratio)
# print("\nMoins de drawdown:")
# print(less_drawdown)
# print("\nMeilleure combinaison de paramètres basée sur la métrique composite:")
# print(best_composite_metric)
