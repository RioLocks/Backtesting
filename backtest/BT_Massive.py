import os
import pandas as pd
from strategies import *
import warnings
import csv

# Filtrer les avertissements pour les ignorer
warnings.filterwarnings("ignore")

class BTMassive:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.data = {}
        self.trades = {}
        self.days = {}
        # self.timeframes = timeframes
        self.strategies = {
            "Bollinger_Crossover": BollingerCrossover,
            "Bollinger_Volatility": BollingerVolatility,
            "Alligator_Strategy": AlligatorStrategy,
            "TSuperTrend_Strategy": TSuperTrendStrategy,
            "CrossEMAStochRSI": CrossEMAStochRSI,
            "ichiCloudStochRSI": ichiCloudStochRSI,
            "VolumeAnomaly": VolumeAnomaly
        }

    def load_data(self):
        for root, dirs, files in os.walk(self.data_dir):
            for dir_name in dirs:
                # Si le nom du sous-dossier correspond à un timeframe
                if dir_name.startswith("data_"):
                    current_timeframe = dir_name.split("_")[1]
                    data_dir = os.path.join(root, dir_name)
                    
                    # Parcours de tous les fichiers CSV dans le sous-dossier
                    for file in os.listdir(data_dir):
                        if file.endswith(".csv"):
                            # Analyse du nom de fichier pour obtenir la paire
                            current_symbol = file.split(".")[0]
                            
                            # Chargement des données depuis le fichier CSV dans un DataFrame
                            filepath = os.path.join(data_dir, file)
                            df = pd.read_csv(filepath)
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)  # Définir le timestamp comme index

                            # Stockage du DataFrame dans la structure de données
                            if current_symbol not in self.data:
                                self.data[current_symbol] = {}
                            self.data[current_symbol][current_timeframe] = df
                            
        print("Data loaded")


    def calculate_indicators(self, symbol, timeframe, strategy_name):
        print(f"Calculating indicators for {symbol} in {timeframe} with {strategy_name} strategy...")
        strategy_class = self.strategies[strategy_name]
        df = self.data[symbol][timeframe]  # Obtenez le DataFrame correspondant au symbole et au timeframe
        strategy = strategy_class(df)  # Passez le DataFrame à la classe de stratégie
        indicators = strategy.calculate_indicators()  # Utilisez la méthode correcte pour calculer les indicateurs
        print("Indicators calculated")
        return indicators

    
    def run_backtest(self, symbol, timeframe, strategy_name):
        print(f"Running backtest for {symbol} in {timeframe} with {strategy_name} strategy...")
        balance = 1000
        position = None
        fees = 0.0007
        previous_day = -1
        df = self.data[symbol][timeframe]
        self.trades[f"{symbol}_{timeframe}_{strategy_name}"] = []  # Initialisation des trades spécifiques à la paire, au timeframe et à la stratégie
        self.days[f"{symbol}_{timeframe}_{strategy_name}"] = []    # Initialisation des jours spécifiques à la paire, au timeframe et à la stratégie
        for index in range(len(df)):
            current_day = df.index[index].day
            if previous_day != current_day:
                temp_balance = balance
                if position:
                    close_price = df.iloc[index]["close"]
                    trade_result = (close_price - position["open_price"]) / position["open_price"]
                    close_size = position["open_size"] + position["open_size"] * trade_result
                    fee = close_size * fees
                    close_size -= fee
                    temp_balance = temp_balance + close_size - position["open_size"]
                self.days[f"{symbol}_{timeframe}_{strategy_name}"].append({  # Ajout des résultats spécifiques à la paire, au timeframe et à la stratégie
                    "day": df.index[index].date(),
                    "balance": temp_balance,
                    "price": df.iloc[index]['close']
                })
            previous_day = current_day

            if position is None and df.iloc[index]["buy_signal"]:
                open_price = df.iloc[index]["close"]
                open_size = balance
                fee = open_size * fees
                open_size -= fee
                balance -= fee
                stop_loss = open_price - (open_price * 0.1)
                position = {
                    "open_price": open_price,
                    "open_size": balance,
                    "open_date": df.index[index],
                    "open_fee": fee,
                    "open_reason": "Market Buy",
                    "open_balance": balance,
                    "stop_loss": stop_loss,
                }

            elif position and df.iloc[index]["sell_signal"]:
                close_price = df.iloc[index]["close"]
                trade_result = (close_price - position["open_price"]) / position["open_price"]
                close_size = position["open_size"] + position["open_size"] * trade_result
                fee = close_size * fees
                close_size -= fee
                balance += close_size - position["open_size"]
                self.trades[f"{symbol}_{timeframe}_{strategy_name}"].append({  # Ajout des résultats spécifiques à la paire, au timeframe et à la stratégie
                    "open_date": position["open_date"],
                    "close_date": df.index[index],
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
                })
                position = None
        print("Backtest completed")
                        

    def basic_multi_asset_backtest(self):
        print("Running basic multi-asset backtest...")
        # Initialize a dictionary to store results for each pair and timeframe
        all_results = {}

        # Loop over each symbol and timeframe
        for symbol, timeframe_data in self.data.items():
            for timeframe, df in timeframe_data.items():
                for strategy_name in self.strategies.keys():
                    # Initialize a list to store results for this timeframe
                    if timeframe not in all_results:
                        all_results[timeframe] = []

                    # Initialize dictionaries to store results for each pair and timeframe
                    pair_results = {  
                        "Pair": symbol,
                        "Strategy": strategy_name,
                        "Period_Start": [],
                        "Period_End": [],
                        "Initial_Balance": [],
                        "Final_Balance": [],
                        "Sharpe_Ratio": [],
                        "Total_Trades": [],
                        "Global_Win_Rate": [],
                        "Average_Profit": [],
                        "Worst_Drawdown": [],
                        "Mean_Trades_Duration": []
                    }

                    # Perform backtest calculations
                    df_trades = pd.DataFrame(self.trades.get(f"{symbol}_{timeframe}_{strategy_name}", {}))
                    df_days = pd.DataFrame(self.days.get(f"{symbol}_{timeframe}_{strategy_name}", {}))

                    if df_trades.empty:
                        print(f"No trades found for {symbol} - {timeframe} - {strategy_name}")
                        continue
                    if df_days.empty:
                        print(f"No days found for {symbol} - {timeframe} - {strategy_name}")
                        continue

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
                    # mean_trades_duration_seconds = mean_trades_duration.total_seconds()  # Convert to seconds
                    mean_trades_per_days = total_trades / total_days

                    best_trade = df_trades.loc[df_trades["trade_result_pct"].idxmax()]
                    worst_trade = df_trades.loc[df_trades["trade_result_pct"].idxmin()]

                    sharpe_ratio = (365 ** (0.5) * df_days['daily_return'].mean()) / df_days['daily_return'].std()

                    # Store results in dictionary
                    pair_results["Period_Start"] = str(df_days.iloc[0]["day"])
                    pair_results["Period_End"] = str(df_days.iloc[-1]["day"])
                    pair_results["Initial_Balance"] = df_days.iloc[0]["balance"]
                    pair_results["Final_Balance"] = df_days.iloc[-1]["balance"]
                    pair_results["Sharpe_Ratio"] = sharpe_ratio
                    pair_results["Total_Trades"] = total_trades
                    pair_results["Global_Win_Rate"] = global_win_rate * 100
                    pair_results["Average_Profit"] = avg_profit * 100
                    pair_results["Worst_Drawdown"] = max_days_drawdown * 100
                    pair_results["Mean_Trades_Duration"] = mean_trades_duration

                    # Append results to the list for this timeframe
                    all_results[timeframe].append(pair_results)

        # Save the results to CSV files by timeframe
        output_directory = "./database/backtest_results"
        for timeframe, results in all_results.items():
            filename = os.path.join(output_directory, f"backtest_results_{timeframe}.csv")  # Nom du fichier CSV en fonction du timeframe
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ["Pair", "Strategy", "Period_Start", "Period_End", "Initial_Balance",
                            "Final_Balance", "Sharpe_Ratio", "Total_Trades", "Global_Win_Rate",
                            "Average_Profit", "Worst_Drawdown", "Mean_Trades_Duration"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
        print("Backtest results saved to CSV files.")

# Définir le répertoire contenant les fichiers CSV
data_dir = r"./database/Binance"

# Initialiser l'objet MultiStrat
multi_strat = BTMassive(data_dir)

# Charger les données pour toutes les paires et timeframes disponibles
multi_strat.load_data()

# Calculer les indicateurs et effectuer le backtesting pour chaque paire, timeframe et stratégie
for symbol, symbol_data in multi_strat.data.items():
    for timeframe, df in symbol_data.items():
        for strategy_name in multi_strat.strategies.keys():
            # Calculer les indicateurs
            indicators = multi_strat.calculate_indicators(symbol, timeframe, strategy_name)
            # Effectuer le backtesting
            backtest = multi_strat.run_backtest(symbol, timeframe, strategy_name)

# Exécuter l'analyse de base du backtesting une fois que tous les backtests ont été effectués
backtest_analysis = multi_strat.basic_multi_asset_backtest()


