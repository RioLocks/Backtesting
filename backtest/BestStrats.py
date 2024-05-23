import os
import pandas as pd

# Chemin du répertoire contenant les fichiers de backtest
backtest_directory = r"./database/backtest_results"

# Initialisation d'un DataFrame pour stocker les meilleurs résultats
best_results = pd.DataFrame(columns=['Pair', 'Strategy', 'Sharpe_Ratio', 'Timeframe'])

# Liste tous les fichiers CSV dans le répertoire de backtest
file_paths = [os.path.join(backtest_directory, file) for file in os.listdir(backtest_directory) if file.endswith(".csv")]

# Parcourir les fichiers et extraire les meilleurs résultats pour chaque paire
for file_path in file_paths:
    # Charger le fichier CSV
    df = pd.read_csv(file_path)
    
    # Trouver la meilleure stratégie pour chaque paire dans ce fichier
    best_strategy = df.groupby('Pair').apply(lambda x: x.loc[x['Sharpe_Ratio'].idxmax()]).reset_index(drop=True)
    
    # Extraire le timeframe à partir du nom du fichier
    timeframe = os.path.basename(file_path).split('_')[-1].split('.')[0]
    
    # Ajouter la colonne Timeframe
    best_strategy['Timeframe'] = timeframe
    
    # Concaténer avec les meilleurs résultats précédents
    best_results = pd.concat([best_results, best_strategy[['Pair', 'Strategy', 'Timeframe', 'Period_Start', 'Period_End', 'Sharpe_Ratio', 'Initial_Balance', 'Final_Balance', 'Total_Trades', 'Global_Win_Rate', 'Worst_Drawdown']]])

# Supprimer les stratégies en double pour chaque paire
final_results_grouped = best_results.sort_values(by='Sharpe_Ratio', ascending=False).drop_duplicates(subset='Pair')

# Réinitialiser l'index
final_results_grouped.reset_index(drop=True, inplace=True)

# Chemin du répertoire où tu veux sauvegarder le fichier final
final_results_directory = r"./database/backtest_results"

# Nom du fichier CSV de sortie
output_file = 'best_strategies_resume.csv'

# Chemin complet du fichier de sortie
output_path = os.path.join(final_results_directory, output_file)

# Enregistrer les résultats dans un fichier CSV dans le dossier spécifié
final_results_grouped.to_csv(output_path, index=False)

# Afficher un message de confirmation
print(f"Les meilleures stratégies avec leurs timeframes respectifs ont été enregistrées dans '{output_path}'.")