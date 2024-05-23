# Backtesting Framework

Ce projet de backtesting permet de télécharger des données depuis des exchanges, de backtester plusieurs paires, stratégies et timeframes, ainsi que de rechercher les meilleures stratégies pour chaque paire et timeframe en utilisant la parallélisation.

# Table des matières
1. Installation
2. Utilisation
  - Télécharger les données
  - Backtester les stratégies
  - Rechercher les meilleures stratégies
- Fonctionnalités
- Contribuer
- License

# Installation

Pour installer les dépendances nécessaires, exécutez la commande suivante :

- pip install pandas matplotlib numpy ta joblib

Assurez-vous également que vous avez Node.js et npm installés pour exécuter le script de téléchargement de données.

# Utilisation

- Télécharger les données

Avant de pouvoir backtester, vous devez télécharger les données historiques. Utilisez le script download_data.js pour cela :

1. Ouvrez un terminal.
2. Naviguez vers le répertoire du projet (cd database).
3. Exécutez la commande suivante pour installer les dépendances Node.js :

npm install

4. Utilisez le script download_data.js pour télécharger les données :

node download_data.js

# Backtester les stratégies

Vous pouvez backtester des stratégies en utilisant les scripts fournis. Trois options sont disponibles : backtester au choix une seule ou plusieurs, paires timeframes stratégies avec "BT_Choose.py", Backtester en "masse", toutes les paires et timeframes que vous aurez acquises depuis l'exchange et backtester avec la parallélisation pour rechercher les meilleurs paramètres d'une seule paire.

Backtester une seule paire et timeframe avec une seule stratégie :

Modifiez les paramètres dans le fichier "BT_Choose.py" pour inclure la paire, le timeframe et la stratégie souhaités, puis exécutez :

1. cd backtest
2. python BT_Choose.py

Backtester plusieurs paires, stratégies et timeframes en masse :

1. Se rendre dans le répertoire "backtest" 
2. Exécuter "python BT_Massive.py"

Backtester avec la parallélisation pour rechercher les meilleures paramètres :

1. Se rendre dans le répertoire de "backtest"
2. Exécuter "python BT_Parallel.py"


# Rechercher les meilleures stratégies

Après avoir exécuté les backtests, vous pouvez rechercher les meilleures stratégies pour chaque paire et timeframe en utilisant le script BestStrats.py.

Modifiez les paramètres dans BestStrats.py si nécessaire.

1. Se rendre dans le répertoire "backtest"
2. Exécuter "python BestStrats.py"

Ce script analysera les résultats des backtests et déterminera les meilleures stratégies basées sur les critères que vous avez définis.

# Fonctionnalités

- Téléchargement de données historiques : Téléchargez des données depuis des exchanges pour différentes paires et timeframes.
- Backtesting : Backtestez plusieurs stratégies pour une ou plusieurs paires et timeframes.
- Parallélisation : Utilisez la parallélisation pour accélérer les backtests de multiples paramètres.
- Analyse des résultats : Recherchez les meilleures stratégies en fonction des résultats des backtests.

# Contribuer

Les contributions sont les bienvenues ! Veuillez créer une issue pour discuter de ce que vous souhaitez améliorer ou ajouter. Vous pouvez également soumettre des pull requests avec vos modifications.

# License

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

Avec ces instructions, vous devriez être en mesure de télécharger les données, de backtester des stratégies et de rechercher les meilleures stratégies de manière efficace. N'hésitez pas à ajuster les détails spécifiques selon la configuration de votre projet et vos besoins.