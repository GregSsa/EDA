# Projet EDA : Fouille Interactive de Motifs avec Préférences Utilisateur

Ce projet a été réalisé dans le cadre du cours de Fouille de Données (SCIA-G). Il propose une application interactive permettant d'extraire, d'explorer et de filtrer des règles d'association dans des bases de données transactionnelles, en résolvant le problème de l'explosion combinatoire par l'interaction utilisateur.

## 🚀 Fonctionnalités Clés

L'application implémente un pipeline complet :

1.  **Ingestion de Données Flexible :**
    * Support automatique de multiples formats CSV (Transactions, Matrices binaires, Séquentiel) géré par la classe `TransactionDf`.
    * Pré-traitement et encodage (One-Hot) intégrés.

2.  **Double Approche d'Extraction :**
    * **Exhaustive (FP-Growth) :** Génération classique de règles via `mlxtend` pour les petits datasets.
    * **Output Sampling (MCMC) :** Un algorithme de *Pattern Sampling* basé sur une chaîne de Markov (Metropolis-Hastings) pour générer directement des motifs intéressants sans énumération totale.

3.  **Visualisation Interactive (Plotly) :**
    * **Scatter Plot :** Visualisation *Support vs Confiance* (taille = Lift) pour repérer les règles performantes.
    * **Heatmap :** Matrice de corrélation *Antécédents vs Conséquents* pour identifier les zones chaudes.
    * **Distributions :** Histogrammes comparatifs des scores avant/après feedback.

4.  **Boucle de Feedback (Relevance Feedback) :**
    * Interface "Like" (👍) / "Dislike" (👎) pour pondérer les règles.
    * Algorithme d'échantillonnage pondéré (`light_mcmc`) qui apprend des préférences utilisateur pour proposer de nouvelles règles pertinentes.

## 📂 Structure du Projet

L'architecture a été simplifiée pour faciliter l'exécution :

* **`app.py`** : Point d'entrée de l'application Web (Streamlit). Gère l'interface, les onglets, la visualisation et la gestion d'état (Session State).
* **`utils.py`** : Bibliothèque de fonctions backend. Contient le chargement de données, les calculs de scores composites, et les algorithmes MCMC.
* **`requirements.txt`** : Liste des dépendances Python.

## 🛠️ Installation

1.  **Prérequis :** Python 3.8+ installé.
2.  **Installation des dépendances :**

```bash
pip install -r requirements.txt
```

Voici la partie demandée formatée en Markdown, prête à être copiée :

````markdown
*Principales librairies utilisées : `streamlit`, `pandas`, `mlxtend`, `plotly`, `numpy`.*

## ▶️ Utilisation

### Lancement de l'Application

Pour démarrer l'interface utilisateur, exécutez la commande suivante depuis la racine du projet :

```bash
streamlit run app.py
````

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

### Guide de l'Utilisateur

1.  **Onglet 📊 Données :**
    Chargez votre fichier CSV. L'application détecte le format via `TransactionDf` et affiche les statistiques de base (Top items, nombre de transactions).

2.  **Onglet ⚙️ Extraction & Pool :**

      * Choisissez la méthode : **FP-Growth** (classique) ou **Sampling** (MCMC).
      * Cliquez sur **"Lancer l'Extraction"**.
      * Analysez le *Scatter Plot* global et utilisez les filtres par produits pour explorer le pool de règles.

3.  **Onglet 🎮 Échantillonnage Interactif :**

      * Cliquez sur **"Générer un nouvel échantillon"** pour voir un sous-ensemble de règles (ex: k=5).
      * Donnez votre avis : **Like (👍)** pour renforcer un type de règle, **Dislike (👎)** pour le pénaliser.
      * Régénérez : l'algorithme vous proposera des règles similaires à celles que vous avez aimées, basées sur le score composite (Lift, Confiance, Support, Surprise).

## 👥 Auteurs

Projet réalisé par le groupe SCIA-G - EPITA.

```