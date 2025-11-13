# Projet EDA : Fouille Interactive de Motifs avec Préférences Utilisateur

Ce projet a été réalisé dans le cadre du cours de Fouille de Données (SCIA-G). Il propose une solution complète pour extraire, filtrer et explorer des motifs fréquents (règles d'association) dans des bases de données transactionnelles.

L'objectif principal est de résoudre le problème de l'explosion combinatoire des motifs en intégrant une boucle de rétroaction utilisateur (feedback) et des techniques d'échantillonnage pondéré.

## Fonctionnalités Principales

Le projet implémente un pipeline de traitement complet comprenant :

1.  **Ingestion de Données Multi-formats :** Support des fichiers transactionnels (CSV, TXT) sous différents formats :
      * *Basic :* Listes d'articles séparés par des virgules.
      * *Long (Transactionnel) :* Une ligne par article (ID Transaction, Article).
      * *Wide :* Matrice binaire ou articles en colonnes.
      * *Sequential :* Support préliminaire pour les données séquentielles.
2.  **Extraction de Motifs :** Utilisation de l'algorithme FP-Growth pour une extraction efficace des itemsets fréquents et génération de règles d'association (via la bibliothèque `mlxtend`).
3.  **Scoring Composite :** Classement des règles basé sur une combinaison linéaire de métriques objectives (Lift, Confiance, Support) et de mesures subjectives (Surprise, Pénalité de redondance).
4.  **Échantillonnage Intelligent :**
      * Échantillonnage pondéré standard.
      * Algorithme MCMC léger (Markov Chain Monte Carlo) pour l'exploration de l'espace des solutions.
5.  **Interface Interactive (Streamlit) :**
      * Configuration dynamique des paramètres d'extraction et de scoring.
      * Visualisation des règles.
      * Mécanisme de feedback ("Like"/"Dislike") permettant à l'utilisateur de réorienter la recherche en temps réel.

## Architecture du Projet

Le code source est organisé de manière modulaire dans le dossier `src/` pour assurer la maintenabilité et la séparation des responsabilités.

  * `src/app.py` : Point d'entrée de l'application Web Streamlit. Gère l'interface, l'état de session et l'orchestration.
  * `src/load_data.py` : Classes et fonctions pour le chargement, le nettoyage et la transformation des données (One-Hot Encoding).
  * `src/mining.py` : Encapsulation des algorithmes de fouille (FP-Growth, Association Rules).
  * `src/scoring.py` : Logique de calcul des scores composites et normalisation des métriques.
  * `src/sampling.py` : Implémentation des algorithmes de sélection (Weighted Sampling, Light MCMC).
  * `src/feedback.py` : Gestion de la mise à jour des poids basée sur les interactions utilisateur.

## Installation

### Prérequis

  * Python 3.8 ou supérieur.
  * Un environnement virtuel est recommandé.

### Installation des dépendances

Installez les bibliothèques nécessaires via `pip` :

```bash
pip install pandas numpy mlxtend streamlit scikit-learn
```

Ou si un fichier `requirements.txt` est présent :

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancement de l'Application

Pour démarrer l'interface utilisateur, exécutez la commande suivante depuis la racine du projet :

```bash
streamlit run src/app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

### Guide de l'Utilisateur

1.  **Chargement :**
      * Importez votre fichier CSV ou TXT via la barre latérale.
      * Sélectionnez le format approprié (ex: "Long" pour un fichier type base de données).
      * Indiquez les colonnes correspondant à l'identifiant de transaction et aux articles.
2.  **Extraction :**
      * Ajustez le curseur "Support Minimum". Pour des données éparses, commencez avec une valeur faible (ex: 0.05 soit 5%).
3.  **Exploration :**
      * Visualisez les motifs suggérés par l'algorithme.
      * Utilisez les curseurs de "Poids" pour privilégier certaines métriques (ex: augmenter le poids du Lift pour trouver des corrélations fortes).
4.  **Feedback :**
      * Cliquez sur "Like" pour renforcer un type de motif pertinent.
      * Cliquez sur "Dislike" pour pénaliser un motif non pertinent.
      * L'échantillon se mettra à jour automatiquement pour refléter vos préférences.

## Auteurs

Projet réalisé par le groupe SCIA-G - EPITA.