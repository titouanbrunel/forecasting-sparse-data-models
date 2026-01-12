# Système de Prédiction de Demande

## Vue d'ensemble

Système de prédiction automatisé pour deux types de familles de produits :
- **Familles fréquentes** : commandées régulièrement (plusieurs fois par mois)
- **Familles rares** : commandées sporadiquement (moins d'une fois par mois)

Le système utilise des modèles différenciés selon la nature intermittente ou régulière de la demande.

## Architecture des Dossiers

```
projet/
├── data/
│   ├── weather.json                     # Données météorologiques quotidiennes
│   ├── events.json                      # Événements (grèves, festivals, confinements)
│   ├── ...                              # données initiales (commandes, article)
│   ├── quantites.py                     # Script de création des fichiers de commande (sans Gratuit/Avoir)
│   ├── quantites.csv                    # Données de commandes initiales regroupées par familles et commandes
│   ├── quantites_w.csv                  # Données de commandes de quantites.csv regroupées par semaines et par clients
│   ├── {client_id}/                     # Dossier par client (ex: 230248/)
│   │   ├── frequentes.csv               # INPUT: Familles fréquemment commandées
│   │   ├── rares.csv                    # INPUT: Familles rarement commandées
│   │   ├── predictions.csv              # OUTPUT: Prédictions familles fréquentes
│   │   ├── predictions_rares.csv        # OUTPUT: Prédictions familles rares
│   │   ├── coef_families.csv            # OUTPUT: Coefficients Modèle Historique
│   │   ├── frequente_prediction/        # OUTPUT: Graphiques familles fréquentes
│   │   └── rares_prediction/            # OUTPUT: Graphiques familles rares
│   ├── families_w/
│   │   ├── {famille}.csv                # INPUT: Données globales par famille
│   │   └── coefficients.csv             # OUTPUT: Coefficients Modèle 1
│   └── not_b4_2024/                     # clients sans commandes avant 2024
│       ├── client_2024.csv
│       └── recapitulatif.csv
├── meteo_marseille/ 
│   ├── vacances/                        # INPUT: Données Google de vacances
│   ├── weather_data_{date}.csv          # INPUT: Données Google de météo par jour
│   └── create_weather.py                # Script pour la création des données météo pour les modèles --> weather.json
├── main_frequentes.py                   # Script principal familles fréquentes
├── main_rares.py                        # Script principal familles rares
├── model_comparison_results.csv         # OUTPUT: Métriques familles fréquentes
├── rare_model_comparison_results.csv    # OUTPUT: Métriques familles rares
└── coef_families.csv                    # OUTPUT: Coefficients globaux Modèle 3
```

## Guide d'Exécution Complet

### Étape 1 : Préparation des Données Brutes

**1.1. Données météorologiques**
```bash
# Exécuter depuis meteo_marseille/
python create_weather.py
# Mettre ensuite le json creer dans data/
```
**Résultat** : Génère `data/weather.json` à partir des fichiers `weather_data_{date}.csv`
(Très important de mettre le fichier `weather.json` initialement dans `meteo_marseille/` dans `data/` !!)

### Étape 2 : Création des Fichiers par Client

**2.1. Données de commandes**
```bash
# Exécuter depuis data/
python quantites.py
```
**Résultat globaux** : 
- Crée `quantites.csv` (données nettoyées sans Gratuit/Avoir)
- Génère `quantites_w.csv` (agrégation hebdomadaire par client)


**Resultats Individuels importants : Séparation familles fréquentes/rares**
 - Le script creer surtout les csv principaux :
```bash
data/{client_id}/frequentes.csv
data/{client_id}/rares.csv
```

**Resultats par famille importants : Données globales par famille**
- Il cree egalement les fichiers pour le model commun aux clients d'une meme famille et les sauvegarde dans `data/families_w/` :
```bash
data/families_w/ALCOOLS.csv
data/families_w/BIERE_BOITE.csv
...
```

### Étape 3 : Préparation des Événements

**3.1. Fichier events.json**
Créer manuellement ou a partir d une IA `data/events.json` avec la structure :
```json
{
  "events": [
    {
      "date": "2020-03-17",
      "name": "Confinement COVID",
      "type": "lockdown",
      "duration_days": 55,
      "impact": -0.6
    }
  ]
}
```

### Étape 4 : Exécution des Modèles de Prédiction

**4.1. Familles fréquentes**
```bash
python main_frequentes.py
```
**Durée estimée** : 5-7 heures (200 clients)

**Résultats générés** :
```bash
data/families_w/coefficients.csv           # Coefficients Modèle 1
data/{client_id}/coef_families.csv         # Coefficients Modèle 3
data/{client_id}/predictions.csv           # Prédictions Modèle 2
data/{client_id}/frequente_prediction/     # Graphiques comparatifs
model_comparison_results.csv               # Métriques globales
```

**4.2. Familles rares**
```bash
python main_rares.py
```
**Durée estimée** : 1 heure

**Résultats générés** :
```bash
data/{client_id}/predictions_rares.csv     # Prédictions 2 modèles
data/{client_id}/rares_prediction/         # Graphiques comparatifs
rare_model_comparison_results.csv          # Métriques globales
```

## Scripts Principaux

### main_frequentes.py

**Objectif** : Prédiction des familles commandées régulièrement via 3 modèles différents

**Classes principales** :
- `DataLoader` : Chargement et agrégation données météo/événements
- `CoefficientModel` : Génération coefficients multiplicateurs (2 méthodes)
- `EnhancedXGBoostHurdleModel` : Modèle hurdle avec XGBoost
- `HurdleClientModel` : Prédictions directes individualisées
- `ModelComparison` : Comparaisons et graphiques

**Données sources** :
```bash
data/weather.json
data/events.json
data/families_w/*.csv
data/{client_id}/frequentes.csv
```

**Sorties générées** :
```bash
data/families_w/coefficients.csv
data/{client_id}/coef_families.csv
data/{client_id}/predictions.csv
data/{client_id}/frequente_prediction/*.png
```

### main_rares.py

**Objectif** : Prédiction des familles à demande intermittente via 2 approches

**Classes principales** :
- `CrostonModel` : Implémentation algorithme Croston
- `RareFamiliesModel` : Génération prédictions
- `RareModelComparison` : Comparaisons et graphiques

**Données sources** :
```bash
data/{client_id}/rares.csv
```

**Sorties générées** :
```bash
data/{client_id}/predictions_rares.csv
data/{client_id}/rares_prediction/*.png
```

## Modèles pour Familles Fréquentes

### Modèle 1 : Hurdle Prédictif Global

**Principe** :
1. Agrège les quantités de tous les clients par famille
2. Entraîne un modèle hurdle XGBoost sur les données globales
3. Prédit la demande globale pour chaque semaine 2024
4. Calcule un coefficient : `prédiction_globale / moyenne_52_semaines`
5. Applique ce coefficient à la moyenne historique des 52 dernières semaines de chaque client

**Features utilisées** :
- Lags temporels (1, 2, 3, 4 semaines)
- Moyennes et écarts-types mobiles (4, 8 semaines)
- Variables saisonnières (semaine, mois, trimestre)
- Tendance et accélération

**Sortie** :
```bash
data/families_w/coefficients.csv
```

### Modèle 2 : Hurdle Direct Client

**Principe** :
1. Entraîne un modèle hurdle individualisé pour chaque client
2. Utilise l'historique complet de toutes les familles du client
3. Intègre des features météorologiques et d'événements
4. Prédit directement les quantités par famille

**Features utilisées** :
- Features temporelles enrichies (cycliques, saisonnières)
- Historique client (moyennes, fréquences, volatilité)
- Données météo hebdomadaires (température, précipitations, soleil)
- Événements (confinements, grèves, événements sportifs/culturels)
- Variables composites (qualité météo, intensité vacances)

**Architecture hurdle** :
- **Étape 1** : Régression logistique pour P(commande > 0)
- **Étape 2** : XGBoost pour E[quantité | commande > 0]
- **Prédiction finale** : P(commande > 0) × E[quantité | commande > 0]

**Sortie** :
```bash
data/{client_id}/predictions.csv
```

### Modèle 3 : Coefficient Historique

**Principe** :
1. Calcule pour chaque semaine historique : `coefficient = quantité / moyenne_52_semaines_précédentes`
2. Pour chaque semaine 2024, prédit le coefficient comme la moyenne des coefficients de cette même semaine des années précédentes
3. Applique ce coefficient à la moyenne historique des 52 dernières semaines du client

**Avantage** : Capture les patterns saisonniers récurrents sans machine learning

**Sortie** :
```bash
data/{client_id}/coef_families.csv
```

## Modèles pour Familles Rares

### Modèle Reproduction Année Passée

**Principe** :
1. Pour la semaine N de 2024, copie la valeur de la semaine N de 2023
2. Si aucune donnée en 2023, calcule la fréquence historique des commandes
3. Applique cette fréquence à la quantité moyenne historique

**Logique** : Les commandes rares suivent souvent des patterns annuels fixes

### Modèle Croston (Modèle en phase de test) #!mauvais resultats pour le moment

**Principe** : Algorithme spécialisé pour demandes intermittentes
1. Sépare la série en deux composantes :
   - Tailles des demandes non-nulles
   - Intervalles entre demandes
2. Applique un lissage exponentiel à chaque composante séparément
3. Prédiction = `demande_lissée / intervalle_lissé`

**Paramètres** :
- Alpha = 0.2 (paramètre de lissage)
- Adapté aux séries avec nombreux zéros

**Avantage** : Conçu spécifiquement pour la nature sporadique des familles rares


### Fichiers de sortie

**predictions.csv / predictions_rares.csv** :
```csv
date,famille,prediction
2024-W01,ALCOOLS,3.2
2024-W01,BIERE_BOITE,1.8
```

**coefficients.csv / coef_families.csv** :
```csv
date,famille,coefficient
2024-01-01,ALCOOLS,1.15
2024-01-01,BIERE_BOITE,0.92
```

## Points de Contrôle

### Vérifications intermédiaires

**Après Étape 1** :
- Vérifier la présence de `data/weather.json`


**Après Étape 2** :
- Contrôler `data/quantites.csv` et `data/quantites_w.csv`
- Valider la création des dossiers `data/{client_id}/`
- Vérifier les fichiers `frequentes.csv` et `rares.csv` pour quelques clients

**Après Étape 4.1** :
- Contrôler la génération des prédictions pour les premiers clients
- Vérifier les graphiques de comparaison

**Après Étape 4.2** :
- Valider les prédictions pour familles rares
- Contrôler la cohérence avec les données d'entrée

## Gestion des Erreurs

**Clients sans historique** :
Les clients sans commandes avant 2024 sont automatiquement exclus et placés dans :
```bash
data/not_b4_2024/client_2024.csv
data/not_b4_2024/recapitulatif.csv
```

**Fichiers manquants** :
Le système continue l'exécution même si certains fichiers sont absents, en excluant automatiquement les clients concernés.

## Performance Estimée

**Temps d'exécution** (200 clients, 30 familles) :
- `main_frequentes.py` : 5-7 heures
- `main_rares.py` : 1 heure
- Génération graphiques : 30-45 minutes

**Optimisations possibles** :
- Parallélisation par client
- Réduction paramètres XGBoost pour tests (si les résultats conviennent)

## Fichier en plus deja present

Le système génère automatiquement :
- **Graphiques comparatifs** : Réel vs prédictions des différents modèles 

Le système est conçu pour être robuste aux données manquantes et s'adapte automatiquement aux cas d'erreur.