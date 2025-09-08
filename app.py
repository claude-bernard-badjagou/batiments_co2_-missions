
# Application Streamlit : Prédicteur de CO₂ pour bâtiments (version monofichier)

# 1) --- Importations ---
import os                                   # Chargeons os pour manipuler les chemins de fichiers et tester la présence des ressources
import io                                   # Chargeons io pour gérer des flux mémoire (export binaire des modèles)
import joblib                               # Chargeons joblib pour sérialiser/désérialiser le modèle et le scaler
import numpy as np                          # Chargeons numpy pour les calculs numériques (tableaux, histogrammes)
import pandas as pd                         # Chargeons pandas pour charger et manipuler le tableau de données
import streamlit as st                      # Chargeons Streamlit pour construire l'interface web
from typing import Tuple, Optional, List    # Chargeons les types pour annoter proprement les fonctions

# Importations scikit-learn pour prétraitement, modèle et métriques
from sklearn.model_selection import train_test_split   # Chargeons train_test_split pour créer un jeu d'entraînement/test
from sklearn.preprocessing import StandardScaler       # Chargeons StandardScaler pour mettre à l'échelle les variables explicatives
from sklearn.ensemble import RandomForestRegressor     # Chargeons RandomForestRegressor pour construire un modèle robuste et non linéaire
from sklearn.metrics import r2_score, mean_squared_error  # Chargeons les métriques R² et RMSE pour évaluer les performances

# 2) --- Constantes de ressources partagées ---
CHEMIN_DONNEES_PAR_DEFAUT = "df_resultat_analyse_batiments.csv"  # Définissons le nom du fichier CSV attendu
CHEMIN_MODELE = "model_co2.pkl"                                  # Définissons le nom du fichier modèle
CHEMIN_SCALER = "scaler_co2.pkl"                                 # Définissons le nom du fichier scaler

# 3) --- Colonnes par défaut (selon le code initial fourni) ---
COLONNES_FEATURES_PAR_DEFAUT: List[str] = [
    'Electricity(kBtu)',                    # Définissons la colonne de consommation électrique (kBtu)
    'NaturalGas(kBtu)',                     # Définissons la colonne de consommation de gaz (kBtu)
    'BuildingAge',                          # Définissons la colonne d'âge du bâtiment
    'Prop_LargestPropertyUseTypeGFA'        # Définissons la proportion de la surface principale
]
COLONNE_CIBLE_PAR_DEFAUT = 'TotalGHGEmissions'          # Définissons la colonne cible : émissions de GES totales

# 4) --- Config de page Streamlit ---
st.set_page_config(                         # Configurons la page pour un affichage large et un titre explicite
    page_title="Prédicteur CO₂ des bâtiments",
    page_icon="🌍",
    layout="wide"
)

# 5) --- Fonctions utilitaires (chargement, entraînement, prédiction) ---
@st.cache_data(show_spinner=False)
def charger_donnees(chemin: Optional[str] = None, fichier_televerse: Optional[io.BytesIO] = None) -> Optional[pd.DataFrame]:
    """Chargeons les données en priorité depuis le fichier fourni par l'utilisateur,
    sinon depuis le chemin par défaut, pour permettre l'exploration et l'entraînement."""
    # Si l'utilisateur a téléversé un fichier via l'interface, nous l'utilisons en priorité
    if fichier_televerse is not None:
        try:
            df = pd.read_csv(fichier_televerse)      # Lisons le CSV téléversé pour récupérer les données tabulaires
            return df                                # Renvoyons le DataFrame pour la suite de l'application
        except Exception as e:
            st.error(f"Erreur de lecture du CSV téléversé : {e}")  # Affichons une erreur en cas de problème de parsing
            return None

    # Sinon, tentons de lire depuis le chemin fourni ou celui par défaut
    chemin_effectif = chemin or CHEMIN_DONNEES_PAR_DEFAUT  # Déterminons le chemin effectif à utiliser
    if os.path.exists(chemin_effectif):
        try:
            df = pd.read_csv(chemin_effectif)        # Lisons le CSV local pour charger les données
            return df                                # Renvoyons le DataFrame si tout va bien
        except Exception as e:
            st.error(f"Erreur de lecture de '{chemin_effectif}' : {e}")  # Signalons l'exception
            return None
    else:
        return None                                   # Si le fichier n'existe pas, renvoyons None (géré par l'UI)


def preparer_X_y(
    df: pd.DataFrame,
    colonnes_features: List[str],
    colonne_cible: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Préparons X et y en vérifiant la présence des colonnes nécessaires pour l'entraînement/prédiction."""
    # Vérifions que toutes les colonnes features existent dans le DataFrame
    manquantes = [c for c in colonnes_features if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes explicatives manquantes : {manquantes}")  # Alertons si des colonnes sont absentes

    # Vérifions que la colonne cible existe
    if colonne_cible not in df.columns:
        raise ValueError(f"Colonne cible manquante : {colonne_cible}")       # Alertons si la cible est absente

    # Sélectionnons X (features) et y (cible)
    X = df[colonnes_features].copy()                 # Copions X pour éviter de modifier le DataFrame original
    y = df[colonne_cible].copy()                     # Copions y pour cohérence

    # Convertissons les colonnes en numérique si nécessaire (erreurs coercitives deviennent NaN)
    X = X.apply(pd.to_numeric, errors='coerce')      # Forçons les types numériques pour le modèle
    y = pd.to_numeric(y, errors='coerce')            # Forçons y en numérique pour la métrique

    # Supprimons les lignes avec NaN dans X ou y pour garantir un entraînement propre
    masque_valide = X.notna().all(axis=1) & y.notna()
    X = X[masque_valide]
    y = y[masque_valide]

    return X, y                                      # Renvoyons X et y prêts à l'emploi


@st.cache_resource(show_spinner=False)
def charger_ou_entrainer_modele(
    df: Optional[pd.DataFrame],
    colonnes_features: List[str] = COLONNES_FEATURES_PAR_DEFAUT,
    colonne_cible: str = COLONNE_CIBLE_PAR_DEFAUT,
    n_estimateurs: int = 300,
    profondeur_max: Optional[int] = None,
    graine: int = 42
):
    """Chargeons le modèle et le scaler depuis les fichiers partagés s'ils existent ;
    sinon, entraînons un nouveau modèle à partir du CSV (si disponible) et sauvegardons-les."""
    # Tentons d'abord de charger les artefacts existants pour éviter de réentraîner systématiquement
    if os.path.exists(CHEMIN_MODELE) and os.path.exists(CHEMIN_SCALER):
        try:
            modele = joblib.load(CHEMIN_MODELE)       # Chargeons le modèle entraîné depuis le fichier .pkl
            scaler = joblib.load(CHEMIN_SCALER)       # Chargeons le scaler entraîné depuis le fichier .pkl
            return modele, scaler, None               # Renvoyons les objets sans métriques (pas de réentraînement)
        except Exception as e:
            st.warning(f"Impossible de charger les artefacts existants : {e}. Entraînement prévu.")

    # Si l'on ne peut charger, essayons d'entraîner si des données sont disponibles
    if df is None:
        return None, None, None                       # Renvoyons None si nous ne pouvons pas entraîner faute de données

    # Préparons X et y en se basant sur les colonnes par défaut (ou personnalisées)
    X, y = preparer_X_y(df, colonnes_features, colonne_cible)

    # Séparons entraînement/test pour évaluer le modèle de manière honnête
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=graine
    )

    # Créons et ajustons le scaler sur l'entraînement (et transformons train/test)
    scaler = StandardScaler()                         # Instancions le scaler pour normaliser les variables explicatives
    X_train_s = scaler.fit_transform(X_train)         # Ajustons le scaler et transformons X_train pour centrer/réduire
    X_test_s = scaler.transform(X_test)               # Transformons X_test avec les mêmes paramètres appris

    # Instancions le modèle RandomForestRegressor avec des hyperparamètres raisonnables
    modele = RandomForestRegressor(
        n_estimators=n_estimateurs,
        max_depth=profondeur_max,
        random_state=graine,
        n_jobs=-1
    )

    # Entraînons le modèle sur les données mises à l'échelle
    modele.fit(X_train_s, y_train)

    # Évaluons les performances pour information (R² et RMSE)
    pred_test = modele.predict(X_test_s)              # Produisons des prédictions sur le test
    r2 = r2_score(y_test, pred_test)                  # Calculons le coefficient de détermination R²
    rmse = mean_squared_error(y_test, pred_test, squared=False)  # Calculons la racine de l'erreur quadratique moyenne

    # Sauvegardons les artefacts entraînés pour réutilisation
    joblib.dump(modele, CHEMIN_MODELE)                # Écrivons le modèle sur disque pour un usage ultérieur
    joblib.dump(scaler, CHEMIN_SCALER)                # Écrivons le scaler sur disque pour un usage ultérieur

    return modele, scaler, {"r2": r2, "rmse": rmse}  # Renvoyons le modèle, le scaler et les métriques


def predire_co2(
    modele: RandomForestRegressor,
    scaler: StandardScaler,
    tableau_features: np.ndarray
) -> float:
    """Appliquons le scaler puis le modèle pour prédire une valeur de CO₂ à partir d'un vecteur de features."""
    donnees_mises_echelle = scaler.transform(tableau_features)  # Transformons les features avec le scaler appris
    prediction = modele.predict(donnees_mises_echelle)           # Appliquons le modèle pour obtenir la prédiction
    return float(prediction[0])                                  # Convertissons la prédiction en float natif

# 6) --- Interface : Barre de navigation horizontale (onglets) ---
# Créons des onglets pour représenter les pages majeures de l'application
onglet_accueil, onglet_exploration, onglet_modele, onglet_prediction, onglet_api = st.tabs([
    "🏠 Accueil",
    "📊 Exploration",
    "🔧 Pretraitement & Modele",
    "🤖 Prediction",
    "📡 API & Export"
])

# 7) --- Gestion initiale des données (disponibles globalement) ---
with st.sidebar:
    # Proposons le téléversement du CSV si l'utilisateur ne souhaite pas s'appuyer sur le fichier local
    st.markdown("### 📥 Données")
    fichier_csv_televerse = st.file_uploader(
        "Téléverser df_resultat_analyse_batiments.csv (optionnel)", type=["csv"]
    )

# Chargeons les données soit depuis l'upload, soit depuis le fichier local par défaut
df_global = charger_donnees(CHEMIN_DONNEES_PAR_DEFAUT, fichier_csv_televerse)

# 8) --- Onglet Accueil ---
with onglet_accueil:
    st.markdown("""
    # 🌍 Prédicteur de CO₂ des bâtiments

    **But** : Estimer rapidement et de manière fiable les **émissions de CO₂** d'un bâtiment
    à partir de variables explicatives simples (consommation d'électricité et de gaz, âge, proportion de surface principale).

    **Ressources partagées** utilisées par défaut :
    - Données : `df_resultat_analyse_batiments.csv`
    - Modèle : `model_co2.pkl`
    - Scaler : `scaler_co2.pkl`

    **Navigation** : utilisez la barre horizontale en haut pour parcourir :
    1. *Exploration* des données (profilage rapide)
    2. *Prétraitement & Modèle* (entraînement, métriques et importances)
    3. *Prédiction* (formulaire interactif)
    4. *API & Export* (documentation et téléchargement des artefacts)
    """)

    # Affichons un rappel sur la disponibilité des données
    if df_global is None:
        st.info(
            "Aucune donnée locale détectée. Téléversez le CSV dans la barre latérale ou placez `df_resultat_analyse_batiments.csv` à la racine."
        )
    else:
        st.success("Données chargées avec succès : `df_resultat_analyse_batiments.csv`.")

# 9) --- Onglet Exploration ---
with onglet_exploration:
    st.markdown("## 📊 Exploration des données")

    if df_global is None:
        st.warning("Exploration indisponible : aucune donnée chargée.")
    else:
        # Affichons un aperçu et des infos générales
        st.markdown("### Aperçu du jeu de données")
        st.dataframe(df_global.head(20), width="stretch")  # Affichons les 20 premières lignes (width='stretch' remplace use_container_width)

        st.markdown("### Statistiques descriptives (numériques)")
        st.dataframe(df_global.describe(include='number'), width="stretch")  # Résumons les colonnes numériques

        st.markdown("### Valeurs manquantes par colonne")
        manquants = df_global.isna().sum().sort_values(ascending=False)  # Comptons les NaN pour chaque colonne
        st.dataframe(manquants.to_frame("nb_nan"), width="stretch")       # Affichons sous forme de tableau

        # Choix de colonnes numériques pour des graphiques rapides
        colonnes_numeriques = df_global.select_dtypes(include='number').columns.tolist()  # Récupérons les colonnes numériques disponibles
        if colonnes_numeriques:
            c1, c2 = st.columns(2)                    # Créons deux colonnes pour juxtaposer les visuels
            with c1:
                st.markdown("#### Histogramme rapide")
                col_hist = st.selectbox("Sélectionnez une colonne numérique", colonnes_numeriques)  # Laissons l'utilisateur choisir la variable
                try:
                    # Construisons un histogramme simple avec numpy
                    valeurs = df_global[col_hist].dropna().values           # Récupérons les valeurs non manquantes
                    if len(valeurs) > 0:
                        nb_bins = st.slider("Nombre de classes (bins)", 10, 100, 30)
                        histo, bords = np.histogram(valeurs, bins=nb_bins)             # Calculons l'histogramme
                        df_histo = pd.DataFrame({"classe": bords[:-1], "frequence": histo})  # Convertissons pour affichage
                        st.bar_chart(df_histo.set_index("classe"))                      # Affichons le graphique en barres
                    else:
                        st.info("Aucune valeur disponible pour tracer l'histogramme.")
                except Exception as e:
                    st.error(f"Erreur histogramme : {e}")

            with c2:
                st.markdown("#### Nuage de points (x vs y)")
                if len(colonnes_numeriques) >= 2:
                    x_col = st.selectbox("Axe X", colonnes_numeriques, index=0)      # Choisissons l'axe X
                    y_col = st.selectbox("Axe Y", colonnes_numeriques, index=1)      # Choisissons l'axe Y
                    try:
                        st.scatter_chart(df_global[[x_col, y_col]].dropna(), x=x_col, y=y_col)  # Traçons un nuage de points
                    except Exception as e:
                        st.error(f"Erreur nuage de points : {e}")
                else:
                    st.info("Nuage de points indisponible : moins de deux colonnes numériques.")

        # Corrélations (si suffisamment de colonnes numériques)
        if len(colonnes_numeriques) >= 2:
            st.markdown("### Matrice de corrélation (numérique)")
            try:
                corr = df_global[colonnes_numeriques].corr()  # Calculons les corrélations linéaires entre variables numériques
                st.dataframe(corr, width="stretch")           # Affichons la matrice (width='stretch')
            except Exception as e:
                st.error(f"Erreur corrélation : {e}")

# 10) --- Onglet Prétraitement & Modèle ---
with onglet_modele:
    st.markdown("## 🔧 Prétraitement & Modèle")

    if df_global is None:
        st.warning("Prétraitement indisponible : aucune donnée chargée.")
    else:
        # Paramétrage des colonnes et hyperparamètres
        st.markdown("### Paramètres de préparation et d'entraînement")
        colonnes_proposees = [c for c in COLONNES_FEATURES_PAR_DEFAUT if c in df_global.columns]  # Filtrons les features disponibles
        if len(colonnes_proposees) < len(COLONNES_FEATURES_PAR_DEFAUT):
            st.warning("Certaines colonnes par défaut manquent dans le CSV. Adaptez la sélection ci-dessous si nécessaire.")

        colonnes_features = st.multiselect(
            "Colonnes explicatives",
            options=df_global.columns.tolist(),
            default=colonnes_proposees
        )
        colonne_cible = st.selectbox(
            "Colonne cible (émissions de GES)",
            options=df_global.columns.tolist(),
            index=df_global.columns.get_loc(COLONNE_CIBLE_PAR_DEFAUT) if COLONNE_CIBLE_PAR_DEFAUT in df_global.columns else 0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            n_est = st.number_input("n_estimators (arbres)", min_value=50, max_value=1000, value=300, step=50)
        with c2:
            # ⚠️ Correctif Streamlit Cloud : autoriser 0 pour que la valeur initiale 0 ne plante pas
            profondeur_max = st.number_input(
                "max_depth (None = automatique)",
                min_value=0, max_value=100, value=0, step=1, help="0 = None (illimité)"
            )
            profondeur_max = None if profondeur_max == 0 else int(profondeur_max)
        with c3:
            graine = st.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

        # Entraînement / Chargement
        st.markdown("### Entraînement ou chargement du modèle")
        lancer_reentrainement = st.checkbox("Réentraîner maintenant (écrase les artefacts existants)")

        # Si l'utilisateur force le réentraînement, nous l'exécutons; sinon nous tentons de charger
        if lancer_reentrainement:
            try:
                X, y = preparer_X_y(df_global, colonnes_features, colonne_cible)
                modele, scaler, metriques = charger_ou_entrainer_modele(
                    df_global, colonnes_features, colonne_cible,
                    n_estimateurs=n_est, profondeur_max=profondeur_max, graine=graine
                )
                if metriques:
                    st.success(f"Modèle réentraîné. R² = {metriques['r2']:.4f} | RMSE = {metriques['rmse']:.4f}")
            except Exception as e:
                st.error(f"Erreur de réentraînement : {e}")
        else:
            # Tentons de charger, sinon entraînons automatiquement si possible
            modele, scaler, metriques = charger_ou_entrainer_modele(
                df_global, colonnes_features, colonne_cible,
                n_estimateurs=n_est, profondeur_max=profondeur_max, graine=graine
            )
            if metriques:
                st.info(f"Modèle entraîné (artefacts absents). R² = {metriques['r2']:.4f} | RMSE = {metriques['rmse']:.4f}")

        # Affichons les importances si un modèle est disponible et compatible
        if isinstance(modele, RandomForestRegressor) and len(colonnes_features) == len(modele.feature_importances_):
            st.markdown("### Importance des variables (Random Forest)")
            importances = pd.Series(modele.feature_importances_, index=colonnes_features).sort_values(ascending=False)
            st.bar_chart(importances)
        else:
            st.info("Importances non disponibles.")

# 11) --- Onglet Prédiction ---
with onglet_prediction:
    st.markdown("## 🤖 Prédiction interactive des émissions de CO₂")

    # Assurons-nous que nous pouvons charger le modèle et le scaler existants (entraînés ou chargés auparavant)
    modele_actuel, scaler_actuel, _ = charger_ou_entrainer_modele(df_global)

    if (modele_actuel is None) or (scaler_actuel is None):
        st.warning("Aucun modèle/scaler disponible. Rendez-vous dans l'onglet 'Pretraitement & Modele' pour entraîner/charger.")
    else:
        # Définissons des valeurs par défaut (médianes) si les données sont disponibles, sinon des valeurs génériques
        if df_global is not None and all(c in df_global.columns for c in COLONNES_FEATURES_PAR_DEFAUT):
            med_elec = float(pd.to_numeric(df_global['Electricity(kBtu)'], errors='coerce').median())
            med_gaz  = float(pd.to_numeric(df_global['NaturalGas(kBtu)'], errors='coerce').median())
            med_age  = float(pd.to_numeric(df_global['BuildingAge'], errors='coerce').median())
            med_prop = float(pd.to_numeric(df_global['Prop_LargestPropertyUseTypeGFA'], errors='coerce').median())
        else:
            med_elec, med_gaz, med_age, med_prop = 5_000.0, 1_000.0, 30.0, 0.6

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            conso_electricite = st.number_input("Consommation électrique (kBtu)", min_value=0.0, value=med_elec, step=100.0)
        with c2:
            conso_gaz = st.number_input("Consommation gaz (kBtu)", min_value=0.0, value=med_gaz, step=50.0)
        with c3:
            age_batiment = st.number_input("Âge du bâtiment (années)", min_value=0.0, value=med_age, step=1.0)
        with c4:
            prop_surface = st.number_input("Prop. surface principale (0-1)", min_value=0.0, max_value=1.0, value=med_prop, step=0.01)

        if st.button("Prédire les émissions de CO₂"):
            try:
                # Construisons le vecteur de features dans l'ordre d'entraînement par défaut
                vecteur = np.array([[conso_electricite, conso_gaz, age_batiment, prop_surface]], dtype=float)
                valeur_predite = predire_co2(modele_actuel, scaler_actuel, vecteur)
                st.success(f"Émissions de CO₂ estimées : {valeur_predite:.2f} (unités de la cible)")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

# 12) --- Onglet API & Export ---
with onglet_api:
    st.markdown("## 📡 API & Export des artefacts")

    st.markdown("""
    ### Documentation rapide de l'API prévue
    Nous exposons un point d'entrée **POST** `/predict` (FastAPI dans une application séparée) acceptant le JSON suivant :

    ```json
    {
      "electricity": 5000.0,
      "gas": 1000.0,
      "age": 30.0,
      "prop_gfa": 0.6
    }
    ```

    La réponse attendue :
    ```json
    { "prediction_CO2": 123.45 }
    ```

    > Remarque : dans cette version **monofichier** Streamlit, l'API n'est pas démarrée. 
    > Pour déployer une API, conservez les mêmes ressources partagées (`model_co2.pkl`, `scaler_co2.pkl`) 
    > et démarrez un service FastAPI séparé (ex. `uvicorn main:app --host 0.0.0.0 --port 8000`).
    """)

    # Proposons le téléchargement des artefacts si présents
    st.markdown("### Téléchargement des artefacts (si disponibles)")

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(CHEMIN_MODELE):
            try:
                with open(CHEMIN_MODELE, "rb") as f:
                    b = f.read()                               # Lisons le binaire du modèle pour le proposer en téléchargement
                st.download_button("Télécharger model_co2.pkl", b, file_name="model_co2.pkl")
            except Exception as e:
                st.error(f"Téléchargement modèle impossible : {e}")
        else:
            st.info("Aucun modèle trouvé à télécharger.")

    with col_b:
        if os.path.exists(CHEMIN_SCALER):
            try:
                with open(CHEMIN_SCALER, "rb") as f:
                    b = f.read()                               # Lisons le binaire du scaler pour le proposer en téléchargement
                st.download_button("Télécharger scaler_co2.pkl", b, file_name="scaler_co2.pkl")
            except Exception as e:
                st.error(f"Téléchargement scaler impossible : {e}")
        else:
            st.info("Aucun scaler trouvé à télécharger.")

    st.markdown("""
    ---
    **Bonnes pratiques** :
    - Conservez exactement les **mêmes colonnes** et **le même ordre** qu'à l'entraînement lors de la prédiction.
    - Appliquez **le même scaler** (StandardScaler) appris sur les données d'entraînement.
    - Mettez à jour régulièrement le modèle si les conditions d'exploitation changent (nouvelles typologies de bâtiments, nouvelles normes, etc.).
    """)
