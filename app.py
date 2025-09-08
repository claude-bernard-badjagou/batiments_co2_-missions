"""
Application Streamlit : Prédicteur de CO₂ pour bâtiments (monofichier, Streamlit Cloud)
-----------------------------------------------------------------------------

Correctifs intégrés (version finale) :
- Champ max_depth : min_value=0 (0 = None), valeur par défaut = 1 (évite le crash).
- Remplacement des use_container_width=True par width="stretch" pour st.dataframe.
- RMSE calculée comme sqrt(MSE) (au lieu de squared=False).
- Nuage de points : passage à Altair avec sécurisation des noms de colonnes et encodage explicite (quantitative).
- Variables/fonctions en français (sans accents), commentaires explicatifs.

Ressources partagées attendues à la racine :
- Données : df_resultat_analyse_batiments.csv (facultatif : upload possible dans l'UI)
- Modèle : model_co2.pkl (sera créé si absent)
- Scaler : scaler_co2.pkl (sera créé si absent)
"""

# 1) --- Importations ---
import os                                   # Chargeons os pour tester la présence des ressources (CSV, .pkl)
import io                                   # Chargeons io pour gérer des flux mémoire (téléversement, téléchargements)
import joblib                               # Chargeons joblib pour sérialiser/désérialiser modèle et scaler
import numpy as np                          # Chargeons numpy pour les tableaux et calculs numériques
import pandas as pd                         # Chargeons pandas pour charger et manipuler les données tabulaires
import streamlit as st                      # Chargeons Streamlit pour construire l'interface web
import altair as alt                        # Chargeons Altair pour tracer des graphiques robustes
from typing import Tuple, Optional, List    # Chargeons des types pour annoter proprement nos fonctions

# scikit-learn : prétraitement, modèle, métriques
from sklearn.model_selection import train_test_split   # Pour former un train/test honnête
from sklearn.preprocessing import StandardScaler       # Pour standardiser les variables explicatives
from sklearn.ensemble import RandomForestRegressor     # Modèle robuste non linéaire
from sklearn.metrics import r2_score, mean_squared_error  # R² et MSE (RMSE via sqrt)

# 2) --- Constantes (ressources partagées) ---
CHEMIN_DONNEES_PAR_DEFAUT = "df_resultat_analyse_batiments.csv"   # CSV attendu à la racine du repo
CHEMIN_MODELE = "model_co2.pkl"                                   # Fichier du modèle
CHEMIN_SCALER = "scaler_co2.pkl"                                  # Fichier du scaler

# 3) --- Schéma des variables par défaut ---
COLONNES_FEATURES_PAR_DEFAUT: List[str] = [
    "Electricity(kBtu)",                 # Consommation électrique (kBtu)
    "NaturalGas(kBtu)",                  # Consommation gaz (kBtu)
    "BuildingAge",                       # Âge du bâtiment
    "Prop_LargestPropertyUseTypeGFA"     # Proportion de la surface principale
]
COLONNE_CIBLE_PAR_DEFAUT = "TotalGHGEmissions"                    # Émissions de GES totales (cible)

# 4) --- Configuration de la page Streamlit ---
st.set_page_config(                           # Configurons la page (large), titre et icône
    page_title="Prédicteur CO₂ des bâtiments",
    page_icon="🌍",
    layout="wide"
)

# 5) --- Fonctions utilitaires ---
@st.cache_data(show_spinner=False)
def charger_donnees(chemin: Optional[str] = None,
                    fichier_televerse: Optional[io.BytesIO] = None) -> Optional[pd.DataFrame]:
    """Chargeons le CSV : d'abord l'upload utilisateur, sinon le fichier local par défaut."""
    # Si l'utilisateur a téléversé un fichier via l'interface, nous l'utilisons en priorité
    if fichier_televerse is not None:
        try:
            df = pd.read_csv(fichier_televerse)     # Lisons le CSV téléversé
            return df                               # Renvoyons le DataFrame
        except Exception as e:
            st.error(f"Erreur de lecture du CSV téléversé : {e}")
            return None

    # Sinon, tentons de lire depuis le chemin fourni ou celui par défaut
    chemin_effectif = chemin or CHEMIN_DONNEES_PAR_DEFAUT
    if os.path.exists(chemin_effectif):
        try:
            df = pd.read_csv(chemin_effectif)       # Lisons le CSV local si présent
            return df
        except Exception as e:
            st.error(f"Erreur de lecture de '{chemin_effectif}' : {e}")
            return None
    return None


def preparer_X_y(df: pd.DataFrame,
                 colonnes_features: List[str],
                 colonne_cible: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Préparons X et y : vérifions les colonnes, forçons en numérique, filtrons les NaN."""
    manquantes = [c for c in colonnes_features if c not in df.columns]  # Colonnes explicatives manquantes
    if manquantes:
        raise ValueError(f"Colonnes explicatives manquantes : {manquantes}")

    if colonne_cible not in df.columns:                                  # Colonne cible manquante
        raise ValueError(f"Colonne cible manquante : {colonne_cible}")

    X = df[colonnes_features].copy()                                     # Copions X
    y = df[colonne_cible].copy()                                         # Copions y

    X = X.apply(pd.to_numeric, errors="coerce")                          # Forçons numérique (coerce → NaN)
    y = pd.to_numeric(y, errors="coerce")                                # Forçons y en numérique

    masque_valide = X.notna().all(axis=1) & y.notna()                    # Lignes complètes
    X = X[masque_valide]
    y = y[masque_valide]
    return X, y


@st.cache_resource(show_spinner=False)
def charger_ou_entrainer_modele(df: Optional[pd.DataFrame],
                                colonnes_features: List[str] = COLONNES_FEATURES_PAR_DEFAUT,
                                colonne_cible: str = COLONNE_CIBLE_PAR_DEFAUT,
                                n_estimateurs: int = 300,
                                profondeur_max: Optional[int] = None,
                                graine: int = 42):
    """Chargeons modèle & scaler si disponibles; sinon entraînons et sauvegardons."""
    # 1) Tentons le chargement des artefacts existants
    if os.path.exists(CHEMIN_MODELE) and os.path.exists(CHEMIN_SCALER):
        try:
            modele = joblib.load(CHEMIN_MODELE)      # Chargeons le modèle .pkl
            scaler = joblib.load(CHEMIN_SCALER)      # Chargeons le scaler .pkl
            return modele, scaler, None              # Pas de métriques (pas de réentraînement)
        except Exception as e:
            st.warning(f"Impossible de charger les artefacts existants : {e}. Entraînement prévu.")

    # 2) Si pas d'artefacts et pas de données, impossible d'entraîner
    if df is None:
        return None, None, None

    # 3) Préparons X/y
    X, y = preparer_X_y(df, colonnes_features, colonne_cible)

    # 4) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=graine)

    # 5) Standardisation
    scaler = StandardScaler()                        # Instancions un scaler standard
    X_train_s = scaler.fit_transform(X_train)        # Fit sur train + transform
    X_test_s = scaler.transform(X_test)              # Transform sur test

    # 6) Modèle Random Forest
    modele = RandomForestRegressor(
        n_estimators=n_estimateurs,
        max_depth=profondeur_max,
        random_state=graine,
        n_jobs=-1
    )
    modele.fit(X_train_s, y_train)                   # Entraînons le modèle

    # 7) Évaluation (R², RMSE robuste)
    pred_test = modele.predict(X_test_s)             # Prédictions de test
    r2 = r2_score(y_test, pred_test)                 # R²
    mse = mean_squared_error(y_test, pred_test)      # MSE (sans arg nommé)
    rmse = float(np.sqrt(mse))                       # RMSE via racine carrée

    # 8) Sauvegarde des artefacts
    joblib.dump(modele, CHEMIN_MODELE)               # Sauvegardons le modèle
    joblib.dump(scaler, CHEMIN_SCALER)               # Sauvegardons le scaler

    return modele, scaler, {"r2": r2, "rmse": rmse}  # Renvoyons objets + métriques


def predire_co2(modele: RandomForestRegressor,
                scaler: StandardScaler,
                tableau_features: np.ndarray) -> float:
    """Appliquons le scaler puis le modèle pour prédire la valeur de CO₂."""
    donnees_mises_echelle = scaler.transform(tableau_features)   # Transformons avec le scaler appris
    prediction = modele.predict(donnees_mises_echelle)           # Produisons la prédiction
    return float(prediction[0])                                  # Convertissons en float natif

# 6) --- Interface : barre d'onglets (navigation horizontale) ---
onglet_accueil, onglet_exploration, onglet_modele, onglet_prediction, onglet_api = st.tabs([
    "🏠 Accueil",
    "📊 Exploration",
    "🔧 Pretraitement & Modele",
    "🤖 Prediction",
    "📡 API & Export"
])

# 7) --- Barre latérale : téléversement de données ---
with st.sidebar:
    st.markdown("### 📥 Données")
    fichier_csv_televerse = st.file_uploader(
        "Téléverser df_resultat_analyse_batiments.csv (optionnel)", type=["csv"]
    )

# 8) --- Chargement global des données ---
df_global = charger_donnees(CHEMIN_DONNEES_PAR_DEFAUT, fichier_csv_televerse)

# 9) --- Accueil ---
with onglet_accueil:
    st.markdown(
        """
        # 🌍 Prédicteur de CO₂ des bâtiments

        **But** : estimer rapidement et de manière fiable les **émissions de CO₂** d'un bâtiment
        à partir de variables simples (électricité, gaz, âge, proportion de surface principale).

        **Ressources partagées** :
        - Données : `df_resultat_analyse_batiments.csv`
        - Modèle : `model_co2.pkl`
        - Scaler : `scaler_co2.pkl`
        """
    )
    if df_global is None:
        st.info("Aucune donnée locale détectée. Téléversez le CSV dans la barre latérale ou placez-le à la racine.")
    else:
        st.success("Données chargées avec succès.")

# 10) --- Exploration ---
with onglet_exploration:
    st.markdown("## 📊 Exploration des données")
    if df_global is None:
        st.warning("Exploration indisponible : aucune donnée chargée.")
    else:
        st.markdown("### Aperçu")
        st.dataframe(df_global.head(20), width="stretch")                     # width='stretch'

        st.markdown("### Statistiques descriptives (numériques)")
        st.dataframe(df_global.describe(include="number"), width="stretch")   # width='stretch'

        st.markdown("### Valeurs manquantes par colonne")
        manquants = df_global.isna().sum().sort_values(ascending=False)
        st.dataframe(manquants.to_frame("nb_nan"), width="stretch")           # width='stretch'

        colonnes_numeriques = df_global.select_dtypes(include="number").columns.tolist()
        if colonnes_numeriques:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Histogramme rapide")
                col_hist = st.selectbox("Sélectionnez une colonne numérique", colonnes_numeriques)
                try:
                    valeurs = df_global[col_hist].dropna().values
                    if len(valeurs) > 0:
                        nb_bins = st.slider("Nombre de classes (bins)", 10, 100, 30)
                        histo, bords = np.histogram(valeurs, bins=nb_bins)
                        df_histo = pd.DataFrame({"classe": bords[:-1], "frequence": histo}).set_index("classe")
                        st.bar_chart(df_histo)
                    else:
                        st.info("Aucune valeur disponible pour tracer l'histogramme.")
                except Exception as e:
                    st.error(f"Erreur histogramme : {e}")

            with c2:
                st.markdown("#### Nuage de points (x vs y)")
                if len(colonnes_numeriques) >= 2:
                    # 1) Sélection des colonnes
                    x_col = st.selectbox("Axe X", colonnes_numeriques, index=0)
                    y_col = st.selectbox("Axe Y", colonnes_numeriques, index=1)
                    try:
                        # 2) Sous-dataframe propre (dropna et forçage numérique)
                        df_xy = df_global[[x_col, y_col]].copy()
                        df_xy = df_xy.apply(pd.to_numeric, errors="coerce").dropna()

                        # 3) Sécurisation des noms (strip et remplacement des ':')
                        def _secure(name: str) -> str:
                            name_stripped = str(name).strip()
                            return name_stripped.replace(":", "_")
                        x_safe = _secure(x_col)
                        y_safe = _secure(y_col)
                        if x_safe != x_col or y_safe != y_col:
                            df_xy = df_xy.rename(columns={x_col: x_safe, y_col: y_safe})

                        # 4) Chart Altair explicite (quantitative)
                        chart = (
                            alt.Chart(df_xy)
                            .mark_point()
                            .encode(
                                x=alt.X(x_safe, type="quantitative"),
                                y=alt.Y(y_safe, type="quantitative"),
                                tooltip=[x_safe, y_safe]
                            )
                            .interactive()
                        )
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur nuage de points : {e}")
                else:
                    st.info("Nuage de points indisponible : moins de deux colonnes numériques.")

        if len(colonnes_numeriques) >= 2:
            st.markdown("### Matrice de corrélation (numérique)")
            try:
                corr = df_global[colonnes_numeriques].corr()
                st.dataframe(corr, width="stretch")                           # width='stretch'
            except Exception as e:
                st.error(f"Erreur corrélation : {e}")

# 11) --- Prétraitement & Modèle ---
with onglet_modele:
    st.markdown("## 🔧 Prétraitement & Modèle")
    if df_global is None:
        st.warning("Prétraitement indisponible : aucune donnée chargée.")
    else:
        st.markdown("### Paramètres")
        colonnes_proposees = [c for c in COLONNES_FEATURES_PAR_DEFAUT if c in df_global.columns]
        if len(colonnes_proposees) < len(COLONNES_FEATURES_PAR_DEFAUT):
            st.warning("Certaines colonnes par défaut manquent. Adaptez la sélection si nécessaire.")

        colonnes_features = st.multiselect("Colonnes explicatives",
                                           options=df_global.columns.tolist(),
                                           default=colonnes_proposees)

        colonne_cible = st.selectbox("Colonne cible (émissions de GES)",
                                     options=df_global.columns.tolist(),
                                     index=df_global.columns.get_loc(COLONNE_CIBLE_PAR_DEFAUT)
                                     if COLONNE_CIBLE_PAR_DEFAUT in df_global.columns else 0)

        c1, c2, c3 = st.columns(3)
        with c1:
            n_est = st.number_input("n_estimators (arbres)", min_value=50, max_value=1000, value=300, step=50)
        with c2:
            # ✅ Correctif Cloud : autoriser 0 et mettre la valeur par défaut à 1 (l'utilisateur peut mettre 0=None)
            profondeur_max = st.number_input(
                "max_depth (None = automatique)",
                min_value=0, max_value=100, value=1, step=1,
                help="0 = None (illimité)"
            )
            profondeur_max = None if profondeur_max == 0 else int(profondeur_max)
        with c3:
            graine = st.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

        st.markdown("### Entraînement / Chargement")
        lancer_reentrainement = st.checkbox("Réentraîner maintenant (écrase les artefacts existants)")

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
            modele, scaler, metriques = charger_ou_entrainer_modele(
                df_global, colonnes_features, colonne_cible,
                n_estimateurs=n_est, profondeur_max=profondeur_max, graine=graine
            )
            if metriques:
                st.info(f"Modèle entraîné (artefacts absents). R² = {metriques['r2']:.4f} | RMSE = {metriques['rmse']:.4f}")

        if isinstance(modele, RandomForestRegressor) and len(colonnes_features) == len(modele.feature_importances_):
            st.markdown("### Importance des variables (Random Forest)")
            importances = pd.Series(modele.feature_importances_, index=colonnes_features).sort_values(ascending=False)
            st.bar_chart(importances)
        else:
            st.info("Importances non disponibles.")

# 12) --- Prédiction ---
with onglet_prediction:
    st.markdown("## 🤖 Prédiction interactive des émissions de CO₂")
    modele_actuel, scaler_actuel, _ = charger_ou_entrainer_modele(df_global)
    if (modele_actuel is None) or (scaler_actuel is None):
        st.warning("Aucun modèle/scaler disponible. Entraînez/chargez dans l'onglet 'Pretraitement & Modele'.")
    else:
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
                vecteur = np.array([[conso_electricite, conso_gaz, age_batiment, prop_surface]], dtype=float)
                valeur_predite = predire_co2(modele_actuel, scaler_actuel, vecteur)
                st.success(f"Émissions de CO₂ estimées : {valeur_predite:.2f} (unités de la cible)")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

# 13) --- API & Export ---
with onglet_api:
    st.markdown("## 📡 API & Export des artefacts")
    st.markdown(
        """
        ### Point d'API prévu (service séparé FastAPI)
        POST `/predict` avec :
        {
          "electricity": 5000.0,
          "gas": 1000.0,
          "age": 30.0,
          "prop_gfa": 0.6
        }
        Réponse :
        { "prediction_CO2": 123.45 }
        """
    )

    st.markdown("### Téléchargement des artefacts")
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(CHEMIN_MODELE):
            try:
                with open(CHEMIN_MODELE, "rb") as f:
                    b = f.read()
                st.download_button("Télécharger model_co2.pkl", b, file_name="model_co2.pkl")
            except Exception as e:
                st.error(f"Téléchargement modèle impossible : {e}")
        else:
            st.info("Aucun modèle trouvé à télécharger.")
    with col_b:
        if os.path.exists(CHEMIN_SCALER):
            try:
                with open(CHEMIN_SCALER, "rb") as f:
                    b = f.read()
                st.download_button("Télécharger scaler_co2.pkl", b, file_name="scaler_co2.pkl")
            except Exception as e:
                st.error(f"Téléchargement scaler impossible : {e}")
        else:
            st.info("Aucun scaler trouvé à télécharger.")

    st.markdown(
        """
        ---
        **Bonnes pratiques** :
        - Conservez les **mêmes colonnes** et le **même ordre** qu'à l'entraînement lors de la prédiction.
        - Appliquez **le même scaler** (StandardScaler) appris sur les données d'entraînement.
        - Réentraînez périodiquement si vos données réelles évoluent.
        """
    )
