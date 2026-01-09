import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍷"
)

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown(
    "Kompleksowa aplikacja do analizy jakości czerwonych win, "
    "modelowania ML oraz rekomendacji parowania wina z jedzeniem."
)

# =========================================================
# HELPERS
# =========================================================
def safe_col(df, col):
    return col if col in df.columns else None

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_wine_quality():
    return pd.read_csv("winequality-red.csv")

@st.cache_data
def load_pairings():
    return pd.read_csv("wine_food_pairings.csv")

wine_error, pairing_error = None, None
try:
    wine_df = load_wine_quality()
except Exception as e:
    wine_error = str(e)
    wine_df = None

try:
    pairings_df = load_pairings()
except Exception as e:
    pairing_error = str(e)
    pairings_df = None

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Moduły")
module = st.sidebar.radio(
    "Wybierz moduł:",
    [
        "Dashboard",
        "Analiza jakości wina",
        "Model ML & Predykcja",
        "PCA",
        "Food Pairing Explorer",
        "Rekomendacje"
    ]
)

# =========================================================
# DASHBOARD
# =========================================================
if module == "Dashboard":
    st.subheader("📊 Dashboard")

    if wine_df is None:
        st.error(wine_error)
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Liczba win", wine_df.shape[0])
    col2.metric("Średnia jakość", round(wine_df["quality"].mean(), 2))
    if pairings_df is not None:
        col3.metric("Liczba pairingów", pairings_df.shape[0])

    fig, ax = plt.subplots()
    ax.hist(wine_df["quality"], bins=range(3, 10), edgecolor="black")
    ax.set_title("Rozkład jakości wina")
    ax.set_xlabel("Quality")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# =========================================================
# ANALIZA JAKOŚCI
# =========================================================
elif module == "Analiza jakości wina":
    st.subheader("🍇 Analiza jakości wina")

    if wine_df is None:
        st.error(wine_error)
        st.stop()

    df = wine_df.copy()

    st.dataframe(df.head())

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title("Macierz korelacji")
    st.pyplot(fig)

    feature = st.selectbox(
        "Wybierz cechę:",
        [c for c in df.columns if c != "quality"]
    )

    fig2, ax2 = plt.subplots()
    ax2.scatter(df[feature], df["quality"], alpha=0.6)
    ax2.set_xlabel(feature)
    ax2.set_ylabel("quality")
    st.pyplot(fig2)

# =========================================================
# MODEL ML
# =========================================================
elif module == "Model ML & Predykcja":
    st.subheader("🤖 Model ML – Random Forest")

    if wine_df is None:
        st.error(wine_error)
        st.stop()

    X = wine_df.drop("quality", axis=1)
    y = wine_df["quality"]

    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    n_estimators = st.slider("Liczba drzew", 50, 500, 200, step=50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    st.subheader("Ważność cech")
    st.bar_chart(importances)

    st.subheader("🔮 Predykcja jakości")
    user_input = {}
    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        c = cols[i % 3]
        user_input[col] = c.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

    pred = model.predict(pd.DataFrame([user_input]))[0]
    st.success(f"Przewidywana jakość: **{pred:.2f}**")

# =========================================================
# PCA
# =========================================================
elif module == "PCA":
    st.subheader("📉 PCA")

    if wine_df is None:
        st.error(wine_error)
        st.stop()

    X = wine_df.drop("quality", axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["quality"] = wine_df["quality"]

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["quality"],
        cmap="viridis"
    )
    plt.colorbar(scatter, ax=ax)
    ax.set_title("PCA – redukcja wymiarów")
    st.pyplot(fig)

# =========================================================
# FOOD PAIRING
# =========================================================
elif module == "Food Pairing Explorer":
    st.subheader("🍽️ Food Pairing Explorer")

    if pairings_df is None:
        st.error(pairing_error)
        st.stop()

    dfp = pairings_df.copy()
    st.dataframe(dfp.head())

    wine_sel = st.multiselect(
        "Wine type",
        sorted(dfp["wine_type"].unique()),
        default=[]
    )

    cuisine_sel = st.multiselect(
        "Cuisine",
        sorted(dfp["cuisine"].unique()),
        default=[]
    )

    min_quality = st.slider(
        "Minimalna jakość parowania",
        int(dfp["pairing_quality"].min()),
        int(dfp["pairing_quality"].max()),
        3
    )

    filt = dfp[dfp["pairing_quality"] >= min_quality]
    if wine_sel:
        filt = filt[filt["wine_type"].isin(wine_sel)]
    if cuisine_sel:
        filt = filt[filt["cuisine"].isin(cuisine_sel)]

    st.dataframe(
        filt.sort_values("pairing_quality", ascending=False).head(50)
    )

# =========================================================
# REKOMENDACJE
# =========================================================
elif module == "Rekomendacje":
    st.subheader("⭐ Rekomendacje")

    if pairings_df is None:
        st.error(pairing_error)
        st.stop()

    wine_type = st.selectbox(
        "Wybierz typ wina",
        sorted(pairings_df["wine_type"].unique())
    )

    top = (
        pairings_df[pairings_df["wine_type"] == wine_type]
        .sort_values("pairing_quality", ascending=False)
        .head(10)
    )

    for _, row in top.iterrows():
        st.success(
            f"""
            🍷 **{row['wine_type']}**  
            🌍 {row['cuisine']}  
            ⭐ {row['quality_label']}  
            📝 {row['description']}
            """
        )