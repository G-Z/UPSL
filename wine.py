import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics Dashboard",
    layout="wide",
    page_icon="🍷"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    wine = pd.read_csv("winequality-red.csv")
    pairings = pd.read_csv("wine_food_pairings.csv")
    return wine, pairings

wine, pairings = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🍷 Nawigacja")
page = st.sidebar.radio(
    "Wybierz sekcję",
    [
        "Dashboard",
        "Analiza jakości wina",
        "Korelacje",
        "PCA",
        "Food Pairing Explorer",
        "Rekomendacje"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    st.title("📊 Wine Analytics Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Liczba win", wine.shape[0])
    col2.metric("Średnia jakość", round(wine["quality"].mean(), 2))
    col3.metric("Liczba pairingów", pairings.shape[0])

    st.divider()

    fig = px.histogram(
        wine,
        x="quality",
        color="quality",
        title="Rozkład jakości wina"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        wine,
        x="quality",
        y="alcohol",
        title="Zawartość alkoholu vs jakość wina"
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# QUALITY ANALYSIS
# --------------------------------------------------
elif page == "Analiza jakości wina":
    st.title("🍇 Analiza cech chemicznych")

    feature = st.selectbox(
        "Wybierz cechę",
        wine.drop(columns="quality").columns
    )

    fig = px.scatter(
        wine,
        x=feature,
        y="quality",
        title=f"{feature} vs jakość wina"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.violin(
        wine,
        x="quality",
        y=feature,
        box=True,
        title=f"Rozkład cechy: {feature}"
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# CORRELATIONS
# --------------------------------------------------
elif page == "Korelacje":
    st.title("🔗 Korelacje cech chemicznych")

    corr = wine.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Macierz korelacji"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PCA
# --------------------------------------------------
elif page == "PCA":
    st.title("📉 Analiza PCA")

    X = wine.drop(columns="quality")
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"]
    )
    pca_df["quality"] = wine["quality"]

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="quality",
        title="PCA – redukcja wymiarów"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Wyjaśniona wariancja: "
        f"{round(pca.explained_variance_ratio_[0]*100,1)}% + "
        f"{round(pca.explained_variance_ratio_[1]*100,1)}%"
    )

# --------------------------------------------------
# FOOD PAIRING
# --------------------------------------------------
elif page == "Food Pairing Explorer":
    st.title("🍽️ Food Pairing Explorer")

    cuisine = st.multiselect(
        "Wybierz kuchnię",
        pairings["cuisine"].unique(),
        default=pairings["cuisine"].unique()[:2]
    )

    min_quality = st.slider(
        "Minimalna jakość pairingu",
        1, 5, 3
    )

    filtered = pairings[
        (pairings["cuisine"].isin(cuisine)) &
        (pairings["pairing_quality"] >= min_quality)
    ]

    fig = px.bar(
        filtered,
        x="wine_type",
        color="quality_label",
        title="Jakość pairingów wg typu wina"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered.head(50), use_container_width=True)

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------
elif page == "Rekomendacje":
    st.title("🤖 Rekomendacje wino–jedzenie")

    wine_type = st.selectbox(
        "Wybierz typ wina",
        pairings["wine_type"].unique()
    )

    recommendations = (
        pairings[pairings["wine_type"] == wine_type]
        .sort_values("pairing_quality", ascending=False)
        .head(10)
    )

    for _, row in recommendations.iterrows():
        st.success(
            f"""
            🍷 **{row['wine_type']}**
            🌍 Kuchnia: **{row['cuisine']}**
            ⭐ Jakość: **{row['quality_label']}**
            📝 {row['description']}
            """
        )