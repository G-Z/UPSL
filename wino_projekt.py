import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Wine Analytics – Hurtownik → Restaurator",
    layout="wide",
    page_icon="🍷"
)

st.title("🍷 Wine Analytics – narzędzie dla hurtownika wina")
st.caption(
    "Aplikacja wspierająca dobór win i rekomendacji food pairing "
    "dla restauratorów na podstawie danych."
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return (
        pd.read_csv("winequality-red.csv"),
        pd.read_csv("wine_food_pairings.csv")
    )

wine_df, pairings_df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
section = st.sidebar.radio(
    "Sekcja:",
    [
        "Eksploracja danych",
        "Filtrowanie oferty",
        "Rozkłady i porównania",
        "Wizualizacja 3D",
        "Wnioski biznesowe"
    ]
)

# =========================================================
# 1️⃣ EKSPLORACJA DANYCH
# =========================================================
if section == "Eksploracja danych":
    st.header("📊 Podstawowa eksploracja danych")

    tab1, tab2 = st.tabs(["🍷 Jakość wina", "🍽️ Food pairing"])

    with tab1:
        st.subheader("winequality-red.csv")
        st.dataframe(wine_df.head())

        c1, c2, c3 = st.columns(3)
        c1.metric("Wiersze", wine_df.shape[0])
        c2.metric("Kolumny", wine_df.shape[1])
        c3.metric("Duplikaty", wine_df.duplicated().sum())

        with st.expander("Braki danych i typy"):
            st.write("Braki:")
            st.write(wine_df.isnull().sum())
            st.write("Typy danych:")
            st.write(wine_df.dtypes)

    with tab2:
        st.subheader("wine_food_pairings.csv")
        st.dataframe(pairings_df.head())

        c1, c2, c3 = st.columns(3)
        c1.metric("Wiersze", pairings_df.shape[0])
        c2.metric("Kolumny", pairings_df.shape[1])
        c3.metric("Duplikaty", pairings_df.duplicated().sum())

        with st.expander("Braki danych i typy"):
            st.write(pairings_df.isnull().sum())
            st.write(pairings_df.dtypes)

# =========================================================
# 2️⃣ FILTROWANIE
# =========================================================
elif section == "Filtrowanie oferty":
    st.header("🔎 Filtrowanie oferty hurtownika")

    tab1, tab2 = st.tabs(["🍷 Wina", "🍽️ Pairingi"])

    with tab1:
        q_min, q_max = st.slider(
            "Zakres jakości:",
            int(wine_df.quality.min()),
            int(wine_df.quality.max()),
            (5, 7)
        )

        feature = st.selectbox(
            "Cecha:",
            [c for c in wine_df.columns if c != "quality"]
        )

        f_min, f_max = st.slider(
            "Zakres cechy:",
            float(wine_df[feature].min()),
            float(wine_df[feature].max()),
            (
                float(wine_df[feature].min()),
                float(wine_df[feature].max())
            )
        )

        filt = wine_df[
            wine_df.quality.between(q_min, q_max)
            & wine_df[feature].between(f_min, f_max)
        ]

        st.success(f"Liczba win: {filt.shape[0]}")
        st.dataframe(filt.head(20))
        st.write(filt[[feature, "quality"]].describe().loc[["mean", "min", "max"]])

    with tab2:
        wine_type = st.multiselect(
            "Typ wina:",
            sorted(pairings_df.wine_type.unique())
        )
        cuisine = st.multiselect(
            "Kuchnia:",
            sorted(pairings_df.cuisine.unique())
        )
        min_q = st.slider("Minimalna jakość:", 1, 5, 3)

        filt = pairings_df[pairings_df.pairing_quality >= min_q]
        if wine_type:
            filt = filt[filt.wine_type.isin(wine_type)]
        if cuisine:
            filt = filt[filt.cuisine.isin(cuisine)]

        st.success(f"Liczba rekomendacji: {filt.shape[0]}")
        st.dataframe(filt.head(30))

# =========================================================
# 3️⃣ ROZKŁADY I PORÓWNANIA
# =========================================================
elif section == "Rozkłady i porównania":
    st.header("📈 Rozkłady i porównania jakości")

    feature = st.selectbox(
        "Cecha:",
        [c for c in wine_df.columns if c != "quality"],
        index=wine_df.columns.get_loc("alcohol") - 1
    )

    split = st.slider("Granica jakości:", 3, 8, 6)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(wine_df[feature], bins=30)
        ax.set_title("Histogram")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(y=wine_df[feature], ax=ax)
        ax.set_title("Boxplot")
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        wine_df[wine_df.quality <= split][feature],
        alpha=0.6,
        label=f"quality ≤ {split}"
    )
    ax.hist(
        wine_df[wine_df.quality > split][feature],
        alpha=0.6,
        label=f"quality > {split}"
    )
    ax.legend()
    ax.set_title("Porównanie grup jakości")
    st.pyplot(fig)

# =========================================================
# 4️⃣ WYKRES 3D
# =========================================================
elif section == "Wizualizacja 3D":
    st.header("🧊 Profile win – wykres 3D")

    x = st.selectbox("Oś X:", wine_df.columns[:-1], index=10)
    y = st.selectbox("Oś Y:", wine_df.columns[:-1], index=1)
    z = st.selectbox("Oś Z:", wine_df.columns[:-1], index=7)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        wine_df[x],
        wine_df[y],
        wine_df[z],
        c=wine_df.quality,
        cmap="viridis",
        s=20
    )
    fig.colorbar(sc, label="Quality")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    st.pyplot(fig)

# =========================================================
# 5️⃣ WNIOSKI
# =========================================================
elif section == "Wnioski biznesowe":
    st.header("📌 Wnioski dla hurtownika")

    st.markdown(
        """
        - Hurtownik może selekcjonować wina pod konkretne profile restauracji
        - Wina o wyższej jakości mają wyraźnie inny profil chemiczny
        - Dane food pairing pozwalają szybko dobrać ofertę pod kuchnię lokalu
        - Analiza danych wspiera sprzedaż opartą na faktach, nie intuicji
        """
    )

    st.success("Aplikacja spełnia rolę narzędzia decyzyjnego dla hurtownika wina.")