import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Wine Analytics for Wholesalers",
    layout="wide",
    page_icon="🍷"
)

st.title("🍷 Wine Analytics – narzędzie dla hurtownika wina")
st.markdown(
    """
    Aplikacja wspierająca **hurtownika wina** w analizie jakości produktów
    oraz w **proponowaniu odpowiednich win restauratorom**
    na podstawie danych chemicznych i rekomendacji food pairing.
    """
)

# =========================================================
# HELPERS
# =========================================================
def dataset_overview(df):
    col1, col2, col3 = st.columns(3)
    col1.metric("Liczba wierszy", df.shape[0])
    col2.metric("Liczba kolumn", df.shape[1])
    col3.metric("Duplikaty", df.duplicated().sum())

    st.markdown("**Brakujące wartości:**")
    st.write(df.isnull().sum())

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_wine():
    return pd.read_csv("winequality-red.csv")

@st.cache_data
def load_pairings():
    return pd.read_csv("wine_food_pairings.csv")

wine_df = load_wine()
pairings_df = load_pairings()

# =========================================================
# SIDEBAR
# =========================================================
section = st.sidebar.radio(
    "Wybierz sekcję:",
    [
        "1️⃣ Eksploracja – jakość wina",
        "2️⃣ Eksploracja – food pairing",
        "3️⃣ Filtrowanie oferty",
        "4️⃣ Rozkłady i porównania jakości",
        "5️⃣ Analiza 3D profili win",
        "6️⃣ Wnioski dla hurtownika"
    ]
)

# =========================================================
# 1️⃣ EKSPLORACJA – WINEQUALITY
# =========================================================
if section == "1️⃣ Eksploracja – jakość wina":
    st.header("📊 Podstawowa eksploracja danych – jakość wina")

    st.subheader("Podgląd danych")
    st.dataframe(wine_df.head())

    with st.expander("Informacje o datasetcie"):
        dataset_overview(wine_df)
        st.markdown("**Typy danych:**")
        st.write(wine_df.dtypes)

# =========================================================
# 2️⃣ EKSPLORACJA – FOOD PAIRING
# =========================================================
elif section == "2️⃣ Eksploracja – food pairing":
    st.header("🍽️ Podstawowa eksploracja danych – parowanie wina z jedzeniem")

    st.subheader("Podgląd danych")
    st.dataframe(pairings_df.head())

    with st.expander("Informacje o datasetcie"):
        dataset_overview(pairings_df)
        st.markdown("**Typy danych:**")
        st.write(pairings_df.dtypes)

# =========================================================
# 3️⃣ FILTROWANIE I SZYBKIE WNIOSKI
# =========================================================
elif section == "3️⃣ Filtrowanie oferty":
    st.header("🔎 Filtrowanie oferty hurtownika")

    tab1, tab2 = st.tabs(["🍷 Jakość wina", "🍽️ Food pairing"])

    # ---- WINE QUALITY ----
    with tab1:
        st.subheader("Filtrowanie win wg jakości i cech")

        q_min, q_max = st.slider(
            "Zakres jakości (quality):",
            int(wine_df.quality.min()),
            int(wine_df.quality.max()),
            (5, 7)
        )

        feature = st.selectbox(
            "Wybierz cechę:",
            [c for c in wine_df.columns if c != "quality"]
        )

        f_min, f_max = st.slider(
            f"Zakres dla {feature}:",
            float(wine_df[feature].min()),
            float(wine_df[feature].max()),
            (float(wine_df[feature].min()), float(wine_df[feature].max()))
        )

        filt = wine_df[
            (wine_df.quality.between(q_min, q_max)) &
            (wine_df[feature].between(f_min, f_max))
        ]

        st.write(f"**Liczba win spełniających kryteria:** {filt.shape[0]}")
        st.dataframe(filt.head(20))

        st.markdown("**Szybkie statystyki:**")
        st.write(filt[["quality", feature]].describe().loc[["mean", "min", "max"]])

    # ---- PAIRINGS ----
    with tab2:
        st.subheader("Filtrowanie rekomendacji dla restauratora")

        wine_type = st.multiselect(
            "Typ wina:",
            sorted(pairings_df["wine_type"].unique())
        )
        food_cat = st.multiselect(
            "Kategoria jedzenia:",
            sorted(pairings_df["food_category"].unique())
        )
        cuisine = st.multiselect(
            "Kuchnia:",
            sorted(pairings_df["cuisine"].unique())
        )
        min_quality = st.slider(
            "Minimalna jakość parowania:",
            int(pairings_df.pairing_quality.min()),
            int(pairings_df.pairing_quality.max()),
            3
        )

        filt = pairings_df[pairings_df.pairing_quality >= min_quality]
        if wine_type:
            filt = filt[filt.wine_type.isin(wine_type)]
        if food_cat:
            filt = filt[filt.food_category.isin(food_cat)]
        if cuisine:
            filt = filt[filt.cuisine.isin(cuisine)]

        st.write(f"**Liczba rekomendacji:** {filt.shape[0]}")
        st.dataframe(filt.head(30))

        st.markdown("**Średnia jakość parowania:**")
        st.write(filt["pairing_quality"].mean())

# =========================================================
# 4️⃣ ROZKŁADY I PORÓWNANIA
# =========================================================
elif section == "4️⃣ Rozkłady i porównania jakości":
    st.header("📈 Rozkłady cech i porównanie jakości")

    feature = st.selectbox(
        "Wybierz cechę:",
        [c for c in wine_df.columns if c != "quality"],
        index=wine_df.columns.get_loc("alcohol") - 1
    )

    q_split = st.slider("Granica jakości:", 3, 8, 6)

    low = wine_df[wine_df.quality <= q_split]
    high = wine_df[wine_df.quality > q_split]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(wine_df[feature], bins=30)
        ax.set_title("Histogram")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(y=wine_df[feature], ax=ax)
        ax.set_title("Boxplot")
        st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.hist(low[feature], alpha=0.6, label=f"quality ≤ {q_split}")
    ax.hist(high[feature], alpha=0.6, label=f"quality > {q_split}")
    ax.legend()
    ax.set_title("Porównanie rozkładów jakości")
    st.pyplot(fig)

# =========================================================
# 5️⃣ WYKRES 3D
# =========================================================
elif section == "5️⃣ Analiza 3D profili win":
    st.header("🧊 Analiza 3D – profile win")

    x = st.selectbox("Oś X:", wine_df.columns[:-1], index=10)
    y = st.selectbox("Oś Y:", wine_df.columns[:-1], index=1)
    z = st.selectbox("Oś Z:", wine_df.columns[:-1], index=7)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        wine_df[x],
        wine_df[y],
        wine_df[z],
        c=wine_df["quality"],
        cmap="viridis"
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    fig.colorbar(scatter, label="Quality")
    st.pyplot(fig)

# =========================================================
# 6️⃣ WNIOSKI BIZNESOWE
# =========================================================
elif section == "6️⃣ Wnioski dla hurtownika":
    st.header("📌 Wnioski biznesowe")

    st.markdown(
        """
        **Na podstawie przeprowadzonej analizy hurtownik może:**
        - wybierać wina o wyższej jakości dla restauracji premium,
        - dopasowywać profil chemiczny wina do rodzaju kuchni,
        - proponować restauratorom sprawdzone parowania wine–food,
        - ograniczyć ofertę do win najlepiej ocenianych przez dane,
        - budować rekomendacje oparte na danych, nie intuicji.
        """
    )

    st.success(
        "Aplikacja spełnia rolę **narzędzia decyzyjnego** wspierającego sprzedaż hurtową wina."
    )