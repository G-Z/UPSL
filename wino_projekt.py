import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =========================================================
# KONFIGURACJA STRONY
# =========================================================
st.set_page_config(
    page_title="Wine Analytics – Hurtownik → Restaurator",
    layout="wide",
    page_icon="🍷"
)

st.title("🍷 Wine Analytics – narzędzie dla hurtownika wina")
st.caption(
    "Analiza jakości win oraz rekomendacje food pairing "
    "wspierające sprzedaż dla restauratorów."
)

# =========================================================
# WCZYTYWANIE DANYCH
# =========================================================
@st.cache_data
def load_data():
    wine = pd.read_csv("winequality-red.csv")
    pairings = pd.read_csv("wine_food_pairings.csv")
    return wine, pairings

wine_df, pairings_df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
section = st.sidebar.radio(
    "Wybierz sekcję:",
    [
        "Eksploracja danych",
        "Filtrowanie oferty",
        "Rozkłady i porównania",
        "Wizualizacja 3D",
        "Wnioski biznesowe"
    ]
)

# =========================================================
# 1. EKSPLORACJA DANYCH
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
            st.write("Braki danych:")
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
# 2. FILTROWANIE
# =========================================================
elif section == "Filtrowanie oferty":
    st.header("🔎 Filtrowanie oferty hurtownika")

    tab1, tab2 = st.tabs(["🍷 Wina", "🍽️ Pairingi"])

    # -------- WINA --------
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

        st.success(f"Liczba win po filtrze: {filt.shape[0]}")
        st.dataframe(filt.head(20))
        st.write(
            filt[[feature, "quality"]]
            .describe()
            .loc[["mean", "min", "max"]]
        )

    # -------- PAIRINGI --------
    with tab2:
        wine_type = st.multiselect(
            "Typ wina:",
            sorted(pairings_df.wine_type.unique())
        )
        food_cat = st.multiselect(
            "Kategoria jedzenia:",
            sorted(pairings_df.food_category.unique())
        )
        cuisine = st.multiselect(
            "Kuchnia:",
            sorted(pairings_df.cuisine.unique())
        )
        min_q = st.slider("Minimalna jakość parowania:", 1, 5, 3)

        filt = pairings_df[pairings_df.pairing_quality >= min_q]
        if wine_type:
            filt = filt[filt.wine_type.isin(wine_type)]
        if food_cat:
            filt = filt[filt.food_category.isin(food_cat)]
        if cuisine:
            filt = filt[filt.cuisine.isin(cuisine)]

        st.success(f"Liczba rekomendacji: {filt.shape[0]}")
        st.dataframe(filt.head(30))

# =========================================================
# 3. ROZKŁADY I PORÓWNANIA
# =========================================================
elif section == "Rozkłady i porównania":
    st.header("📈 Rozkłady i porównania cech wina")

    feature = st.selectbox(
        "Wybierz cechę:",
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
# 4. WIZUALIZACJA 3D – PLOTLY
# =========================================================
elif section == "Wizualizacja 3D":
    st.header("🧊 Profile win – interaktywna wizualizacja 3D")

    st.write(
        "Wizualizacja pozwala hurtownikowi zobaczyć "
        "segmentację win według profilu chemicznego i jakości."
    )

    cols = wine_df.columns.drop("quality").tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.selectbox("Oś X", cols, index=cols.index("alcohol"))
    with c2:
        y = st.selectbox("Oś Y", cols, index=cols.index("volatile acidity"))
    with c3:
        z = st.selectbox("Oś Z", cols, index=cols.index("sulphates"))

    sample_df = wine_df.sample(n=800, random_state=42)

    fig = px.scatter_3d(
        sample_df,
        x=x,
        y=y,
        z=z,
        color="quality",
        opacity=0.7,
        height=600,
        title="Profil chemiczny win a jakość"
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text="Quality"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 5. WNIOSKI
# =========================================================
elif section == "Wnioski biznesowe":
    st.header("📌 Wnioski dla hurtownika")

    st.markdown(
        """
        - Dane pozwalają segmentować ofertę win według jakości i profilu chemicznego
        - Hurtownik może proponować wina dopasowane do stylu restauracji
        - Food pairing wspiera sprzedaż opartą na rekomendacjach
        - Analiza danych zwiększa trafność decyzji handlowych
        """
    )

    st.success("Aplikacja spełnia rolę narzędzia decyzyjnego dla hurtownika wina.")