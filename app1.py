import streamlit as st
import pandas as pd
import os
import plotly.express as px
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import product
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

# Defineeri andmete asukoht
data_dir = "Data"

# Andmete laadimine
@st.cache_data
def load_data():
    hinnastatistika = pd.read_csv(
        os.path.join(data_dir, "Kinnisvara-hinnastatistika-2018-2024.csv"),
        sep=";",
        decimal="."
    )
    kv_thi = pd.read_csv(os.path.join(data_dir, "KV_THI_brutopalk_eluasemehinnaindeks.csv"))
    thi = pd.read_csv(os.path.join(data_dir, "THI_brutopalk_eluasemehinnaindeks.csv"))
    total_data = pd.read_csv(os.path.join(data_dir, "Total_data_annual.csv"))
    return hinnastatistika, kv_thi, thi, total_data

@st.cache_data
def load_geo():
    geo = gpd.read_file("maakonnad.geojson")
    geo["Maakond"] = geo["MNIMI"]  # Veendu, et see veerg eksisteerib
    return geo

# Andmete laadimine
hinnastatistika, kv_thi, thi, total_data = load_data()
geo_df = load_geo()

# 🧼 Andmetöötlus
thi.columns = thi.columns.str.strip().str.lower()
thi['quarter'] = thi['quarter'].str.strip()
thi = thi.set_index('quarter')
thi = thi.sort_index()

# 🎯 Kasutaja valib kvartali
available_quarters = thi.index[thi.index.get_loc("2022Q1"):]  # alates 2022Q1

# 🎯 Teisendusfunktsioon hoone liikidele
def teisenda_hooneliik(kood):
    if not isinstance(kood, str):
        return kood
    if "korter" in kood and "_" in kood:
        osad = kood.split("_")
        if len(osad) == 3:
            return f"{osad[0]} {osad[1]}–{osad[2]}m²"
    return kood.replace("_", " ").capitalize()


# Dashboardi nimi ja üldine selgitus
st.title("Kinnisvara hindade muuutuste analüüsi Dashboard")
st.markdown("""
See rakendus võimaldab uurida kinnisvara hindade muutumise põhjuseid Eestis erinevate vaadete lõikes:
* **Hetkeolukord** – vaata viimase aasta andmeid
* **Muutused ajas** – jälgi trende
* **Prognoosid** – visuaalid tulevikutrendidest
""")

# Vasakpoolne menüü (sidebar)
valik = st.sidebar.radio(
    "Vali vaade:",
    ("Hetkeolukord", "Muutused ajas", "Prognoosid")
)

# Tingimuslik renderdus vastavalt valikule
if valik == "Hetkeolukord":
    st.header("📊 Hetkeolukord")
    st.write("Siin kuvatakse valitud perioodi analüüs.")
    st.markdown("## Hetkeolukord")

    
    selected_q = st.selectbox("Vali kvartal", available_quarters, index=len(available_quarters)-1)

    
    # 🎯 Kasutajasõbralik valik hooneliikide jaoks
    hoone_liigid_raw = sorted(kv_thi["Hoone_liik"].dropna().unique())
    hooneliigid_map = {k: teisenda_hooneliik(k) for k in hoone_liigid_raw}
    reverse_map = {v: k for k, v in hooneliigid_map.items()}
    valikuga = ["Kokku"] + list(hooneliigid_map.values())
    hoonevalik_nice = st.selectbox("Vali hoone liik", valikuga)
    hoonevalik = reverse_map.get(hoonevalik_nice, "Kokku")

    # 📊 Funktsioon muutuste arvutamiseks
    def calc_percent(current, previous):
        if previous == 0 or pd.isna(previous):
            return None
        return round(((current - previous) / previous) * 100, 2)

    def calculate_changes(df, selected_q):
        prev_q = df.index[df.index.get_loc(selected_q) - 1] if df.index.get_loc(selected_q) > 0 else None
        selected_year = int(selected_q[:4])
        selected_quarter = int(selected_q[-1])
        year_start = f"{selected_year - 1}Q4" if selected_quarter == 1 else f"{selected_year}Q1"
        prev_year = str(int(selected_q[:4]) - 1) + selected_q[4:]

        current = df.loc[selected_q] if selected_q in df.index else None
        prev_q_data = df.loc[prev_q] if prev_q and prev_q in df.index else None
        year_start_data = df.loc[year_start] if year_start in df.index else None
        prev_year_data = df.loc[prev_year] if prev_year in df.index else None

        return {
            'thi': {
                'current': round(current['thi'], 2) if current is not None else None,
                'prev_q': calc_percent(current['thi'], prev_q_data['thi']) if current is not None and prev_q_data is not None else None,
                'year_start': calc_percent(current['thi'], year_start_data['thi']) if current is not None and year_start_data is not None else None,
                'prev_year': calc_percent(current['thi'], prev_year_data['thi']) if current is not None and prev_year_data is not None else None,
            },
            'avg_salary': {
                'current': round(current['avg_salary'], 2) if current is not None else None,
                'prev_q': calc_percent(current['avg_salary'], prev_q_data['avg_salary']) if current is not None and prev_q_data is not None else None,
                'year_start': calc_percent(current['avg_salary'], year_start_data['avg_salary']) if current is not None and year_start_data is not None else None,
                'prev_year': calc_percent(current['avg_salary'], prev_year_data['avg_salary']) if current is not None and prev_year_data is not None else None,
            },
            'housing_index': {
                'current': round(current['housing_index'], 2) if current is not None else None,
                'prev_q': calc_percent(current['housing_index'], prev_q_data['housing_index']) if current is not None and prev_q_data is not None else None,
                'year_start': calc_percent(current['housing_index'], year_start_data['housing_index']) if current is not None and year_start_data is not None else None,
                'prev_year': calc_percent(current['housing_index'], prev_year_data['housing_index']) if current is not None and prev_year_data is not None else None,
            }
        }

    # 🎨 Vormindus protsendimuutustele
    def format_change(value):
        if value is None:
            return "–"
        arrow = "🔼" if value > 0 else "🔽" if value < 0 else "➡️"
        color = "green" if value > 0 else "red" if value < 0 else "gray"
        return f":{color}[{arrow} {abs(value)}%]"

    # 👉 Arvuta muutused
    changes = calculate_changes(thi, selected_q)

    # Andmete filtreerimine
    filtered = kv_thi[kv_thi["quarter"] == selected_q]
    if hoonevalik != "Kokku":
        filtered = filtered[filtered["Hoone_liik"] == hoonevalik]
    df_map = filtered.groupby("Maakond", as_index=False)["Keskmine_pinnaühikuhind"].mean()

    # Ühenda ruumiandmetega
    merged = geo_df.merge(df_map, how="left", left_on="Maakond", right_on="Maakond")

    st.markdown(f"### Ülevaade: Mis on praegune seis makromajandusnäitajates? **{selected_q}**")

    # 3 veergu
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Tarbijahinna-indeks (THI)")
        st.markdown(f"### **{changes['thi']['current']}**")
        st.markdown("Muutus eelmise kvartaliga")
        st.markdown(format_change(changes['thi']['prev_q']))
        st.markdown("Muutus aasta algusest")
        st.markdown(format_change(changes['thi']['year_start']))
        st.markdown("Muutus aasta tagusega")
        st.markdown(format_change(changes['thi']['prev_year']))

    with col2:
        st.subheader("Keskmine brutopalk")
        st.markdown(f"### **{changes['avg_salary']['current']}**")
        st.markdown("Muutus eelmise kvartaliga")
        st.markdown(format_change(changes['avg_salary']['prev_q']))
        st.markdown("Muutus aasta algusest")
        st.markdown(format_change(changes['avg_salary']['year_start']))
        st.markdown("Muutus aasta tagusega")
        st.markdown(format_change(changes['avg_salary']['prev_year']))

    with col3:
        st.subheader("Eluaseme hinnaindeks")
        st.markdown(f"### **{changes['housing_index']['current']}**")
        st.markdown("Muutus eelmise kvartaliga")
        st.markdown(format_change(changes['housing_index']['prev_q']))
        st.markdown("Muutus aasta algusest")
        st.markdown(format_change(changes['housing_index']['year_start']))
        st.markdown("Muutus aasta tagusega")
        st.markdown(format_change(changes['housing_index']['prev_year']))

    # 🗺️ Kaardi pealkiri
    hoone_nimi = hoonevalik_nice if hoonevalik != "Kokku" else "Kõik hooneliigid"
    pealkiri_kv = f"{hoone_nimi} keskmine ruutmeetri hind maakonniti, {selected_q}"

    fig = px.choropleth_mapbox(
        merged,
        geojson=merged.geometry.__geo_interface__,
        locations=merged.index,
        color="Keskmine_pinnaühikuhind",
        hover_name="Maakond",
        mapbox_style="carto-positron",
        center={"lat": 58.5953, "lon": 25.0136},
        zoom=5.5,
        opacity=0.7,
        color_continuous_scale="Viridis",
        labels={"Keskmine_pinnaühikuhind": "€/m²"}
    )

    fig.update_layout(
        title_text=pealkiri_kv,
        title_x=0.3,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Veendume, et veerg "Aasta" on olemas ja korrastatud
    aastad = sorted([a for a in total_data["Aasta"].unique() if a >= 2022])

    # Leia vaikimisi valitava aasta indeks, nt 2023, kui see eksisteerib, muidu viimane aasta
    default_year = 2023
    if default_year in aastad:
        default_index = aastad.index(default_year)
    else:
        default_index = len(aastad) - 1  # viimane saadaval olev aasta

    valitud_aasta = st.selectbox("Vali aasta", aastad, index=default_index)

    naidikud = {
        "Netosissetulek": "netosissetulek",
        "Hõivatute arv": "hoivatute_arv",
        "Mitteaktiivsete arv": "mitteaktiivsed",
        "Tööhõive määr": "toohive_maar",
        "Tööjõud ja mitteaktiivsed": "toojoud_ja_mitteaktiivsed",
        "Tööjõu arv": "toojoud_arv",
        "Tööjõus osalemise määr": "toojous_osalemine",
        "Töötuse määr": "tootuse_maar",
        "Töötute arv": "tootute_arv",
        "Leibkondade arv": "leibkondade_arv"
    }

    kuvatav_valik = st.selectbox("Vali näitaja kaardile kuvamiseks", list(naidikud.keys()))
    valitud_naidik = naidikud[kuvatav_valik]

    # Kontrolli, kas eelmise aasta andmed on olemas
    eelmine_aasta = valitud_aasta - 1
    on_eelmine_aasta = eelmine_aasta in total_data["Aasta"].values

    # Valime ainult vastava aasta andmed
    praegune_aasta_df = total_data[total_data["Aasta"] == valitud_aasta]

    # Arvutame kõikide numbriliste veergude keskmised (v.a leibkondade_arv)
    praegune = praegune_aasta_df.drop(columns=["leibkondade_arv"]).mean(numeric_only=True)

    # Leibkondade arv: maakondade keskmine → summa
    leibkondade_summa = praegune_aasta_df.groupby("Maakond")["leibkondade_arv"].mean().sum()*1000
    praegune["leibkondade_arv"] = leibkondade_summa

    # Eelmise aasta andmed
    eelmine = None
    if on_eelmine_aasta:
        eelmine_aasta_df = total_data[total_data["Aasta"] == eelmine_aasta]
        eelmine = eelmine_aasta_df.drop(columns=["leibkondade_arv"]).mean(numeric_only=True)
        leibkondade_summa_eelmine = (
            eelmine_aasta_df.groupby("Maakond")["leibkondade_arv"].mean().sum()*1000
        )
        eelmine["leibkondade_arv"] = leibkondade_summa_eelmine

    # Funktsioon protsendimuutuse jaoks
    def protsendimuutus(nuud, enne):
        if pd.isna(nuud) or pd.isna(enne) or enne == 0:
            return "–"
        muutus = ((nuud - enne) / enne) * 100
        nool = "🔼" if muutus > 0 else "🔽" if muutus < 0 else "➡️"
        värv = "green" if muutus > 0 else "red" if muutus < 0 else "black"
        return f"<span style='color:{värv}'>{nool} {muutus:.1f}%</span>"

    # Valitavad näitajad
    naidikud = {
        "Netosissetulek": "netosissetulek",
        "Tööhõivemäär": "toohive_maar",
        "Töötuse määr": "tootuse_maar",
        "Leibkondade arv": "leibkondade_arv"
    }

    # Kuvame kaardid
    st.markdown(f"## Ülevaade: Mis on praegune seis tööjõu turul? ({valitud_aasta})")
    cols = st.columns(4)

    for idx, (nimi, veeru_nimi) in enumerate(naidikud.items()):
        with cols[idx]:
            st.markdown(f"#### {nimi}")
        
            vaartus = praegune[veeru_nimi]
            if veeru_nimi in ["toohive_maar", "tootuse_maar"]:
                formatted = f"{vaartus:.1f}%"
            else:
                formatted = f"{vaartus:,.0f}"
        
            st.markdown(f"<h2 style='text-align:center'>{formatted}</h2>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Muutus eelmise aastaga:**")
            if eelmine is not None:
                st.markdown(protsendimuutus(praegune[veeru_nimi], eelmine[veeru_nimi]), unsafe_allow_html=True)
            else:
                st.markdown("–")


    # 2. Andmestik, mis sisaldab ainult valitud aasta andmeid
    df_aasta = total_data[total_data["Aasta"] == valitud_aasta]

    # 3. Leia keskmine valitud näitaja väärtus iga maakonna kohta
    maakondade_naidik = df_aasta.groupby("Maakond")[valitud_naidik].mean().reset_index()

    # 4. Ühenda geoandmetega (eeldusel, et sul on 'merged', millel on geojson-objekt ja 'Maakond' veerg)
    kaart_df = merged.merge(maakondade_naidik, on="Maakond", how="left")
    kaart_df = kaart_df.set_index("Maakond")

    # Loome pealkirja dünaamiliselt kasutaja valikute põhjal
    naidiku_nimi = valitud_naidik.replace("_", " ").capitalize()
    pealkiri = f"{naidiku_nimi} maakonniti, {valitud_aasta}"

    fig = px.choropleth_mapbox(
        kaart_df,
        geojson=kaart_df.geometry.__geo_interface__,
        locations=kaart_df.index,
        color=valitud_naidik,
        hover_name=kaart_df.index,
        mapbox_style="carto-positron",
        center={"lat": 58.5953, "lon": 25.0136},
        zoom=5.5,
        opacity=0.7,
        color_continuous_scale="Viridis",
        labels={valitud_naidik: naidiku_nimi}
    )

    # Lisa pealkiri kaardile
    fig.update_layout(
        title_text=pealkiri,
        title_x=0.5,  # Keskenda pealkiri
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    # Kuvame kaardi Streamlitis
    st.plotly_chart(fig, use_container_width=True)

    # Eeldame, et total_data sisaldab veergu "keskmine_ruutmeetrihind"
    valitud_aasta = st.selectbox("Vali aasta korrelatsiooni vaatamiseks", sorted(total_data["Aasta"].unique()))

    df_korr = total_data[total_data["Aasta"] == valitud_aasta][[
        "Keskmine_pinnaühikuhind",
        "netosissetulek", "hoivatute_arv", "mitteaktiivsed",
        "toohive_maar", "toojoud_ja_mitteaktiivsed", "toojoud_arv",
        "toojous_osalemine", "tootuse_maar", "tootute_arv", "leibkondade_arv"
    ]].dropna()

    corr_matrix = df_korr.corr()

    fig, ax = plt.subplots(figsize=(12, 6))  # Suurem ja laiem maatriks

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="viridis",            
        center=0,
        annot_kws={"size": 8},     
        linewidths=0.5             
    )

    st.pyplot(fig)

elif valik == "Muutused ajas":
    st.header("📈 Muutused ajas")
    st.write("Vaata kinnisvara hindasid mõjutavate näitajate muutumise trende ajateljel.")
    st.markdown(f"## Muutused ajas - trendid")

    # 📈 Visualiseerime THI, brutokuupalga ja eluasemehinnaindeksi muutust ajas
    st.subheader("Bruto kuupalga, tarbijahinnaindeksi ja eluaseme hinnaindeksi muutus ajas")

    # Kui sul on netosissetulek total_data-s, siis pead selle kõigepealt arvutama kvartalite lõikes:
    # Aga kui soovid lihtsalt avg_salary kasutada vasakul teljel, siis:
    thi_reset = thi.reset_index()

    # Ajavahemiku valik
    quarters = thi_reset["quarter"].tolist()
    start_q = st.selectbox("Vali algkvartal (kahe teljega joonis)", quarters, index=0, key="start_q2")
    end_q = st.selectbox("Vali lõppkvartal (kahe teljega joonis)", quarters, index=len(quarters)-1, key="end_q2")

    # Ajavahemiku filter
    if quarters.index(start_q) < quarters.index(end_q):
        df_filtered = thi_reset[(thi_reset["quarter"] >= start_q) & (thi_reset["quarter"] <= end_q)]
    else:
        df_filtered = thi_reset.copy()

    # Loome joonise
    fig = go.Figure()

    # Vasak Y-telg: brutopalk (või netosissetulek, kui asendad)
    fig.add_trace(
        go.Scatter(
            x=df_filtered["quarter"],
            y=df_filtered["avg_salary"],
            name="Brutopalk",
            line=dict(color="blue"),
            yaxis="y1"
        )
    )

    # Parem Y-telg: THI
    fig.add_trace(
        go.Scatter(
            x=df_filtered["quarter"],
            y=df_filtered["thi"],
            name="THI",
            line=dict(color="green", dash="dot"),
            yaxis="y2"
        )
    )

    # Parem Y-telg: Eluaseme hinnaindeks
    fig.add_trace(
        go.Scatter(
            x=df_filtered["quarter"],
            y=df_filtered["housing_index"],
            name="Eluaseme hinnaindeks",
            line=dict(color="orange", dash="dot"),
            yaxis="y2"
        )
    )

    # Telgede seadistamine
    fig.update_layout(
        title="Brutopalk vs THI ja eluaseme hinnaindeks ajas",
        xaxis=dict(title="Kvartal"),
        yaxis=dict(
        title=dict(text="Brutopalk (€)", font=dict(color="blue")),
        tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title=dict(text="Indeks", font=dict(color="green")),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, max(df_filtered[["thi", "housing_index"]].max()) * 1.1]  # nt 0–350
        ),
        legend=dict(x=0.01, y=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📉 Kinnisvara hinna muutus maakonniti")

    # --- Kasutajasõbralikud nimed ---
    hoone_liik_map = {
        "korter_10_29.99": "Korter 10–29.99m²",
        "korter_30_40.99": "Korter 30–40.99m²",
        "korter_41_54.99": "Korter 41–54.99m²",
        "korter_55_69.99": "Korter 55–69.99m²",
        "korter_70_249.99": "Korter 70–249.99m²",
        "Elamu": "Elamu",
        "Muu": "Muu eluruum",
        "Suvila": "Suvila",
        "Kokku": "Kokku"
    }
    # Vastupidine sõnastik kuvamisnime -> kood
    hoone_liik_reverse_map = {v: k for k, v in hoone_liik_map.items()}

    # Loome valiku kuvamisnimede põhjal
    hoone_liigid_raw = kv_thi["Hoone_liik"].unique().tolist()
    hoone_liigid_raw.append("Kokku")
    hoone_liigid_display = [hoone_liik_map.get(x, x) for x in hoone_liigid_raw]

    # Leiame indeksi "Kokku" vaikevalikuks
    kokku_index = hoone_liigid_display.index("Kokku")

    # Kuvame kasutajale valiku
    hoonevalik_display = st.selectbox(
        "Vali hoone liik selle graafiku jaoks:",
        options=hoone_liigid_display,
        index=kokku_index
    )

    # Leiame vastava toorandmete väärtuse
    hoonevalik = hoone_liik_reverse_map[hoonevalik_display]

    # --- Filtreerime valiku alusel ---
    if hoonevalik != "Kokku":
        filtered_line = kv_thi[kv_thi["Hoone_liik"] == hoonevalik]
    else:
        filtered_line = kv_thi.copy()

    # Grupitakse kvartali ja maakonna kaupa
    grouped = (
        filtered_line.groupby(["quarter", "Maakond"], as_index=False)["Keskmine_pinnaühikuhind"]
        .mean()
        .sort_values("quarter")
    )

    # Loome joondiagrammi
    fig_line = px.line(
        grouped,
        x="quarter",
        y="Keskmine_pinnaühikuhind",
        color="Maakond",
        markers=True,
        labels={"Keskmine_pinnaühikuhind": "€/m²", "quarter": "Kvartal"},
        title=f"{hoonevalik_display} keskmise ruutmeetri hinna muutus ajas maakonniti"
    )

    fig_line.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_line, use_container_width=True)


    # Eeldus: total_data on juba laaditud ja puhastatud
    maakonnad = sorted(total_data["Maakond"].dropna().unique())
    st.subheader("📊 Tunnuste korrelatsioon ajas maakondade lõikes")

    # Tõlgendatavad nimetused
    columns_dict = {
        "Keskmine_pinnaühikuhind": "Keskmine pinnaühikuhind",
        "netosissetulek": "Netosissetulek",
        "hoivatute_arv": "Hõivatute arv",
        "mitteaktiivsed": "Mitteaktiivsete arv",
        "toohive_maar": "Tööhõive määr",
        "toojoud_ja_mitteaktiivsed": "Tööjõud + mitteaktiivsed",
        "toojoud_arv": "Tööjõu arv",
        "toojous_osalemine": "Tööjõus osalemise määr",
        "tootuse_maar": "Töötuse määr",
        "tootute_arv": "Töötute arv",
        "leibkondade_arv": "Leibkondade arv",
        "avg_salary": "Keskmine palk",
        "housing_index": "Eluaseme hinnaindeks"
    }

    columns_keys = list(columns_dict.keys())
    columns_labels = list(columns_dict.values())

    maakond_valik = st.selectbox("Vali maakond", ["Kõik maakonnad"] + list(maakonnad))

    col1_label = st.selectbox("Vali esimene tunnus", columns_labels)
    col2_label = st.selectbox("Vali teine tunnus", [l for l in columns_labels if l != col1_label])

    # Teisendame inimloetavad nimetused tagasi võtmeteks
    col1 = [k for k, v in columns_dict.items() if v == col1_label][0]
    col2 = [k for k, v in columns_dict.items() if v == col2_label][0]

    fig, ax = plt.subplots(figsize=(10, 5))

    if maakond_valik == "Kõik maakonnad":
        vigased_kokku = 0
        for maakond in maakonnad:
            maakond_data = total_data[total_data["Maakond"] == maakond]
            yearly = []
            vigased = 0

            for aasta in sorted(maakond_data["Aasta"].unique()):
                aasta_df = maakond_data[maakond_data["Aasta"] == aasta][[col1, col2]].dropna()
                if len(aasta_df) >= 2:
                    corr = aasta_df[col1].corr(aasta_df[col2])
                    if pd.notnull(corr):
                        yearly.append((aasta, corr))
                    else:
                        vigased += 1
                else:
                    vigased += 1

            if yearly:
                df = pd.DataFrame(yearly, columns=["Aasta", "Korrelatsioon"])
                ax.plot(df["Aasta"], df["Korrelatsioon"], label=maakond)

            vigased_kokku += vigased

        ax.set_title(f"{col1_label} ja {col2_label} korrelatsioon aastati – kõik maakonnad")
        ax.set_ylabel("Pearsoni korrelatsioon")
        ax.set_xlabel("Aasta")
        ax.grid(True)
        ax.axhline(0, color='gray', linestyle='--')
        ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        st.pyplot(fig)

        if vigased_kokku > 0:
            st.info(f"⚠️ Mõned aastad jäeti välja ({vigased_kokku} juhtumit), kuna korrelatsiooni ei saanud usaldusväärselt arvutada.")

        st.markdown(f"""
        Graafikul on näha, kuidas kahe tunnuse seos muutus ajas **igas maakonnas eraldi**.
        Kui joon kõigub palju, võib seos olla ajas ebastabiilne – kui ta püsib stabiilselt kõrge, siis tegemist on tugeva ja püsiva seosega.
        """)

    else:
        maakond_data = total_data[total_data["Maakond"] == maakond_valik]
        yearly_corr = []
        vigased = 0

        for aasta in sorted(maakond_data["Aasta"].unique()):
            aasta_df = maakond_data[maakond_data["Aasta"] == aasta][[col1, col2]].dropna()
            if len(aasta_df) >= 2:
                corr = aasta_df[col1].corr(aasta_df[col2])
                if pd.notnull(corr):
                    yearly_corr.append({"Aasta": aasta, "Korrelatsioon": corr})
                else:
                    vigased += 1
            else:
                vigased += 1

        corr_df = pd.DataFrame(yearly_corr)
        if not corr_df.empty:
            ax.plot(corr_df["Aasta"], corr_df["Korrelatsioon"], marker='o', color='mediumblue')
            ax.set_title(f"{col1_label} ja {col2_label} korrelatsioon maakonnas: {maakond_valik}")
            ax.set_ylabel("Pearsoni korrelatsioon")
            ax.set_xlabel("Aasta")
            ax.grid(True)
            ax.axhline(0, color='gray', linestyle='--')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            st.pyplot(fig)

            if vigased > 0:
                st.info(f"⚠️ {vigased} aastat jäeti välja, kuna korrelatsiooni ei saanud usaldusväärselt arvutada.")

            st.markdown(f"""
            Aastate lõikes on näha, **kui tugevalt seonduvad tunnused _{col1_label}_ ja _{col2_label}_ maakonnas _{maakond_valik}_**.
            """)
        else:
            st.warning("Selle maakonna ja tunnuste kombinatsiooni kohta ei leitud piisavalt andmeid korrelatsiooni arvutamiseks.")

    
elif valik == "Prognoosid":
    st.header("🔮 Prognoosid")
    st.write("Kinnisvara hinda mõjutavate näitajate prognoosid järgnevateks aastateks.")
    st.markdown("## Prognoosid tulevikuks")
    st.subheader("Makromajandusnäitajate prognoos tulevikuks")

    # Nähtavad nimed ja nende vastavus andmetulbanimedele
    naidiku_valikud = {
        "Tarbijahinnaindeks": "thi",
        "Keskmine brutokuupalk": "avg_salary",
        "Eluaseme hinnaindeks": "housing_index"
    }
    
    # === VALIK: Mida prognoosime ===
    valitud_nimi = st.selectbox("Vali ennustatav näitaja:", list(naidiku_valikud.keys()))
    target = naidiku_valikud[valitud_nimi]

    # --- Andmete ettevalmistus ---
    df = thi.copy()  # eelnevalt töödeldud DataFrame, mis sisaldab 'quarter' ja kõiki näitajaid
    df = df.reset_index()
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df[['quarter', target]]
    df.columns = ['ds', 'y']

    # --- PROPHET ---
    model_prophet = Prophet()
    model_prophet.fit(df)

    future = model_prophet.make_future_dataframe(periods=4, freq='Q')
    forecast_prophet = model_prophet.predict(future)

    # --- LINEAARNE REGRESSIOON ---
    df['ordinal'] = df['ds'].map(pd.Timestamp.toordinal)
    X = df['ordinal'].values.reshape(-1, 1)
    y = df['y'].values

    lin_model = LinearRegression()
    lin_model.fit(X, y)

    future_dates = pd.date_range(start=df['ds'].max() + pd.offsets.QuarterEnd(), periods=4, freq='Q')
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

    y_pred = lin_model.predict(future_ordinals)

    # --- Usalduspiirid ---
    y_train_pred = lin_model.predict(X)
    mse = mean_squared_error(y, y_train_pred)
    se = np.sqrt(mse)

    t_val = stats.t.ppf(0.975, df=len(X) - 2)
    margin = t_val * se

    y_pred_upper = y_pred + margin
    y_pred_lower = y_pred - margin

    # === GRAAFIKUD KÕRVUTI ===
    col1, col2 = st.columns(2)

    # Prophet plot
    with col1:
        st.markdown(f"#### Prophet prognoos: {valitud_nimi}")
        fig1 = model_prophet.plot(forecast_prophet)
        fig1.set_size_inches(5, 4)
        st.pyplot(fig1)

    # Linear regression plot
    with col2:
        st.markdown(f"#### Lineaarne regressioon: {valitud_nimi}")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(df['ds'], df['y'], label='Ajaloolised andmed')
        ax2.plot(future_dates, y_pred, label='Prognoos', color='green')
        ax2.fill_between(future_dates, y_pred_lower, y_pred_upper, color='green', alpha=0.2, label='95% usalduspiir')
        ax2.set_title(f'{valitud_nimi} lineaarse regressiooni prognoos')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    # Võiks tulla ka tööjõuturu näitajate prognoos

    st.subheader("Tööjõuturu näitajate prognoos tulevikuks")

    if "total_data" not in globals():
        st.error("Andmestik 'total_data' pole defineeritud. Palun lae andmed sisse.")
        st.stop()

    # Sõnastik: kuva -> sisemine väärtus
    naidikud_dict = {
        "Netosissetulek": "netosissetulek",
        "Hõivatute arv": "hoivatute_arv",
        "Mitteaktiivsete arv": "mitteaktiivsed",
        "Tööhõive määr": "toohive_maar",
        "Tööjõud ja mitteaktiivsed": "toojoud_ja_mitteaktiivsed",
        "Tööjõu arv": "toojoud_arv",
        "Tööjõus osalemise määr": "toojous_osalemine",
        "Töötuse määr": "tootuse_maar",
        "Töötute arv": "tootute_arv",
        "Leibkondade arv": "leibkondade_arv"
    }

    # Valik näitaja kuvamiseks
    kuvatav_valik = st.selectbox("Vali näitaja, mille kohta prognoosida", list(naidikud_dict.keys()))
    valitud_naidik = naidikud_dict[kuvatav_valik]

    # Maakonna valik
    maakonnad_valikud = ["Kõik maakonnad"] + sorted(total_data["Maakond"].unique())
    valitud_maakond = st.selectbox("Vali maakond", maakonnad_valikud, key="maakond_valik")

    # Filter andmed
    if valitud_maakond == "Kõik maakonnad":
        df = total_data[["Aasta", valitud_naidik]].dropna()
        df = df.groupby("Aasta")[valitud_naidik].mean().reset_index()
    else:
        df = total_data[(total_data["Maakond"] == valitud_maakond)][["Aasta", valitud_naidik]].dropna()
        df = df.groupby("Aasta")[valitud_naidik].mean().reset_index()

    if df.empty:
        st.warning("Valitud kombinatsioonil puuduvad andmed.")
        st.stop()

    # Prognoos: mitu aastat edasi
    prognoos_aastaid = 5
    aastad_tulevikus = np.arange(df["Aasta"].max() + 1, df["Aasta"].max() + 1 + prognoos_aastaid)

    # --- Prognoos 1: Lineaarregressioon
    X = df["Aasta"].values.reshape(-1, 1)
    y = df[valitud_naidik].values
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    future_X = aastad_tulevikus.reshape(-1, 1)
    lr_pred = lr_model.predict(future_X)

    # Lihtne usaldusintervall
    residuals = y - lr_model.predict(X)
    std_err = np.std(residuals)
    lr_conf_int = 1.96 * std_err

    # --- Prognoos 2: SARIMA mudel
    sarima_model = SARIMAX(df[valitud_naidik], order=(1,1,1), seasonal_order=(0,0,0,0))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.get_forecast(steps=prognoos_aastaid)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    # --- Y-telg skaleering
    y_all = np.concatenate([df[valitud_naidik].values, lr_pred, sarima_pred])
    y_min = y_all.min() * 0.95
    y_max = y_all.max() * 1.05

    # Kuvame graafikud
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lineaarregressiooni prognoos")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["Aasta"], df[valitud_naidik], label="Ajalooline", marker='o')
        ax1.plot(aastad_tulevikus, lr_pred, label="Prognoos", marker='x')
        ax1.fill_between(
            aastad_tulevikus.flatten(),
            lr_pred - lr_conf_int,
            lr_pred + lr_conf_int,
            color='orange', alpha=0.3, label="±95% usaldus"
        )
        ax1.set_title(f"{kuvatav_valik} – Lineaarregressioon")
        ax1.set_xlabel("Aasta")
        ax1.set_ylabel(kuvatav_valik)
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.subheader("SARIMA prognoos")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Aasta"], df[valitud_naidik], label="Ajalooline", marker='o')
        ax2.plot(aastad_tulevikus, sarima_pred, label="Prognoos", marker='x')
        ax2.fill_between(
            aastad_tulevikus,
            sarima_ci.iloc[:, 0],
            sarima_ci.iloc[:, 1],
            color='lightblue', alpha=0.4, label="95% usaldus"
        )
        ax2.set_title(f"{kuvatav_valik} – SARIMA")
        ax2.set_xlabel("Aasta")
        ax2.set_ylabel(kuvatav_valik)
        ax2.set_ylim(y_min, y_max)
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    # --- Kasutajasõbralik nimede kaardistus ---
    #  Kaardistused
    hoone_liik_map = {
        "korter_10_29.99": "Korter 10–29.99m²",
        "korter_30_40.99": "Korter 30–40.99m²",
        "korter_41_54.99": "Korter 41–54.99m²",
        "korter_55_69.99": "Korter 55–69.99m²",
        "korter_70_249.99": "Korter 70–249.99m²",
        "Elamu": "Elamu",
        "Muu": "Muu eluruum",
        "Suvila": "Suvila"
    }

    # Vastupidine kaardistus
    hoone_liik_reverse_map = {v: k for k, v in hoone_liik_map.items()}
    hoone_liik_reverse_map["Kokku"] = "Kokku"

    st.subheader("Kinnisvara ruutmeetrihinna prognoos 2025 – Eesti maakondades")

    # --- Hoone_liik valik ---
    hoone_liigid_raw = total_data['Hoone_liik'].unique().tolist()
    hoone_liigid_raw.append("Kokku")  # Lisa "Kokku" valikuna

    # Kuvamisnimekirja loomine
    hoone_liigid_display = [hoone_liik_map.get(x, x) if x != "Kokku" else "Kokku" for x in hoone_liigid_raw]

    # Leia "Kokku" indeks kuvamisnimekirjas
    kokku_index = hoone_liigid_display.index("Kokku")

    # Kasutaja valik
    valitud_hoone_display = st.selectbox("Vali hooneliik", hoone_liigid_display, index=kokku_index)
    valitud_hoone = hoone_liik_reverse_map[valitud_hoone_display]


    # --- Andmete ettevalmistus ---
    features = [
        'toohive_maar', 'tootuse_maar', 'leibkondade_arv',
        'avg_salary', 'housing_index', 'THI'
    ]
    target = 'Keskmine_pinnaühikuhind'

    df_model = total_data.dropna(subset=features + [target, 'Hoone_liik', 'Maakond', 'Aasta']).copy()

    latest_year = df_model['Aasta'].max()
    future_year = 2025

    if valitud_hoone != "Kokku":
        df_model_filtered = df_model[df_model['Hoone_liik'] == valitud_hoone]
    
        future_data = df_model_filtered[df_model_filtered['Aasta'] == latest_year].copy()
        future_data['Aasta'] = future_year

        le_hoone = LabelEncoder()
        le_maakond = LabelEncoder()
        df_model_filtered['Hoone_liik_enc'] = le_hoone.fit_transform(df_model_filtered['Hoone_liik'])
        df_model_filtered['Maakond_enc'] = le_maakond.fit_transform(df_model_filtered['Maakond'])
        future_data['Hoone_liik_enc'] = le_hoone.transform(future_data['Hoone_liik'])
        future_data['Maakond_enc'] = le_maakond.transform(future_data['Maakond'])

        features_enc = features + ['Hoone_liik_enc', 'Maakond_enc', 'Aasta']
        X_train = df_model_filtered[features_enc]
        y_train = df_model_filtered[target]
        X_future = future_data[features_enc]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_future_scaled = scaler.transform(X_future)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        future_data['Predicted_Keskmine_pinnaühikuhind'] = model.predict(X_future_scaled)

        agg_cols = ['Maakond', 'Hoone_liik']
        hist_means = df_model_filtered[df_model_filtered['Aasta'] == latest_year].groupby(agg_cols)[target].mean().reset_index()
        pred_means = future_data.groupby(agg_cols)['Predicted_Keskmine_pinnaühikuhind'].mean().reset_index()

        df_compare = hist_means.merge(pred_means, on=agg_cols, how='inner')
        df_compare['Hinna_muutus'] = df_compare['Predicted_Keskmine_pinnaühikuhind'] - df_compare[target]
        df_compare['Protsent_muutus'] = 100 * df_compare['Hinna_muutus'] / df_compare[target]

    else:
        hist_latest = df_model[df_model['Aasta'] == latest_year].copy()
        le_hoone = LabelEncoder()
        le_maakond = LabelEncoder()
        df_model['Hoone_liik_enc'] = le_hoone.fit_transform(df_model['Hoone_liik'])
        df_model['Maakond_enc'] = le_maakond.fit_transform(df_model['Maakond'])
    
        future_data = hist_latest.copy()
        future_data['Aasta'] = future_year
        future_data['Hoone_liik_enc'] = le_hoone.transform(future_data['Hoone_liik'])
        future_data['Maakond_enc'] = le_maakond.transform(future_data['Maakond'])

        features_enc = features + ['Hoone_liik_enc', 'Maakond_enc', 'Aasta']
        X_train = df_model[features_enc]
        y_train = df_model[target]
        X_future = future_data[features_enc]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_future_scaled = scaler.transform(X_future)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        future_data['Predicted_Keskmine_pinnaühikuhind'] = model.predict(X_future_scaled)

        hist_means = hist_latest.groupby('Maakond')[target].mean().reset_index()
        pred_means = future_data.groupby('Maakond')['Predicted_Keskmine_pinnaühikuhind'].mean().reset_index()

        df_compare = hist_means.merge(pred_means, on='Maakond', how='inner')
        df_compare['Hinna_muutus'] = df_compare['Predicted_Keskmine_pinnaühikuhind'] - df_compare[target]
        df_compare['Protsent_muutus'] = 100 * df_compare['Hinna_muutus'] / df_compare[target]
        df_compare['Hoone_liik'] = 'Kokku'

    maakonnad = geo_df['MNIMI'].unique()
    if valitud_hoone == "Kokku":
        hoone_liigid_unik = ['Kokku']
    else:
        hoone_liigid_unik = [valitud_hoone]

    full_combinations = pd.DataFrame(product(maakonnad, hoone_liigid_unik), columns=['Maakond', 'Hoone_liik'])
    df_full = full_combinations.merge(df_compare, on=['Maakond', 'Hoone_liik'], how='left')

    mean_pct_change = df_compare['Protsent_muutus'].mean()
    mean_hist_price = df_compare[target].mean()

    df_full['Protsent_muutus'] = df_full['Protsent_muutus'].fillna(mean_pct_change)
    df_full[target] = df_full[target].fillna(mean_hist_price)
    df_full['Hinna_muutus'] = df_full[target] * df_full['Protsent_muutus'] / 100
    df_full['Predicted_Keskmine_pinnaühikuhind'] = df_full[target] + df_full['Hinna_muutus']

    df_plot = df_full[df_full['Hoone_liik'] == valitud_hoone]
    gdf_plot = geo_df.merge(df_plot, left_on='MNIMI', right_on='Maakond', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    gdf_plot.plot(column='Protsent_muutus', cmap='RdYlGn', legend=True, ax=ax,
                legend_kwds={'label': "Ruutmeetri hinna muutus (%) aastani 2025",
                            'orientation': "vertical"},
                missing_kwds={"color": "lightgrey", "label": "Andmed puuduvad"})

    ax.set_title(f'Hinnamuutuse prognoos maakonniti hooneliigi "{valitud_hoone_display}" kohta aastaks {future_year}')
    ax.axis('off')
    st.pyplot(fig)

