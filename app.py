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
selected_q = st.selectbox("Vali kvartal", available_quarters, index=len(available_quarters)-1)
hoone_liigid = sorted(kv_thi["Hoone_liik"].dropna().unique())
valikuga = ["Kokku"] + hoone_liigid
hoonevalik = st.selectbox("Vali hoone liik", valikuga)

# 📊 Funktsioon muutuste arvutamiseks
def calc_percent(current, previous):
    if previous == 0 or pd.isna(previous):
        return None
    return round(((current - previous) / previous) * 100, 2)

def calculate_changes(df, selected_q):
    # kvartali komponendid
    prev_q = df.index[df.index.get_loc(selected_q) - 1] if df.index.get_loc(selected_q) > 0 else None
    # Extract year and quarter number
    selected_year = int(selected_q[:4])
    selected_quarter = int(selected_q[-1])

    if selected_quarter == 1:
        # Kui Q1, siis aasta alguseks loeme eelmise aasta Q4
        year_start = f"{selected_year - 1}Q4"
    else:
        # Kui Q2, Q3, Q4 → aasta algus = sama aasta Q1
        year_start = f"{selected_year}Q1"

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

filtered = kv_thi[kv_thi["quarter"] == selected_q]
if hoonevalik != "Kokku":
    filtered = filtered[filtered["Hoone_liik"] == hoonevalik]
    df_map = filtered.groupby("Maakond", as_index=False)["Keskmine_pinnaühikuhind"].mean()
else:
    df_map = filtered.groupby("Maakond", as_index=False)["Keskmine_pinnaühikuhind"].mean()

# Ühenda ruumiandmetega
merged = geo_df.merge(df_map, how="left", left_on="Maakond", right_on="Maakond")


# --- DASHBOARD ---
st.title("📊 Kinnisvara dashboard – Ülevaade")
st.markdown(f"### Ülevaade: Mis on praegune seis makromajandusnäitajates? **{selected_q}**")

# 3 veergu: THI, brutopalk, hinnaindeks
col1, col2, col3 = st.columns(3)

# THI
with col1:
    st.subheader("Tarbijahinna-indeks (THI)")
    st.markdown(f"### **{changes['thi']['current']}**")
    st.markdown("Muutus eelmise kvartaliga")
    st.markdown(format_change(changes['thi']['prev_q']))
    st.markdown("Muutus aasta algusest")
    st.markdown(format_change(changes['thi']['year_start']))
    st.markdown("Muutus aasta tagusega")
    st.markdown(format_change(changes['thi']['prev_year']))

# Brutopalk
with col2:
    st.subheader("Keskmine brutopalk")
    st.markdown(f"### **{changes['avg_salary']['current']}**")
    st.markdown("Muutus eelmise kvartaliga")
    st.markdown(format_change(changes['avg_salary']['prev_q']))
    st.markdown("Muutus aasta algusest")
    st.markdown(format_change(changes['avg_salary']['year_start']))
    st.markdown("Muutus aasta tagusega")
    st.markdown(format_change(changes['avg_salary']['prev_year']))

# Eluaseme hinnaindeks
with col3:
    st.subheader("Eluaseme hinnaindeks")
    st.markdown(f"### **{changes['housing_index']['current']}**")
    st.markdown("Muutus eelmise kvartaliga")
    st.markdown(format_change(changes['housing_index']['prev_q']))
    st.markdown("Muutus aasta algusest")
    st.markdown(format_change(changes['housing_index']['year_start']))
    st.markdown("Muutus aasta tagusega")
    st.markdown(format_change(changes['housing_index']['prev_year']))

hoone_nimi = hoonevalik.replace("_", " ").capitalize()
pealkiri_kv = f"{hoone_nimi} keskmine ruutmeetri hind maakonniti, {selected_q}"

# Joonista kaart
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
    title_x=0.3,  # Keskenda pealkiri
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
valitud_naidik = st.selectbox(
    "Vali näitaja kaardile kuvamiseks",
    [
        "netosissetulek", "hoivatute_arv", "mitteaktiivsed",
        "toohive_maar", "toojoud_ja_mitteaktiivsed", "toojoud_arv",
        "toojous_osalemine", "tootuse_maar", "tootute_arv", "leibkondade_arv"
    ]
)

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
st.markdown(f"### Ülevaade: Mis on praegune seis tööjõu turul? ({valitud_aasta})")
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
# Uus hoone liigi valik ainult selle graafiku jaoks
hoonevalik_line = st.selectbox(
    "Vali hoone liik selle graafiku jaoks:",
    options=kv_thi["Hoone_liik"].unique().tolist() + ["Kokku"],
    index=0
)

# Filter hoone liigi alusel
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

# Loome joondiagrammi, kus iga maakond on eraldi joon
fig_line = px.line(
    grouped,
    x="quarter",
    y="Keskmine_pinnaühikuhind",
    color="Maakond",
    markers=True,
    labels={"Keskmine_pinnaühikuhind": "€/m²", "quarter": "Kvartal"},
    title=f"{hoonevalik.replace('_', ' ').capitalize()} keskmise ruutmeetri hinna muutus ajas maakonniti"
)

fig_line.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig_line, use_container_width=True)

st.subheader("📈 Tööjõuturu näitaja trendid maakonniti")

# Näidiku valik ainult trendigraafikule
trend_naidikud = [
    "netosissetulek", "hoivatute_arv", "mitteaktiivsed",
    "toohive_maar", "toojoud_ja_mitteaktiivsed", "toojoud_arv",
    "toojous_osalemine", "tootuse_maar", "tootute_arv", "leibkondade_arv"
]

valitud_trend_naidik = st.selectbox(
    "Vali trendigraafiku näitaja",
    trend_naidikud,
    index=trend_naidikud.index(valitud_naidik) if valitud_naidik in trend_naidikud else 0
)

# Maakondade valik koos "Kõik maakonnad" valikuga
maakonnad = sorted(total_data["Maakond"].unique())
maakonnad_valikud = ["Kõik maakonnad"] + maakonnad

valitud_maakond = st.selectbox(
    "Vali maakond trendigraafikule",
    maakonnad_valikud,
    index=0
)

if valitud_maakond == "Kõik maakonnad":
    trend_df = total_data[
        (total_data[valitud_trend_naidik].notna())
    ][["Aasta", "Maakond", valitud_trend_naidik]].copy()
else:
    trend_df = total_data[
        (total_data["Maakond"] == valitud_maakond) &
        (total_data[valitud_trend_naidik].notna())
    ][["Aasta", "Maakond", valitud_trend_naidik]].copy()

trend_grouped = (
    trend_df.groupby(["Aasta", "Maakond"], as_index=False)[valitud_trend_naidik]
    .mean()
    .sort_values("Aasta")
)

# Näitaja nimi pealkirja jaoks ilusamaks
naidiku_nimi = valitud_trend_naidik.replace("_", " ").capitalize()

fig_trend = px.line(
    trend_grouped,
    x="Aasta",
    y=valitud_trend_naidik,
    color="Maakond",
    markers=True,
    labels={valitud_trend_naidik: naidiku_nimi, "Aasta": "Aasta"},
    title=f"{naidiku_nimi} trendid {valitud_maakond.lower()}"
)
fig_trend.update_layout(xaxis=dict(dtick=1))

st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("Tunnuste korrelatsioon ajas maakondade lõikes")

columns = [
    "Keskmine_pinnaühikuhind", "netosissetulek", "hoivatute_arv", "mitteaktiivsed",
    "toohive_maar", "toojoud_ja_mitteaktiivsed", "toojoud_arv",
    "toojous_osalemine", "tootuse_maar", "tootute_arv",
    "leibkondade_arv", "avg_salary", "housing_index"
]

maakonnad = sorted(total_data["Maakond"].dropna().unique())
maakond_valik = st.selectbox("Vali maakond", ["Kõik maakonnad"] + list(maakonnad))
col1 = st.selectbox("Vali esimene tunnus", columns)
col2 = st.selectbox("Vali teine tunnus", [col for col in columns if col != col1])

# Graafiku joonistamine
fig, ax = plt.subplots(figsize=(10, 5))

if maakond_valik == "Kõik maakonnad":
    grouped = total_data.groupby(["Maakond", "Aasta"])
    korrelatsioonid = []
    vigased_kokku = 0

    for maakond in maakonnad:
        maakond_data = total_data[total_data["Maakond"] == maakond]
        yearly = []
        vigased = 0

        for aasta in sorted(maakond_data["Aasta"].unique()):
            aasta_df = maakond_data[maakond_data["Aasta"] == aasta][[col1, col2]].dropna()
            if len(aasta_df) >= 2:
                corr = aasta_df[col1].corr(aasta_df[col2])
                if pd.notnull(corr) and -1 <= corr <= 1:
                    yearly.append((aasta, corr))
                else:
                    vigased += 1
            else:
                vigased += 1

        if yearly:
            df = pd.DataFrame(yearly, columns=["Aasta", "Korrelatsioon"])
            ax.plot(df["Aasta"], df["Korrelatsioon"], label=maakond)

        vigased_kokku += vigased

    ax.set_title(f"{col1} ja {col2} korrelatsioon aastati – kõik maakonnad")
    ax.set_ylabel("Pearsoni korrelatsioon")
    ax.set_xlabel("Aasta")
    ax.grid(True)
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    st.pyplot(fig)

    if vigased_kokku > 0:
        st.info(f"⚠️ Mõned aastad jäeti välja ({vigased_kokku} juhtumit), kuna korrelatsiooni ei saanud usaldusväärselt arvutada.")

    st.markdown("""
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
            if pd.notnull(corr) and -1 <= corr <= 1:
                yearly_corr.append({"Aasta": aasta, "Korrelatsioon": corr})
            else:
                vigased += 1
        else:
            vigased += 1

    corr_df = pd.DataFrame(yearly_corr)
    if not corr_df.empty:
        ax.plot(corr_df["Aasta"], corr_df["Korrelatsioon"], marker='o', color='mediumblue')
        ax.set_title(f"{col1} ja {col2} korrelatsioon maakonnas: {maakond_valik}")
        ax.set_ylabel("Pearsoni korrelatsioon")
        ax.set_xlabel("Aasta")
        ax.grid(True)
        ax.axhline(0, color='gray', linestyle='--')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        st.pyplot(fig)

        if vigased > 0:
            st.info(f"⚠️ {vigased} aastat jäeti välja, kuna korrelatsiooni ei saanud usaldusväärselt arvutada.")

        st.markdown(f"""
        Aastate lõikes on näha, **kui tugevalt seonduvad tunnused {col1} ja {col2} maakonnas {maakond_valik}**.
        """)
    else:
        st.warning("Selle maakonna ja tunnuste kombinatsiooni kohta ei leitud piisavalt andmeid korrelatsiooni arvutamiseks.")


# === VALIK: Mida prognoosime ===
target = st.selectbox("Vali ennustatav näitaja:", ['thi', 'avg_salary', 'housing_index'])

# Andmete ettevalmistus
df = thi.copy()  # sinu eelnevalt töödeldud data
df = df.reset_index()
df['quarter'] = pd.to_datetime(df['quarter'])  # veendu, et kuupäev
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

# --- Usalduspiirid (lihtsustatud lineaarse mudeli puhul) ---
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

y_train_pred = lin_model.predict(X)
mse = mean_squared_error(y, y_train_pred)
se = np.sqrt(mse)

t_val = stats.t.ppf(0.975, df=len(X) - 2)  # 95% usaldus
margin = t_val * se

y_pred_upper = y_pred + margin
y_pred_lower = y_pred - margin

# === GRAAFIKUD KÕRVUTI ===
col1, col2 = st.columns(2)

# Prophet plot
with col1:
    st.markdown("#### Prophet prognoos")
    fig1 = model_prophet.plot(forecast_prophet)
    fig1.set_size_inches(8, 4)  # fikseeri suurus (laius x kõrgus tollides)
    st.pyplot(fig1)

# Linear regression plot
with col2:
    st.markdown("#### Lineaarne regressioon")
    fig2, ax2 = plt.subplots(figsize=(6, 4))  # sama suurus
    ax2.plot(df['ds'], df['y'], label='Ajaloolised andmed')
    ax2.plot(future_dates, y_pred, label='Prognoos', color='green')
    ax2.fill_between(future_dates, y_pred_lower, y_pred_upper, color='green', alpha=0.2, label='95% usalduspiir')
    ax2.set_title('Lineaarse regressiooni prognoos')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# Võiks tulla ka tööjõuturu näitajate prognoos


st.title("Kinnisvara ruutmeetrihinna prognoos 2025 – Eesti maakondades")

# --- Hoone_liik valik ---
hoone_liigid = total_data['Hoone_liik'].unique().tolist()
hoone_liigid.append("Kokku")  # lisame "Kokku" valiku

valitud_hoone = st.selectbox("Vali hooneliik", hoone_liigid)

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
    # Kui valiti konkreetne hooneliik, filtreeri tavaliselt
    df_model_filtered = df_model[df_model['Hoone_liik'] == valitud_hoone]
    
    future_data = df_model_filtered[df_model_filtered['Aasta'] == latest_year].copy()
    future_data['Aasta'] = future_year

    # Kategooriate kodeerimine
    le_hoone = LabelEncoder()
    le_maakond = LabelEncoder()
    df_model_filtered['Hoone_liik_enc'] = le_hoone.fit_transform(df_model_filtered['Hoone_liik'])
    df_model_filtered['Maakond_enc'] = le_maakond.fit_transform(df_model_filtered['Maakond'])
    future_data['Hoone_liik_enc'] = le_hoone.transform(future_data['Hoone_liik'])
    future_data['Maakond_enc'] = le_maakond.transform(future_data['Maakond'])

    # Mudel
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

    # Hinnamuutuste arvutus
    agg_cols = ['Maakond', 'Hoone_liik']
    hist_means = df_model_filtered[df_model_filtered['Aasta'] == latest_year].groupby(agg_cols)[target].mean().reset_index()
    pred_means = future_data.groupby(agg_cols)['Predicted_Keskmine_pinnaühikuhind'].mean().reset_index()

    df_compare = hist_means.merge(pred_means, on=agg_cols, how='inner')
    df_compare['Hinna_muutus'] = df_compare['Predicted_Keskmine_pinnaühikuhind'] - df_compare[target]
    df_compare['Protsent_muutus'] = 100 * df_compare['Hinna_muutus'] / df_compare[target]

else:
    # Kui valiti "Kokku", siis arvutame kaalutud keskmise maakonna kaupa

    # Kõik andmed viimase aasta kohta
    hist_latest = df_model[df_model['Aasta'] == latest_year].copy()
    # Prognoosi tegemiseks tuleb kasutada kõiki hooneliike korraga

    # Kategooriate kodeerimine kõigi hooneliikide ja maakondade jaoks
    le_hoone = LabelEncoder()
    le_maakond = LabelEncoder()
    df_model['Hoone_liik_enc'] = le_hoone.fit_transform(df_model['Hoone_liik'])
    df_model['Maakond_enc'] = le_maakond.fit_transform(df_model['Maakond'])
    
    future_data = hist_latest.copy()
    future_data['Aasta'] = future_year
    future_data['Hoone_liik_enc'] = le_hoone.transform(future_data['Hoone_liik'])
    future_data['Maakond_enc'] = le_maakond.transform(future_data['Maakond'])

    # Mudel treenimine kõigi hooneliikide ja maakondadega koos
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

    # Arvutame kaalutud keskmise maakonna tasandil

    # kaalud on ajalooline keskmine pindala või hulk - siin lihtsuse mõttes kasutame lihtsalt ridade arvu (või võid lisada tegeliku kaalu)
    # Kuna sul on ainult ruutmeetri hind, kasutame lihtsalt keskmist (kaaludega saab ka lisada)

    # Ajaloolised keskmised
    hist_means = hist_latest.groupby('Maakond')[target].mean().reset_index()
    # Prognoos
    pred_means = future_data.groupby('Maakond')['Predicted_Keskmine_pinnaühikuhind'].mean().reset_index()

    df_compare = hist_means.merge(pred_means, on='Maakond', how='inner')

    df_compare['Hinna_muutus'] = df_compare['Predicted_Keskmine_pinnaühikuhind'] - df_compare[target]
    df_compare['Protsent_muutus'] = 100 * df_compare['Hinna_muutus'] / df_compare[target]

    df_compare['Hoone_liik'] = 'Kokku'  # Lisa veerg "Kokku"

# --- Kõik maakonnad + täitmine ---
maakonnad = geo_df['MNIMI'].unique()
if valitud_hoone == "Kokku":
    hoone_liigid_unik = ['Kokku']
else:
    hoone_liigid_unik = [valitud_hoone]

full_combinations = pd.DataFrame(product(maakonnad, hoone_liigid_unik), columns=['Maakond', 'Hoone_liik'])
df_full = full_combinations.merge(df_compare, on=['Maakond', 'Hoone_liik'], how='left')

# Täida puuduvaid väärtusi Eesti keskmisega
mean_pct_change = df_compare['Protsent_muutus'].mean()
mean_hist_price = df_compare[target].mean()

df_full['Protsent_muutus'] = df_full['Protsent_muutus'].fillna(mean_pct_change)
df_full[target] = df_full[target].fillna(mean_hist_price)
df_full['Hinna_muutus'] = df_full[target] * df_full['Protsent_muutus'] / 100
df_full['Predicted_Keskmine_pinnaühikuhind'] = df_full[target] + df_full['Hinna_muutus']

# --- Filtreerimine kaardiks ---
df_plot = df_full[df_full['Hoone_liik'] == valitud_hoone]
gdf_plot = geo_df.merge(df_plot, left_on='MNIMI', right_on='Maakond', how='left')

# --- Kaardi joonistamine ---
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
gdf_plot.plot(column='Protsent_muutus', cmap='RdYlGn', legend=True, ax=ax,
              legend_kwds={'label': "Ruutmeetri hinna muutus (%) aastani 2025",
                           'orientation': "vertical"},
              missing_kwds={"color": "lightgrey", "label": "Andmed puuduvad"})

ax.set_title(f'Hinnamuutuse prognoos maakonniti hooneliigi "{valitud_hoone}" kohta aastaks {future_year}')
ax.axis('off')
st.pyplot(fig)


