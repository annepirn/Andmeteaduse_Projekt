import streamlit as st
import pandas as pd
import os
import plotly.express as px
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np

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
    st.subheader("Tarbijahinnaindeks (THI)")
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
st.markdown(f"### Ülevaade: Mis on praegune seis tõõjõu turul? ({valitud_aasta})")
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

# 📈 Visualiseerime THI, brutokuupalga ja eluasemehinnaindeksi muutust ajas
import plotly.graph_objects as go

st.subheader("Netosissetulek võrreldes THI ja eluaseme hinnaindeksiga")

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
if hoonevalik_line != "Kokku":
    filtered_line = kv_thi[kv_thi["Hoone_liik"] == hoonevalik_line]
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
    title=f"{hoonevalik_line.replace('_', ' ').capitalize()} keskmise ruutmeetri hinna muutus ajas maakonniti"
)

fig_line.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig_line, use_container_width=True)