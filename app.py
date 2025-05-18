import streamlit as st
import pandas as pd

st.title("Kinnisvaraturu ülevaade")

# Näiteks: andmete laadimine
df_total = pd.read_csv("andmed/df_total.csv")

st.write(df_total.head())
