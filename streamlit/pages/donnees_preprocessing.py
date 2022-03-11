import streamlit as st
import json
import pandas as pd
from PIL import Image

def app():

    st.header("Bookmakerspy - Données & pre-processing")
    st.subheader("Introduction")

    st.markdown("* Contexte")
    st.write("Football - English Premier League 2014/15 - 2017/18")
    st.write("Paris sportifs: mise en place d'une startégie pour tenter de battre les bookmakers")

    st.markdown("* Origine des données")
    st.write("Kaggle: données concernant les matchs et les joueurs")
    st.write("Datahub: données concernant les cotes")

    st.subheader("Collecte des données")


    st.markdown("* Données équipes et joueurs")

    st.caption("Données JSON")
    f = json.load(open('./data/season_stats_min.json'))
    st.json(f['1190174'])

    st.caption("Dataframe équipes")
    
    df_home = pd.read_csv('./data/df_home.csv')
    st.dataframe(df_home)

    st.caption("Dataframe joueurs")
    df_position_home_rating = pd.read_csv('./data/df_position_home_rating.csv')
    st.dataframe(df_position_home_rating)

    st.subheader("Pre-processing")
    
    st.caption("Jointures")
    img_jointures = Image.open('./data/img/columns_merge.png')
    st.image(img_jointures)

    st.caption("Traitement des valeurs manquantes")
    st.write("Données de match: valeurs manquantes considérées comme absence d'événement (0)")
    st.write("Données joueurs: valeurs manquantes considérées comme absence d'événement (0)")

    st.caption("Regroupements de variables")

    st.caption("Création de variables")



    
    