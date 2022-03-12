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

    st.caption("Dataframe cotes")
    df_odds = pd.read_csv('./data/df_odds.csv')
    st.dataframe(df_odds)

    st.subheader("Pre-processing")
    
    st.caption("Jointures")
    img_jointures = Image.open('./data/img/jointures_df.png')
    st.image(img_jointures)

    st.caption("Traitement des valeurs manquantes")
    st.write("Données de match: valeurs manquantes considérées comme absence d'événement (0)")
    st.write("Données joueurs: les joueurs non notés (0) n'ont pas été pris en compte pour calculer les moyennes par position.")

    st.caption("Suppression de variables")
    img_supp = Image.open('./data/img/suppression_variables.png')
    st.image(img_supp)

    st.caption("Regroupements de variables")
    st.write("home/away pass: ratio entre home/away accurate pass et home/away total pass")
    
    st.caption("Création de variables")
    st.write("Différences entre les notes joueurs home et away")
    st.write("Différences entre le nombre de goals home et away")

    st.caption("Moyennes des 3 derniers matchs")
    st.write("Moyennes effectuées sur les stats pour chaque équipe sur les 3 derniers matchs où une équipe a joué dans la même situation (home/away)")



    
    