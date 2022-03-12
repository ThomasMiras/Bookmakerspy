import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

import os
os.chdir('C:/Users/LANDRYGOL/Documents/GitHub/Bookmakerspy/streamlit')

sns.set_theme(style="white")
sns.set_palette("pastel")

def app():

    df = pd.read_csv('./data/df_results.csv', index_col = 0)
    df.rename(columns={'FTHG': 'home_goal', 'FTAG': 'away_goal'}, inplace = 'True')
    df['total_goal'] = df['home_goal'] + df['away_goal']

    st.header("Simulation par un modèle de Poisson")
    st.subheader("Rappels sur le nombre de buts par match")
    
    st.dataframe(df)
    fig = plt.figure(figsize=(5, 5))
    g = sns.histplot(data = df, x = 'total_goal', bins = 10, hue = 'season', kde = True, stat = 'density', multiple = 'dodge');
    g.set_xticks(range(10))
    st.pyplot(fig)
    
    
    #fig, (g1, g2) = plt.subplots(1, 2)
    #g1 = sns.histplot(data = df, x = 'home_goal', bins = 10, hue = 'season', kde = True, stat = 'density', multiple = 'dodge');
    #g1.set_xticks(range(10))
    #st.pyplot(g1)
    #g2 = sns.histplot(data = df, x = 'away_goal', bins = 10, hue = 'season', kde = True, stat = 'density', multiple = 'dodge');
    #g2.set_xticks(range(10))
    #st.pyplot(g2)