import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

import os
os.chdir('C:/Users/LANDRYGOL/Documents/GitHub/Bookmakerspy/streamlit')

sns.set_theme(style="white")
sns.set_palette("pastel")

df = pd.read_csv('./data/df_results.csv')
df.rename(columns={'FTHG': 'home_goal', 'FTAG': 'away_goal'}, inplace = 'True')
df['total_goal'] = df['home_goal'] + df['away_goal']

def app():

    st.header("Bookmakerspy - Simulation par un modèle de Poisson")
    st.subheader("Rappels sur le nombre de buts par match")
    
    fig = plt.figure()
    g = sns.histplot(data = df, x = 'total_goal', bins=10, hue = 'season', kde = True, stat = 'density', multiple = 'dodge');
    g.set_xticks(range(10))