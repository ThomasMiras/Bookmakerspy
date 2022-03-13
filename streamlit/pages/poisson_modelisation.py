import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

import os
os.chdir('C:/Users/LANDRYGOL/Documents/GitHub/Bookmakerspy/streamlit')

sns.set_theme(style = 'white')
sns.set_palette('pastel')

def app():

    df = pd.read_csv('./data/df_results.csv', index_col = 0)
    df.rename(columns={'FTHG': 'home_goal', 'FTAG': 'away_goal'}, inplace = 'True')
    df['total_goal'] = df['home_goal'] + df['away_goal']

    st.header('Simulation par un modèle de Poisson')
    st.markdown('Rappels sur le nombre de buts par match')
    
    
    season = st.selectbox('season: ', sorted(df['season'].unique()))
    df_selected = df.loc[df['season'] == season]
    
    fig = plt.figure(figsize = (2.5, 2.5))
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    g = sns.histplot(data = df_selected, x = 'total_goal', bins = 9, kde = True, stat = 'density', multiple = 'dodge', legend = False);
    g.set_xticks(range(9))
    st.pyplot(fig)
    
    
    st.markdown('Les coefficients du modèle sont les suivants:')
    
    coeffs = pd.DataFrame([[-0.0074, 0.0099, -0.5456, 0.0085, 0.1875, 0.0175, 0.0240, -0.1687, -0.0200, 0.1342],
                          [-0.0033, 0.0101, -2.0334, 0.0279, -0.2725, 0.0350, 0.0446, 0.1306, 0.1001, 0.0300]],
                          index = ["Home", "Away"],
                          columns = pd.Index(['aerial_won', 'ontarget_scoring_att', 'pass', 
                                              'possession_percentage', 'team_rating', 'won_contest', 'won_corners', 
                                              'midfielder_player_rating', 'forward_player_rating', 'FTG_mean'], name = 'Model'))
    
    st.dataframe(coeffs.style.format("{:20,.4f}"))
    
    
    st.markdown('Application de la simulation: match Chelsea - Liverpool du 6 Mai 2018')
    
    features = pd.DataFrame([[19.33333, 7, 0.83112, 56.83333, 6.85429, 14.33333, 9, 6.93083, 7.68778, 1.33333],
                             [17.66667, 4, 0.82617, 63.76667, 6.73786, 10.66667, 4, 6.91444, 7.14444, 1.33333]],
                            index = ["Chelsea", "Liverpool"],
                            columns = pd.Index(['aerial_won', 'ontarget_scoring_att', 'pass', 
                                                'possession_percentage', 'team_rating', 'won_contest', 'won_corners', 
                                                'midfielder_player_rating', 'forward_player_rating', 'FTG_mean'], name = 'Model'))
    
    st.dataframe(features.style.format("{:20,.5f}"))
    
    
    st.markdown('Les valeurs de lambda sont les suivantes:')
    
    params = pd.DataFrame(index = ["Chelsea", "Liverpool"], columns = pd.Index(['λ'], name = 'Model'))
    params['λ'] = list(np.exp(np.sum(np.multiply(coeffs, features), axis = 1)))
    
    st.dataframe(params.style.format("{:20,.5f}"))
    
    
    st.markdown('Ce qui donne les tableaux suivants:')
    
    df = pd.DataFrame(index = ["Chelsea", "Liverpool", "Result"], columns = pd.Index(['Match 1', 'Match 2', 'Match 3', 'Match 4', 'Match 5'], name = 'Model'))
        
    nb_simu = 10000
    match = {i: pd.DataFrame(columns = range(nb_simu)) for i in ['home', 'away']}
    scorelines = pd.DataFrame(columns = pd.MultiIndex.from_product([['Away'], list(np.arange(9))]), index = pd.MultiIndex.from_product([['Home'], list(np.arange(9))]))

    match['home'] = np.random.poisson(params['λ'][0], nb_simu)
    match['away'] = np.random.poisson(params['λ'][1], nb_simu)
    match['score'] = [str(u) + '-' + str(v) for (u, v) in zip(match['home'], match['away'])]
    
    df.loc['Chelsea', :] = [str(e) for e in match['home'][0:5]]
    df.loc['Liverpool', :] = [str(e) for e in match['away'][0:5]]
    df.loc['Result', :] = ['H' if u > v else 'D' if u == v else 'A' for (u, v) in zip(df.loc['Chelsea', :], df.loc['Liverpool', :])]
    
    st.dataframe(df.style.apply(lambda x: ['background-color: #add8e6;' if v =='H' else 'background-color: #90ee90;' if v == 'D' else 'background-color: #ffcccb;' if v == 'A' else 'background-color: #ffffff;' for v in x]))
    

    for i in range(9):
      for j in range(9):
        scorelines[('Away', j)][('Home', i)] = 100 * sum([1 if (u == i) & (v == j) else 0 for (u, v) in zip (match['home'], match['away'])])
    
    scorelines = scorelines / nb_simu
    
    def cell_color(df):
        color = pd.DataFrame(index = df.index, columns = df.columns)
        for i in range(9):
            for j in range(9):
                if i > j:
                    color[('Away', j)][('Home', i)] = 'background-color: #add8e6;'
                    if df[('Away', j)][('Home', i)] == max(df.max()):
                        color[('Away', j)][('Home', i)] = color[('Away', j)][('Home', i)] + 'border: dashed black;'
                if i == j:
                    color[('Away', j)][('Home', i)] = 'background-color: #90ee90;'
                    if df[('Away', j)][('Home', i)] == max(df.max()):
                        color[('Away', j)][('Home', i)] = color[('Away', j)][('Home', i)] + 'border: dashed black;'
                if i < j:
                    color[('Away', j)][('Home', i)] = 'background-color: #ffcccb;'
                    if df[('Away', j)][('Home', i)] == max(df.max()):
                        color[('Away', j)][('Home', i)] = color[('Away', j)][('Home', i)] + 'border: dashed black;'
        return color
  
    st.dataframe(scorelines.style.apply(cell_color, axis = None).format("{:20,.2f}"))
    
    
    probaH = 0
    probaD = 0
    probaA = 0
    
    for i in range(9):
        for j in range(9):
            if i > j:
                probaH = probaH + scorelines[('Away', j)][('Home', i)]
            if i == j:
                probaD = probaD + scorelines[('Away', j)][('Home', i)]
            if i < j:
                probaA = probaA + scorelines[('Away', j)][('Home', i)]
    
    probaH = round(probaH, 2)
    probaD = round(probaD, 2)
    probaA = round(probaA, 2)
                
    st.write('La probabilité pour H est de ', probaH, '%')
    st.write('La probabilité pour D est de ', probaD, '%') 
    st.write('La probabilité pour A est de ', probaA, '%') 