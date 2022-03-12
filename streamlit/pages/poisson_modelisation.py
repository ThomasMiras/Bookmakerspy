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
    st.subheader('Rappels sur le nombre de buts par match')
    
    
    season = st.selectbox('season: ', sorted(df['season'].unique()))
    df_selected = df.loc[df['season'] == season]
    
    fig = plt.figure(figsize = (2.5, 2.5))
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    g = sns.histplot(data = df_selected, x = 'total_goal', bins = 9, kde = True, stat = 'density', multiple = 'dodge', legend = False);
    g.set_xticks(range(9))
    st.pyplot(fig)
    
    
    index_names = {'selector': '.index_name','props': 'color: white; background-color: #ffffff;'}
    headers = {'selector': 'th:not(.index_name)','props': 'border: 1px solid black; text-align: center; background-color: #d9d9d9;'}
    grid = {'selector': 'td', 'props': 'border: 1px solid black; text-align: center; background-color: #ffffff;'}

    coeffs = pd.DataFrame([[-0.0074, 0.0099, -0.5456, 0.0085, 0.1875, 0.0175, 0.0240, -0.1687, -0.0200, 0.1342],
                          [-0.0033, 0.0101, -2.0334, 0.0279, -0.2725, 0.0350, 0.0446, 0.1306, 0.1001, 0.0300]],
                          index = ["Home", "Away"],
                          columns = pd.Index(['aerial_won', 'ontarget_scoring_att', 'pass', 
                                              'possession_percentage', 'team_rating', 'won_contest', 'won_corners', 
                                              'midfielder_player_rating', 'forward_player_rating', 'FTG_mean'], name = 'Model'))
    
    st.dataframe(coeffs.style.set_table_styles([index_names, headers, grid]).format("{:20,.4f}"))
    
    
    features = pd.DataFrame([[19.33333, 7, 0.83112, 56.83333, 6.85429, 14.33333, 9, 6.93083, 7.68778, 1.33333],
                             [17.66667, 4, 0.82617, 63.76667, 6.73786, 10.66667, 4, 6.91444, 7.14444, 1.33333]],
                            index = ["Chelsea", "Liverpool"],
                            columns = pd.Index(['aerial_won', 'ontarget_scoring_att', 'pass', 
                                                'possession_percentage', 'team_rating', 'won_contest', 'won_corners', 
                                                'midfielder_player_rating', 'forward_player_rating', 'FTG_mean'], name = 'Model'))
    
    st.dataframe(features.style.set_table_styles([index_names, headers, grid]).format("{:20,.5f}"))
    
    
    params = pd.DataFrame(index = ["Chelsea", "Liverpool"], columns = pd.Index(['λ'], name = 'Model'))
    params['λ'] = list(np.exp(np.sum(np.multiply(coeffs, features), axis = 1)))
    
    st.dataframe(params.style.set_table_styles([index_names, headers, grid]).format("{:20,.5f}"))
    
    
    df = pd.DataFrame([[1, 0, 1, 1, 1], 
                  [1, 1, 0, 0, 2], 
                  ['D', 'A', 'H', 'H', 'A']], 
                  index = ["Chelsea", "Liverpool", "Result"],
                  columns = pd.Index(['Match 1', 'Match 2', 'Match 3', 'Match 4', 'Match 5'], name = 'Model'))

    st.dataframe(df.style.apply(lambda x: ['border: 1px solid black; text-align: center; background-color: #add8e6;' if v =='H' else 'border: 1px solid black; text-align: center; background-color: #90ee90;' if v == 'D' else 'border: 1px solid black; text-align: center; background-color: #ffcccb;' if v == 'A' else 'border: 1px solid black; text-align: center; background-color: #ffffff;' for v in x]))
