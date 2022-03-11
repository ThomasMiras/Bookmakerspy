import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
sns.set_palette("pastel")

def app():

    st.header("Bookmakerspy - Exploration & dataviz")
    st.subheader("Distribution des probabilités pour le nombre total de goals par match")
    st.caption("Le graphe semble indiquer une distribution de Poisson")
    df = pd.read_csv('./data/df_results.csv')
    df_orig = df.copy()
    
    df.rename(columns={'FTR': 'Match_Result'}, inplace = 'True')
    df['Match_Result'].replace(to_replace=['H', 'D', 'A'],value=['Home', 'Draw', 'Away'], inplace = True)

    df['total_goal'] = df['FTHG'] + df['FTAG']
    
    fig = plt.figure()
    g = sns.histplot(data=df, x='total_goal', bins=10, hue='season', kde=True, stat='density', multiple='dodge');
    #sns.despine(top=True, right=True, left=False, bottom=False)
    g.set_xticks(range(10))

    st.pyplot(fig)

    st.subheader("Part des résultats de matchs (H/D/A)")
    st.caption("Jouer à domicile semble avoir un impact important sur le résultat du match")

    
    x = df['Match_Result'].value_counts().reset_index().drop(columns='index').squeeze()
    
    fig = plt.figure(figsize=(6,6))
    colors = sns.color_palette('pastel')

    plt.pie(x, labels = ['Home', 'Away', 'Draw'],
            autopct = lambda x: str(round(x, 2)) + '%',
            pctdistance = 0.7, labeldistance = 1.1, colors = colors
                )
    plt.legend()
    plt.title('Home-field advantage')

    st.pyplot(fig)

    st.subheader("Impact de la notation de l'équipe")
    st.caption("Avec les données post-match")

    df_orig.rename(columns={'FTR': 'Match_Result'}, inplace = 'True')
    df_orig['Match_Result'].replace(to_replace=['H', 'D', 'A'],value=['Home', 'Draw', 'Away'], inplace = True)

    
    fig = plt.figure(figsize=(6,6))
   
    g = sns.relplot(x='home_team_rating', y='away_team_rating', hue = 'Match_Result', hue_order=['Away', 'Home', 'Draw'], data=df_orig)
    
    plt.plot([5,8], [5,8], '--k', alpha = 0.5)
    plt.text(5.1, 6.5,'Away  > Home') 
    plt.text(6, 5.5,'Home > Away') 
    plt.title('Team Rating impact on result')
    plt.axis([5, 8, 5, 8])
    
    st.pyplot(fig)





