import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
sns.set_palette("pastel")

def app():

    st.header("Bookmakerspy - Exploration & dataviz")
    df = pd.read_csv('./data/df_results.csv')
    df_results = df.copy()
    
    df.rename(columns={'FTR': 'Match_Result'}, inplace = 'True')
    df['Match_Result'].replace(to_replace=['H', 'D', 'A'],value=['Home', 'Draw', 'Away'], inplace = True)

    df['total_goal'] = df['FTHG'] + df['FTAG']

    st.subheader("Part des résultats de matchs (H/D/A)")
    st.caption("Jouer à domicile semble avoir un impact important sur le résultat du match")

    
    x = df['Match_Result'].value_counts().reset_index().drop(columns='index').squeeze()
    
    fig = plt.figure(figsize=(6,6))
    colors = sns.color_palette('pastel')

    plt.pie(x, labels = ['Home', 'Away', 'Draw'],
            autopct = lambda x: str(round(x, 2)) + '%',
            pctdistance = 0.7, labeldistance = 1.1, colors = [colors[1],colors[0],colors[2]]
                )
    plt.legend()
    plt.title('Home-field advantage')

    col1, col2 = st.columns([2, 2])
    with col1:
        st.pyplot(fig)

    st.subheader("Impact de la notation de l'équipe")
    st.caption("Avec les données post-match")

    df_orig = pd.read_csv("./data/df_stats_odds.csv")
    df_orig.rename(columns={'FTR': 'Match_Result'}, inplace = 'True')
    df_orig['Match_Result'].replace(to_replace=['H', 'D', 'A'],value=['Home', 'Draw', 'Away'], inplace = True)

    fig = sns.relplot(x='home_team_rating', y='away_team_rating', hue = 'Match_Result', hue_order=['Away', 'Home', 'Draw'], data=df_orig)
    plt.plot([5,8], [5,8], '--k', alpha = 0.5)
    plt.text(5.1, 6.5,'Away  > Home')
    plt.text(6, 5.5,'Home > Away') 
    plt.title('Team Rating impact on result')
    plt.axis([5, 8, 5, 8])
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)

    st.caption("Avec les moyennes des 3 derniers matchs")

    fig = sns.relplot(x='home_team_rating', y='away_team_rating', hue = 'Match_Result', hue_order=['Away', 'Home', 'Draw'], data=df);
    plt.plot([5,8], [5,8], '--k', alpha = 0.5)
    plt.text(5.1, 6.5,'Away  > Home') 
    plt.text(6, 5.5,'Home > Away') 
    plt.title('Team Rating impact on result')
    plt.axis([5, 8, 5, 8]);
   
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)

    st.subheader("Analyse d'impact des features")
    st.caption("Total scoring attempts")

    for ind in range(len(df)):
        if df.loc[ind,'Match_Result']=='Home':
            df.loc[ind,'total_scoring_att_winning_team'] = df.loc[ind,'home_total_scoring_att']
            df.loc[ind,'total_scoring_att_losing_team']  = df.loc[ind,'away_total_scoring_att']
        if df.loc[ind,'Match_Result']=='Away':
            df.loc[ind,'total_scoring_att_winning_team'] = df.loc[ind,'away_total_scoring_att']
            df.loc[ind,'total_scoring_att_losing_team']  = df.loc[ind,'home_total_scoring_att']
            
    fig, ax1 = plt.subplots(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.distplot(a=df['home_total_scoring_att'], color='#ffb482', label='home team scoring attempts')
    sns.distplot(a=df['away_total_scoring_att'], color='#a1c9f4', label='away team scoring attempts')
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xlabel('Home/Away total scoring attempts')
    plt.legend()
    plt.subplot(1,2,2)
    sns.distplot(a=df['total_scoring_att_winning_team'], color='#ff9f9b', label='winning team scoring attempts');
    sns.distplot(a=df['total_scoring_att_losing_team'], color='#d0bbff', label='losing team scoring attempts');
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xlabel('total scoring attempts impact on the result')
    plt.legend()
    
    st.pyplot(fig)

    st.caption("Possession percentage")
    
    for ind in range(len(df)):
        if df.loc[ind,'Match_Result']=='Home':
            df.loc[ind,'possession_percentage_winning_team'] = df.loc[ind,'home_possession_percentage']
            df.loc[ind,'possession_percentage_losing_team']  = df.loc[ind,'away_possession_percentage']
        if df.loc[ind,'Match_Result']=='Away':
            df.loc[ind,'possession_percentage_winning_team'] = df.loc[ind,'away_possession_percentage']
            df.loc[ind,'possession_percentage_losing_team']  = df.loc[ind,'home_possession_percentage']
        
    fig, ax1 = plt.subplots(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.distplot(a=df['home_possession_percentage'], color='#ffb482', label='home team possession percentage')
    sns.distplot(a=df['away_possession_percentage'], color='#a1c9f4', label='away team possession percentage')
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xlabel('Home/Away possession_percentage')
    plt.legend()
    plt.subplot(1,2,2)
    sns.distplot(a=df['possession_percentage_winning_team'], color='#ff9f9b', label='winning team possession percentage');
    sns.distplot(a=df['possession_percentage_losing_team'], color='#d0bbff', label='losing team possession percentage');
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xlabel('possession_percentage impact on the result')
    plt.legend()
    st.pyplot(fig)

    st.caption("Team rating")

    for ind in range(len(df)):
        if df.loc[ind,'Match_Result']=='Home':
            df.loc[ind,'team_rating_winning_team'] = df.loc[ind,'home_team_rating']
            df.loc[ind,'team_rating_losing_team']  = df.loc[ind,'away_team_rating']
        if df.loc[ind,'Match_Result']=='Away':
            df.loc[ind,'team_rating_winning_team'] = df.loc[ind,'away_team_rating']
            df.loc[ind,'team_rating_losing_team']  = df.loc[ind,'home_team_rating']
            
    fig, ax1 = plt.subplots(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.distplot(a=df['home_team_rating'], color='#ffb482', label='home team rating')
    sns.distplot(a=df['away_team_rating'], color='#a1c9f4', label='away team rating')
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xlabel('Home/Away team rating')
    plt.legend()
    plt.subplot(1,2,2)
    sns.distplot(a=df['team_rating_winning_team'], color='#ff9f9b', label='winning team team_rating');
    sns.distplot(a=df['team_rating_losing_team'], color='#d0bbff', label='losing team team_rating');
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xlabel('team rating impact on the result')
    plt.legend()
    st.pyplot(fig)

    st.subheader("Résumé des corrélations")
    df_corrs_home = pd.read_csv('./data/df_corrs_home.csv')
    df_corrs_away = pd.read_csv('./data/df_corrs_away.csv')
    
    st.caption("Corrélations avec les résultats Home et le nombre de goals Home")
    fig = sns.catplot(x="variable", y="corr", col="type", data=df_corrs_home, kind="bar", col_wrap=1, height=3, aspect=4)
    fig.set_xticklabels(rotation=90)
    st.pyplot(fig)

    st.caption("Corrélations avec les résultats Away et le nombre de goals Away")
    fig = sns.catplot(x="variable", y="corr", col="type", data=df_corrs_away, kind="bar", col_wrap=1, height=3, aspect=4)
    fig.set_xticklabels(rotation=90)
    st.pyplot(fig)

    st.subheader("Prédictions des bookmakers")
    st.caption("Analyse des mauvaises prédictions")
    
    bookmaker_list = ['B365', 'LB', 'PS', 'WH', 'VC', 'PSC']
    # computing bookmakers results and appending related cols 

    @st.cache
    def bookmaker_results(bookmaker_name, df):
        results = []

        for i in range(len(df[bookmaker_name + 'H'])):
            if df.loc[i, [bookmaker_name + 'H', bookmaker_name + 'D', bookmaker_name + 'A']].min() == df[bookmaker_name + 'H'][i]:
                results.append('H')
            elif df.loc[i, [bookmaker_name+'H', bookmaker_name + 'D', bookmaker_name + 'A']].min() == df[bookmaker_name + 'D'][i]:
                results.append('D')
            elif df.loc[i, [bookmaker_name+'H', bookmaker_name + 'D', bookmaker_name + 'A']].min() == df[bookmaker_name + 'A'][i]:
                results.append('A')
        
        return results
    
    for bookmaker in bookmaker_list:
        df[bookmaker + 'R'] = bookmaker_results(bookmaker, df)
    
    df = df.reset_index(drop=True).set_index('match_id')

    def res_cote(FTR,cote):
        return 'Correct' if cote == FTR else 'Wrong prediction'

    numbers = df_results['FTR'].reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = df.join(numbers)

    bookmaker_list_R = ['B365R', 'LBR', 'PSR', 'WHR', 'VCR', 'PSCR'] 
    for cote in bookmaker_list_R:
        df[f"{cote}_v"] = df.apply(lambda row: res_cote(row['FTR'],row[cote]), axis=1)

    cols_cotes = ['B365R_v', 'LBR_v', 'PSR_v', 'WHR_v', 'VCR_v', 'PSCR_v']
    f = plt.figure(figsize=(15, 15))
    gs = f.add_gridspec(3, 2)
    pos=0

    for bm in cols_cotes:
        row = 0 if pos <=2 else 1 
        col = pos if pos <=2 else pos-3
        df_non_predit = df[df[bm] == 'Wrong prediction']
        ax = f.add_subplot(gs[col, row])
        ax = sns.countplot(x=bm, hue="FTR", hue_order=['A', 'H', 'D'], data=df_non_predit)
        ax.set_xlabel(bm[:-3])
        for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=15)
        pos += 1
    
    st.pyplot(f)

    st.subheader("Distribution des probabilités pour le nombre total de goals par match")
    st.caption("Le graphe semble indiquer une distribution de Poisson")

    
    fig = plt.figure()
    g = sns.histplot(data=df, x='total_goal', bins=10, hue='season', kde=True, stat='density', multiple='dodge');
    sns.despine(top=True, right=True, left=False, bottom=False)
    g.set_xticks(range(10)) 

    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    
    fig1 = plt.figure()
    g = sns.histplot(data=df, x='FTHG', bins=9, hue='season', kde=True, stat='density', multiple='dodge');
    sns.despine(top=True, right=True, left=False, bottom=False)
    g.set_xticks(range(10))
    plt.xlabel('Full time home team goals')

    fig2 = plt.figure()
    g = sns.histplot(data=df, x='FTAG', bins=9, hue='season', kde=True, stat='density', multiple='dodge');
    sns.despine(top=True, right=True, left=False, bottom=False)
    g.set_xticks(range(10))
    plt.xlabel('Full time away team goals')   

    with col2:
        st.pyplot(fig1)
        st.pyplot(fig2)
