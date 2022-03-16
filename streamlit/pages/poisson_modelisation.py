import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import os

sns.set_theme(style = 'white')
sns.set_palette('pastel')

def app():
    
    df = pd.read_csv('./data/df_results.csv', index_col = 0)
    df.rename(columns={'FTHG': 'home_goal', 'FTAG': 'away_goal'}, inplace = 'True')
    df['total_goal'] = df['home_goal'] + df['away_goal']
    
    coeffs = pd.DataFrame([[-0.0074, 0.0099, -0.5456, 0.0085, 0.1875, 0.0175, 0.0240, -0.1687, -0.0200, 0.1342],
                          [-0.0033, 0.0101, -2.0334, 0.0279, -0.2725, 0.0350, 0.0446, 0.1306, 0.1001, 0.0300]],
                          index = ["Home", "Away"],
                          columns = pd.Index(['aerial_won', 'ontarget_scoring_att', 'pass', 
                                              'possession_percentage', 'team_rating', 'won_contest', 'won_corners', 
                                              'midfielder_player_rating', 'forward_player_rating', 'FTG_mean'], name = 'Model'))
    
    
    features = pd.DataFrame([[19.33333, 7, 0.83112, 56.83333, 6.85429, 14.33333, 9, 6.93083, 7.68778, 1.33333],
                             [17.66667, 4, 0.82617, 63.76667, 6.73786, 10.66667, 4, 6.91444, 7.14444, 1.33333]],
                            index = ["Chelsea", "Liverpool"],
                            columns = pd.Index(['aerial_won', 'ontarget_scoring_att', 'pass', 
                                                'possession_percentage', 'team_rating', 'won_contest', 'won_corners', 
                                                'midfielder_player_rating', 'forward_player_rating', 'FTG_mean'], name = 'Model'))
    

    st.header('Simulation par un modèle de Poisson')
    st.caption('Rappels sur le nombre de buts par match')
    
    
    col1, col2 = st.columns(2)
        
    fig1 = plt.figure()
    g1 = sns.histplot(data = df, x = 'home_goal', bins = 9, hue = 'season', kde = True, stat = 'density', multiple = 'dodge');
    sns.despine(top = True, right = True, left = False, bottom = False)
    g1.set_xticks(range(9))
    plt.xlabel('Full time home goals')
    
    with col1:
        st.pyplot(fig1)
    
    fig2 = plt.figure()
    g2 = sns.histplot(data = df, x = 'away_goal', bins = 9, hue = 'season', kde = True, stat = 'density', multiple = 'dodge');
    sns.despine(top = True, right = True, left = False, bottom = False)
    g2.set_xticks(range(9))
    plt.xlabel('Full time away goals')
    
    with col2:
        st.pyplot(fig2)
    
    
    st.caption('Les coefficients du modèle sont les suivants:')    
    st.dataframe(coeffs.style.format("{:20,.4f}"))
    
        
    st.caption("Exemple d'application: match Chelsea - Liverpool du 6 Mai 2018")   
    st.dataframe(features.style.format("{:20,.5f}"))
    
    
    st.caption('Les valeurs de lambda sont les suivantes:')
    
    params = pd.DataFrame(index = ["Chelsea", "Liverpool"], columns = pd.Index(['λ'], name = 'Model'))
    params['λ'] = list(np.exp(np.sum(np.multiply(coeffs, features), axis = 1)))
    
    st.dataframe(params.style.format("{:20,.5f}"))
    
   
    @st.cache
    def match_modelisation():
        nb_simu = 10000
        m = {i: pd.DataFrame(columns = range(nb_simu)) for i in ['home', 'away']}
        s = pd.DataFrame(columns = pd.MultiIndex.from_product([['Away'], list(np.arange(9))]), index = pd.MultiIndex.from_product([['Home'], list(np.arange(9))]))
        p = pd.DataFrame(columns = ['Home', 'Draw', 'Away'], index = ['proba'])

        m['home'] = np.random.poisson(params['λ'][0], nb_simu)
        m['away'] = np.random.poisson(params['λ'][1], nb_simu)
        m['score'] = [str(u) + '-' + str(v) for (u, v) in zip(m['home'], m['away'])]
    
        for i in range(9):
            for j in range(9):
                s[('Away', j)][('Home', i)] = 100 * sum([1 if (u == i) & (v == j) else 0 for (u, v) in zip (m['home'], m['away'])])
                
        s = s / nb_simu
        
        probaH = 0
        probaD = 0
        probaA = 0
    
        for i in range(9):
            for j in range(9):
                if i > j:
                    probaH = probaH + s[('Away', j)][('Home', i)]
                if i == j:
                    probaD = probaD + s[('Away', j)][('Home', i)]
                if i < j:
                    probaA = probaA + s[('Away', j)][('Home', i)]
    
        p['Home']['proba'] = round(probaH, 2)
        p['Draw']['proba'] = round(probaD, 2)
        p['Away']['proba'] = round(probaA, 2)
    
        return m, s, p
    
    def cell_color(df):
        color = pd.DataFrame(index = df.index, columns = df.columns)
        for i in range(9):
            for j in range(9):
                if i > j:
                    color[('Away', j)][('Home', i)] = 'background-color: #ffb482;'
                if i == j:
                    color[('Away', j)][('Home', i)] = 'background-color: #8de5a1;'
                if i < j:
                    color[('Away', j)][('Home', i)] = 'background-color: #a1c9f4;'
        return color
    
    def cell_color_prediction(df):
        color = pd.DataFrame(index = df.index, columns = df.columns)
        for i in range(9):
            for j in range(9):
                if i > j:
                    color[('Away', j)][('Home', i)] = 'background-color: #ffb482;'
                    if df[('Away', j)][('Home', i)] == max(df.max()):
                        color[('Away', j)][('Home', i)] = color[('Away', j)][('Home', i)] + 'border: dashed black;'
                if i == j:
                    color[('Away', j)][('Home', i)] = 'background-color: #8de5a1;'
                    if df[('Away', j)][('Home', i)] == max(df.max()):
                        color[('Away', j)][('Home', i)] = color[('Away', j)][('Home', i)] + 'border: dashed black;'
                if i < j:
                    color[('Away', j)][('Home', i)] = 'background-color: #a1c9f4;'
                    if df[('Away', j)][('Home', i)] == max(df.max()):
                        color[('Away', j)][('Home', i)] = color[('Away', j)][('Home', i)] + 'border: dashed black;'
        return color
    
    def cell_color_probability(df):
        color = pd.DataFrame(index = df.index, columns = df.columns)
        color['Home']['proba'] = 'background-color: #ffb482;'
        color['Draw']['proba'] = 'background-color: #8de5a1;'
        color['Away']['proba'] = 'background-color: #a1c9f4;'
        return color

    
    match, scorelines, proba = match_modelisation()
    
    df = pd.DataFrame(index = ["Chelsea", "Liverpool", "Result"], columns = pd.Index(['Match 1', 'Match 2', 'Match 3', 'Match 4', 'Match 5'], name = 'Model'))
    df.loc['Chelsea', :] = [str(e) for e in match['home'][0:5]]
    df.loc['Liverpool', :] = [str(e) for e in match['away'][0:5]]
    df.loc['Result', :] = ['H' if u > v else 'D' if u == v else 'A' for (u, v) in zip(df.loc['Chelsea', :], df.loc['Liverpool', :])]
    
    
    st.caption('Ce qui donne le tableau suivant:')
    
    st.dataframe(df.style.apply(lambda x: ['background-color: #ffb482;' if v =='H' else 'background-color: #8de5a1;' if v == 'D' else 'background-color: #a1c9f4;' if v == 'A' else 'background-color: #ffffff;' for v in x]))
    
    
    st.markdown("Deux méthodes sont possibles pour déterminer le résultat")
    st.markdown("* En fonction du score probable")
    st.markdown("* En fonction de la probabilité des résultats")  
    
    prediction = st.checkbox('afficher les prédictions')
    
    if prediction:
        st.dataframe(scorelines.style.apply(cell_color_prediction, axis = None).format("{:20,.2f}"))
        st.dataframe(proba.style.apply(cell_color_probability, axis = None).format("{:20,.2f}"))
    else:
        st.dataframe(scorelines.style.apply(cell_color, axis = None).format("{:20,.2f}"))
        
    
    @st.cache
    def get_data_poisson():
        
        data = pd.read_csv('./data/df_results.csv', index_col = 0)
        data.rename(columns={'FTHG': 'home_goal', 'FTAG': 'away_goal'}, inplace = 'True')
        data['total_goal'] = data['home_goal'] + data['away_goal']
        
        nb_simu = 1000

        feats = {i:[] for i in ["home", "away"]}
        simu = {i: pd.DataFrame(index = data.index, columns = range(nb_simu)) for i in ["home", "away", "score"]}
        predictions = pd.DataFrame(columns = ['Score', 'HomeGoalsPrediction', 'AwayGoalsPrediction', 'PredictScore','ProbaHome', 'ProbaDraw', 'ProbaAway', 'PredictProba'], index = data.index)

        for i in ['home', 'away']:
            feats[i] = [i + '_aerial_won', i + '_ontarget_scoring_att', i + '_pass', i + '_possession_percentage', i + '_team_rating', 
              i + '_won_contest', i + '_won_corners', 'midfielder_' + i + '_player_rating', 'forward_' + i + '_player_rating', 'FT' + i.upper()[0] + 'G_mean']
            data['λ_' + i] = np.exp(np.sum(np.multiply(data[feats[i]], coeffs.loc[i.capitalize()]), axis = 1))

            for j in data.index:
                simu[i].loc[j, 0:nb_simu] = np.random.poisson(data['λ_' + i][j] , nb_simu)

        for i in data.index:
            simu["score"].loc[i, 0:nb_simu] = [str(u) + "-" + str(v) for (u, v) in zip (simu["home"].loc[i, 0:nb_simu], simu["away"].loc[i, 0:nb_simu])]
            predictions.loc[i]['Score'] = list(Counter(simu["score"].loc[i]).keys())[list(Counter(simu["score"].loc[i]).values()).index(Counter(simu["score"].loc[i]).most_common(1)[0][1])]
            predictions.loc[i]['HomeGoalsPrediction'] = int(predictions['Score'][i].split("-")[0])
            predictions.loc[i]['AwayGoalsPrediction'] = int(predictions['Score'][i].split("-")[1])
            if int(predictions['Score'][i].split("-")[0]) > int(predictions['Score'][i].split("-")[1]):
                predictions['PredictScore'][i] = "H"
            if int(predictions['Score'][i].split("-")[0]) == int(predictions['Score'][i].split("-")[1]):
                predictions['PredictScore'][i] = "D"
            if int(predictions['Score'][i].split("-")[0]) < int(predictions['Score'][i].split("-")[1]):
                predictions['PredictScore'][i] = "A"

            score = pd.DataFrame(0, columns = range(9), index = range(9))

            probaH = 0
            probaD = 0
            probaA = 0

            for h in range(9):
                for a in range(9):
                    score[h][a] = sum([1 if (u == h) & (v == a) else 0 for (u, v) in zip (simu['home'].loc[i, :], simu['away'].loc[60, :])])
  
            predictions.loc[i][['ProbaHome', 'ProbaDraw', 'ProbaAway']] = [sum(sum(list(np.tril(score, -1)))), sum(list(np.diag(score))), sum(sum(list(np.triu(score, 1))))]
            predictions.loc[i][['ProbaHome', 'ProbaDraw', 'ProbaAway']] = predictions.loc[i][['ProbaHome', 'ProbaDraw', 'ProbaAway']] / 10

            if max(predictions.loc[i][['ProbaHome', 'ProbaDraw', 'ProbaAway']]) == predictions.loc[i]['ProbaHome']:
                predictions.loc[i]['PredictProba'] = 'H'
            if max(predictions.loc[i][['ProbaHome', 'ProbaDraw', 'ProbaAway']]) == predictions.loc[i]['ProbaDraw']:
                predictions.loc[i]['PredictProba'] = 'D'
            if max(predictions.loc[i][['ProbaHome', 'ProbaDraw', 'ProbaAway']]) == predictions.loc[i]['ProbaAway']:
                predictions.loc[i]['PredictProba'] = 'A'

        data = data.merge(predictions, left_index = True, right_index = True)
            
        return data
    
    
    
    df_poisson = get_data_poisson()
    
    def create_dashboard_result(data):

        y_train = data.loc[data["season"] != "2017_2018"]["FTR"]
        y_train_pred_score = data.loc[data["season"] != "2017_2018"]["PredictScore"]
        y_train_pred_proba = data.loc[data["season"] != "2017_2018"]["PredictProba"]

        y_test = data.loc[data["season"] == "2017_2018"]["FTR"]
        y_test_pred_score = data.loc[data["season"] == "2017_2018"]["PredictScore"]
        y_test_pred_proba = data.loc[data["season"] == "2017_2018"]["PredictProba"]
        
        col1, col2, col3 = st.columns([1, .2, 1])
 
        with col1:
            st.write('Prédictions via le score le plus probable')
            
            f1_score_s = list(f1_score(y_test, y_test_pred_score, average = None))
            conf_matrix_s = confusion_matrix(y_test, y_test_pred_score)

            fig, ax = plt.subplots()
            ax.matshow(conf_matrix_s, cmap = plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix_s.shape[0]):
                for j in range(conf_matrix_s.shape[1]):
                    ax.text(x = j, y = i, s = conf_matrix_s[i, j], va = 'center', ha = 'center', fontsize = 18)
            
            labels = ['Away', 'Draw', 'Home']
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            ax.xaxis.set_label_position('top')
            plt.xlabel('Predicted', fontsize = 16)
            plt.ylabel('Real', fontsize = 16)
            st.pyplot(fig)

            d = {'Accuracy Train': [round(accuracy_score(y_train, y_train_pred_score), 2)], 'Accuracy Test': [round(accuracy_score(y_test, y_test_pred_score), 2)]}
            
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            st.markdown(hide_table_row_index, unsafe_allow_html = True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))

            st.caption("F1 Score")
            fig, ax = plt.subplots()
            
            ax.bar(x = ['Away', 'Draw', 'Home'], height = f1_score_s, color = ["#a1c9f4", "#8de5a1", "#ffb482"])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize = 16)
            plt.xticks(fontsize = 16)
            st.pyplot(fig)

            
        with col3:
            
            st.write('Prédictions via la somme des probabilités')

            f1_score_p = list(f1_score(y_test, y_test_pred_proba, average = None))
            conf_matrix_p = confusion_matrix(y_test, y_test_pred_proba)
            

            fig, ax = plt.subplots()
            ax.matshow(conf_matrix_p, cmap = plt.cm.Blues, alpha = 0.3)
            for i in range(conf_matrix_p.shape[0]):
                for j in range(conf_matrix_p.shape[1]):
                    ax.text(x = j, y = i, s = conf_matrix_p[i, j], va = 'center', ha = 'center', fontsize = 18)
            
            labels = ['Away', 'Draw', 'Home']
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            ax.xaxis.set_label_position('top')
            plt.xlabel('Predicted', fontsize = 16)
            plt.ylabel('Real', fontsize = 16)
            st.pyplot(fig)

            d = {'Accuracy Train': [round(accuracy_score(y_train, y_train_pred_proba), 2)], 'Accuracy Test': [round(accuracy_score(y_test, y_test_pred_proba), 2)]}
            
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            st.markdown(hide_table_row_index, unsafe_allow_html = True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))

            
            st.caption("F1 Score")
            fig, ax = plt.subplots()
            ax.bar(x = ['Away','Draw','Home'], height = f1_score_p, color = ["#a1c9f4", "#8de5a1", "#ffb482"])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize = 16)
            plt.xticks(fontsize = 16)
            st.pyplot(fig)
            
    create_dashboard_result(df_poisson)
    
    
    def create_dashboard_scoreline(data):

        y_train_home = data.loc[data["season"] != "2017_2018"]["home_goal"]
        y_train_pred_home = data.loc[data["season"] != "2017_2018"]["HomeGoalsPrediction"]
        
        y_test_home = data.loc[data["season"] == "2017_2018"]["home_goal"]
        y_test_pred_home = data.loc[data["season"] == "2017_2018"]["HomeGoalsPrediction"]
        

        y_train_away = data.loc[data["season"] != "2017_2018"]["away_goal"]
        y_train_pred_away = data.loc[data["season"] != "2017_2018"]["AwayGoalsPrediction"]
        
        y_test_away = data.loc[data["season"] == "2017_2018"]["away_goal"]
        y_test_pred_away = data.loc[data["season"] == "2017_2018"]["AwayGoalsPrediction"]
        
        col1, col2, col3 = st.columns([1, .2, 1])
 
        with col1:
            st.write("Buts de l'équipe à domicile")
            
            f1_score_home = list(f1_score(list(y_test_home), list(y_test_pred_home), average = None))
            conf_matrix_home = confusion_matrix(list(y_test_home), list(y_test_pred_home))

            fig, ax = plt.subplots()
            ax.matshow(conf_matrix_home, cmap = plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix_home.shape[0]):
                for j in range(conf_matrix_home.shape[1]):
                    ax.text(x = j, y = i, s = conf_matrix_home[i, j], va = 'center', ha = 'center', fontsize = 18)
            
            labels = list(np.arange(conf_matrix_home.shape[0]))
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            ax.xaxis.set_label_position('top')
            plt.xlabel('Predicted', fontsize = 16)
            plt.ylabel('Real', fontsize = 16)
            st.pyplot(fig)

            d = {'Accuracy Train': [round(accuracy_score(list(y_train_home), list(y_train_pred_home)), 2)], 'Accuracy Test': [round(accuracy_score(list(y_test_home), list(y_test_pred_home)), 2)]}
            
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            st.markdown(hide_table_row_index, unsafe_allow_html = True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))

            st.caption("F1 Score")
            fig, ax = plt.subplots()
            
            ax.bar(x = range(conf_matrix_home.shape[0]), height = f1_score_home)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize = 16)
            plt.xticks(fontsize = 16)
            st.pyplot(fig)

            
        with col3:
            
            st.write("Buts de l'équipe à l'extérieur")

            f1_score_away = list(f1_score(list(y_test_away), list(y_test_pred_away), average = None))
            conf_matrix_away = confusion_matrix(list(y_test_away), list(y_test_pred_away))
            

            fig, ax = plt.subplots()
            ax.matshow(conf_matrix_away, cmap = plt.cm.Blues, alpha = 0.3)
            for i in range(conf_matrix_away.shape[0]):
                for j in range(conf_matrix_away.shape[1]):
                    ax.text(x = j, y = i, s = conf_matrix_away[i, j], va = 'center', ha = 'center', fontsize = 18)
            
            labels = list(np.arange(conf_matrix_away.shape[0]))
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            ax.xaxis.set_label_position('top')
            plt.xlabel('Predicted', fontsize = 16)
            plt.ylabel('Real', fontsize = 16)
            st.pyplot(fig)

            d = {'Accuracy Train': [round(accuracy_score(list(y_train_away), list(y_train_pred_away)), 2)], 'Accuracy Test': [round(accuracy_score(list(y_test_away), list(y_test_pred_away)), 2)]}
            
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            st.markdown(hide_table_row_index, unsafe_allow_html = True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))

            
            st.caption("F1 Score")
            fig, ax = plt.subplots()
            ax.bar(x = range(conf_matrix_away.shape[0]), height = f1_score_away)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize = 16)
            plt.xticks(fontsize = 16)
            st.pyplot(fig)
            
    create_dashboard_scoreline(df_poisson)
    