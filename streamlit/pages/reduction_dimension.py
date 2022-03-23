   
import streamlit as st
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score

warnings.filterwarnings('ignore')

os.getcwd() 

def app():
    st.header("Réduction de dimension")
    st.markdown("A partir des données post-traitées on va tester plusieurs approches de réduction de dimension de notre dataset en ayant                pour objectif :")
    st.markdown("* Faciliter l'apprentissage de nos modèles")
    st.markdown("* Garder une bonne performance du modèle d'apprentissage")   
    st.markdown(" Deux types de reductions sont testées:")
    st.markdown("* Feature selection (Wrapper / Logistic Regression)")
    st.markdown("* Transformation de variables (PCA / LDA / Manifolds)")  

    # Chargement et affichage
    df = pd.read_csv('./data/df_results.csv', index_col = 0)
    df = df.reset_index(drop=True).set_index('match_id')
    # Features
    feats_list = [ # Features for the 'home' team
       'home_team_rating', 'home_won_contest', 'home_possession_percentage', 'home_total_throws', 'home_blocked_scoring_att', 
       'home_total_scoring_att', 'home_total_tackle', 'home_aerial_won', 'home_aerial_lost', 'home_accurate_pass', 
       'home_total_pass', 'home_won_corners', 'home_shot_off_target', 'home_ontarget_scoring_att','home_total_offside', 
       'home_post_scoring_att', 'home_att_pen_goal', 'home_penalty_save', 'HF', 'HY', 'HR', 'home_pass', 
       'goalkeeper_home_player_rating', 'defender_home_player_rating', 'midfielder_home_player_rating', 'forward_home_player_rating', 
        'FTHG_mean',
               # Features for the 'away' team
       'away_team_rating', 'away_won_contest', 'away_possession_percentage', 'away_total_throws', 'away_blocked_scoring_att',
       'away_total_scoring_att', 'away_total_tackle', 'away_aerial_won', 'away_aerial_lost', 'away_accurate_pass', 
       'away_total_pass', 'away_won_corners', 'away_shot_off_target', 'away_ontarget_scoring_att', 'away_total_offside', 
       'away_post_scoring_att', 'away_att_pen_goal', 'away_penalty_save', 'AF', 'AY', 'AR', 'away_pass',
       'goalkeeper_away_player_rating', 'defender_away_player_rating', 'midfielder_away_player_rating', 'forward_away_player_rating', 
        'FTAG_mean',
        # Bookmakers odds
        # 'B365H', 'B365D', 'B365A', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD',
        # 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA'
    
        # Team Comparison
       'Diff_def_home_fwd_away', 'Diff_def_home_mid_away', 'Diff_mil_home_att_away', 'Diff_mil_home_mid_away',
       'Diff_mil_home_def_away', 'Diff_fwd_home_mid_away', 'Diff_fwd_home_def_away', 'Diff_Goal']
    # Target
    target_list = 'FTR'

    # Split by using shuffle parameter of train_test_split function
    data = df[feats_list]
    target = df[target_list]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, shuffle = False, random_state=789)
    
    st.write("Dataset après pre-processing, Nombre de features :{}".format(len(X_train.columns)))
    st.dataframe(X_train.head(10))
  

    #Standardisation
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)

########################################################################################################################################             
    st.markdown("## Selection des features") 
        
    st.markdown("### Wrapper RFECV")  
    st.markdown("Fonction RFECV permetant une cross validation, on utilise un estimateur DecisionTreeClassifier() car problème de classification")
    n_split_input = st.number_input('Nombre de blocs n_plit:', min_value=1, max_value=10, value=3, step=1)
    if(st.button('Calcul RFECV')):
        dt = DecisionTreeClassifier(random_state=123)
        crossval = KFold(n_splits = n_split_input, random_state = 2, shuffle = True)
        rfecv = RFECV(estimator=dt, cv = crossval, step=1)
        rfecv.fit(X_train, y_train)
        st.write("Nombre optimum de features : {}".format(rfecv.n_features_))
        fig, ax = plt.subplots(figsize=(15,5))
        ax.plot(np.mean(rfecv.grid_scores_,axis=1))
        ax.set_xlabel('Nombre de features')
        ax.set_xlabel('RFECV Score')
        st.pyplot(fig);
        #plt.show()
        
        
        
    st.markdown("### Logistic regression") 
    st.markdown("L'utilisation des coefficients de pénalités de la régression logistique permet d'éliminer des features.")
    st.markdown("Une regression par classe est faite. Pour chaque regression on affiche les coefficients d'importance.")
    C_input = st.number_input('Input C:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    l1_ratio_input = st.number_input('Input l1_ratio:', min_value=0.1, max_value=1.0, value=0.99, step=0.01)
    
    if(st.button('Calcul LogReg')):
        parametres = {'C':[C_input],'l1_ratio': [l1_ratio_input]}
        clf = linear_model.LogisticRegression(penalty = 'elasticnet', solver = 'saga', max_iter = 1000)
        grid_clf = GridSearchCV(estimator=clf, param_grid=parametres)
        grille = grid_clf.fit(X_train,y_train)
        optimal_clf = grid_clf.best_estimator_
        elast_coef = optimal_clf.coef_
        
        elast = np.abs(elast_coef)
        keep = np.where(~((elast[0,:]==0)&(elast[1,:]==0)&(elast[2,:]==0)))
        coeff_keep_0 = elast[0,keep[0]]
        index_keep_0 = X_train.columns.values[keep[0]]
        coeff_sort_0 = coeff_keep_0[coeff_keep_0.argsort()]
        index_sort_0 = index_keep_0[coeff_keep_0.argsort()]

        coeff_keep_1 = elast[1,keep[0]]
        index_keep_1 = X_train.columns.values[keep[0]]
        coeff_sort_1 = coeff_keep_1[coeff_keep_1.argsort()]
        index_sort_1 = index_keep_1[coeff_keep_1.argsort()]

        coeff_keep_2 = elast[2,keep[0]]
        index_keep_2 = X_train.columns.values[keep[0]]
        coeff_sort_2 = coeff_keep_2[coeff_keep_2.argsort()]
        index_sort_2 = index_keep_2[coeff_keep_2.argsort()]
        
        ticks = np.arange(0, len(index_sort_0))
        fig, [ax1, ax2, ax3] = plt.subplots(3,1,figsize=(20,15), sharex = False, sharey = True)
        ax1.bar(ticks, coeff_sort_0)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(index_sort_0,  rotation='vertical', fontsize=14)

        ax2.bar(ticks, coeff_sort_1)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(index_sort_1, rotation='vertical')

        ax3.bar(ticks, coeff_sort_2)
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(index_sort_2, rotation='vertical')

        ax1.set_title("Feature importance: Logistic Regression Coefficients (absolute values)")
        plt.subplots_adjust(hspace= 2)
        st.pyplot(fig);
        st.markdown("On peut supprimer les features ayant un coefficient nul dans chaque regression.")
        st.write("Nombre de features gardés: {}".format(len(keep[0])))
    

######################################################################################################################################## 
    st.markdown("## Transformation du dataset")
    st.markdown("Parmis les differentes possibilités: PCA / LDA / Manifolds, la réduction via la méthode PCA a été retenue:")
    st.markdown("* Dataset réduit ")
    st.markdown("* Bonne variance expliquée ")
    from sklearn.decomposition import PCA

    # First, the optimal number of vectors to represent the dataset is searched
    data = df[feats_list]
    pca = PCA(n_components = 6)
    pca.fit(data)
    plt.xlim(1,5)
    x = np.arange(1,6)
    fig = plt.figure(figsize=(6,3))
    #plt.plot(np.array(range(0,6)),pca.explained_variance_ratio_);
    plt.plot(x,pca.explained_variance_ratio_.cumsum()[:len(x)]);
    plt.axhline(y = 0.9, color ='r', linestyle = '--')
    plt.ylabel("Variance expliquée",fontsize=14)
    plt.xlabel("Composantes",fontsize=14)

    st.pyplot(fig);
    
    pca = PCA(n_components = 0.9)
    pca.fit(data)
    st.write("Nombre de composants necessaires pour 90% de variance expliquée:  {}".format(pca.n_components_))
   
   ########################################################################################################################################  
    st.markdown("## Conclusion")
    
    st.markdown("On a donc consideré pour notre étude trois datasets differents afin de montrer la pertinance de la méthode de réduction:")
    st.markdown("* Dataset non réduit, suffixe _NR")
    st.markdown("* Dataset réduit avec feature selection (LogReg), suffixe _FS")
    st.markdown("* Dataset réduit avec PCA, suffixe _R")
    
    
    
    
    
             


