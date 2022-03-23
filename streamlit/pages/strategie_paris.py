import streamlit as st
import os
import pandas as pd
import numpy as np
import warnings
from joblib import dump, load
import matplotlib.pyplot as plt


def app():
    st.header("Stratégie de paris")    
############ Intro  
    st.subheader("Quand parier ?")    
    st.write(r'''
    - Ratio de décision: $ \quad \frac{p}{p_B} = p \cdot o_B > R$
    - Seuil de probabilité: $ p > \alpha $
    ''')

    st.subheader("Combien parier ?")    
    st.write(r'''
    - Montant de pari fixe
    - Critère de Kelly: $ \quad f = p- \left( \frac{1-p}{o-1} \right)  $, $f$ est la fraction du solde disponible. La formule est le résultat de la maximisation de  l'espérance du logarithme du gain
    ''')

    st.write(r'''$f$ a tendance à être sur-éstimé:  $\quad  f < \lambda$ ''')

    st.write(r'''R, $\alpha$, $\lambda$ doivent être optimisés.''')
     

############ Chargement modele entrainé, code, fonctions
    
    grid_xgb_R = load('./data/XG_boost_R_saved.joblib')
    df = pd.read_csv('./data/df_results.csv', index_col = 0)
    df = df.reset_index(drop=True).set_index('match_id')
    X_test_R = pd.read_csv('./data/X_test_R.csv', index_col = 0)
    y_test   = pd.read_csv('./data/y_test.csv', index_col = 0)
    
    
    X_test_R = X_test_R.to_numpy()
    probs = grid_xgb_R.predict_proba(X_test_R)
    pred = grid_xgb_R.predict(X_test_R)

    # A new dataframe df_bet is created by considering only the matchs we will bet on (last season)
    df_bet  = df.loc[y_test.index,:]

    # Predictions and probabilities computed are added to this dataset
    df_bet['Match_Prediction'] = pred

    df_bet['Proba_A'] = probs[:,0]
    df_bet['Proba_D'] = probs[:,1]
    df_bet['Proba_H'] = probs[:,2]

    # The probability of the event predicted is also added 
    # that is the maximum of the probability of each class named Proba_A, Proba_D, Proba_H
    df_bet['Match_Prediction_proba'] = np.max(probs, axis=1)   

    # This function is not used in the rest and gives the Probability of the event according to bookmakers, 
    # taking into account the margin of the bookmakers:
    def bookmaker_proba(pred_result, df):
        bookmaker_list = ['B365', 'LB', 'PS', 'WH', 'VC', 'PSC']
        prob = []
        for i in range(len(df)):
            prob_book   = []
            for j in bookmaker_list:
                # For each bookmaker,one computes their margin to reajust the wining probability
                odd_A = df.iloc[i, df.columns.get_loc(j+'A')]
                odd_D = df.iloc[i, df.columns.get_loc(j+'D')]
                odd_H = df.iloc[i, df.columns.get_loc(j+'H')]
                margin_book = 1/odd_A + 1/odd_D + 1/odd_H        
                #For each bookmaker, one extract the probability of the predicted event (named p). p = (1/odd)*(1/margin)
                index_col = df.columns.get_loc(j+pred_result.iloc[i])
                p = 1/df.iloc[i, index_col]/margin_book
                prob_book.append(p)

            prob.append(np.mean(prob_book))
            # Prob is the mean of normalized bookmaker probabilities. 
            # It is the probability of the predicted event according to the bookmakers.
        return prob   


    # The probability of the event is added to the dataframe, the columns 'Bookmaker_probability' is created
    df_bet['Bookmaker_probability_prediction'] = bookmaker_proba(df_bet['Match_Prediction'], df_bet)    

    # Comparison between the probability considered by the bookmaker of the predicted event to happen,
    # and the probability computed for the predicted event
    df_bet[['Bookmaker_probability_prediction', 'Match_Prediction_proba']].head(5)    
     # Column that indicates if the prediction is good or not (logical)
    df_bet['Winning_bet'] = (df_bet['Match_Prediction'] == df_bet['FTR'])

    # Column that gives the best bookmaker for betting: Maximum of all odds of the predicted event
    def best_odd(pred_result, df):
        bookmaker_list = ['B365', 'LB', 'PS', 'WH', 'VC', 'PSC']
        best_odd = []
        for i in range(len(df)):
            odds   = []
            for j in bookmaker_list:
                #For each bookmaker, one extract the odd of the predicted event
                index_col = df.columns.get_loc(j+pred_result.iloc[i])
                o = df.iloc[i, index_col]
                odds.append(o)
            best_odd.append(np.max(odds))

        return best_odd


    df_bet['Best_odd'] = best_odd(df_bet['Match_Prediction'], df_bet)


    # Betting Strategy 1: Decision criterion and Kelly criterion
    def betting_workflow_R_Kelly(alpha, beta, limit, initial_bet, prob_computed, odds, winning, match_id):
        gain = []
        match = []
        gain.append(initial_bet)

        for i in range(len(prob_computed)):
            if (prob_computed.iloc[i]*odds.iloc[i]>alpha): # Check for the decision ratio    
                Kelly = beta*(prob_computed.iloc[i]*odds.iloc[i]-1)/(odds.iloc[i]-1)
                if Kelly>limit:
                    Kelly=limit
                if Kelly<0.0:
                    Kelly=0.0    
                bet = gain[-1]*Kelly

                if winning.iloc[i]:
                    g = gain[-1] + bet*(odds.iloc[i]-1) # If one wins, bet*(odd-1) is added to the account balance
                else:
                    g = gain[-1] - bet                  # If one loses, the bet is substracted to the account balance
                gain.append(g)    
                match.append(match_id[i])
        return gain , match


    # Betting Strategy 2: Decision criterion and fixed bets
    def betting_workflow_R_Fixed(alpha, fixed, initial_bet, prob_computed, odds, winning, match_id):
        gain = []
        match = []
        gain.append(initial_bet)

        for i in range(len(prob_computed)):
            if (prob_computed.iloc[i]*odds.iloc[i]>alpha): # Check for the decision ratio  
                bet = fixed
                if winning.iloc[i]:
                    g = gain[-1] + bet*(odds.iloc[i]-1) # If one wins, bet*(odd-1) is added to the account balance
                else:
                    g = gain[-1] - bet                  # If one loses, the bet is substracted to the account balance
                gain.append(g)    
                match.append(match_id[i])
        return gain , match



    # Betting Strategy 3: Computed Probability is above threshold and Kelly criterion
    def betting_workflow_Threshold_Kelly(threshold, beta, limit, initial_bet, prob_computed, odds, winning, match_id):  
        gain = []
        match = []
        gain.append(initial_bet)

        for i in range(len(prob_computed)):
            if (prob_computed.iloc[i]>threshold): # Check for the decision ratio
                Kelly = beta*(prob_computed.iloc[i]*odds.iloc[i]-1)/(odds.iloc[i]-1)
                if Kelly>limit:
                    Kelly=limit
                if Kelly<0.0:
                    Kelly=0.0    
                bet = gain[-1]*Kelly
                if winning.iloc[i]:
                    g = gain[-1] + bet*(odds.iloc[i]-1) # If one wins, bet*(odd-1) is added to the account balance
                else:
                    g = gain[-1] - bet              # If one loses, the bet is substracted to the account balance
                gain.append(g)    
                match.append(match_id[i])
        return gain , match 



    # Betting Strategy 4: Computed Probability is above threshold and fixed bet
    def betting_workflow_Threshold_Fixed(threshold, fixed, initial_bet, prob_computed, odds, winning, match_id):  
        gain = []
        match = []
        gain.append(initial_bet)

        for i in range(len(prob_computed)):
            if (prob_computed.iloc[i]>threshold): # Check for the decision ratio
                bet = fixed
                if winning.iloc[i]:
                    g = gain[-1] + bet*(odds.iloc[i]-1) # If one wins, bet*(odd-1) is added to the account balance
                else:
                    g = gain[-1] - bet              # If one loses, the bet is substracted to the account balance
                gain.append(g)    
                match.append(match_id[i])
        return gain , match  
    

############ Interface
    st.subheader("Tests")
    
    initial_bankroll = 100.0
    R         = st.slider('Valeur du ratio R:', 0.0, 1.5, 1.1)
    alpha     = st.slider('Valeur du seuil alpha: ', 0.1, 1.0, 0.2)
    L         = st.slider('Limite lambda pour le critère de Kelly: ', 0.0, 1.0, 0.1)
    fixed_bet = st.slider('Montant fixe à parier: ', 0, 10, 1)
    
    fig1, ax1 = plt.subplots(figsize=(18,10))
    #1 - betting_workflow_R_Kelly(R, beta, limit, initial_bet, prob_computed, odds, winning, match_id)
    gain1, match1 = betting_workflow_R_Kelly(R , 1.0, L, initial_bankroll, df_bet['Match_Prediction_proba'], df_bet['Best_odd'], df_bet['Winning_bet'], df_bet.index)
    #2 - betting_workflow_R_Fixed(R, fixed, initial_bet, prob_computed, odds, winning, match_id)
    gain2, match2 = betting_workflow_R_Fixed(R , fixed_bet, initial_bankroll, df_bet['Match_Prediction_proba'], df_bet['Best_odd'], df_bet['Winning_bet'], df_bet.index)
    
    #3 - betting_workflow_Threshold_Kelly(threshold, beta, limit, initial_bet, prob_computed, odds, winning, match_id):
    gain3, match3 = betting_workflow_Threshold_Kelly(alpha , 1.0, L, initial_bankroll, df_bet['Match_Prediction_proba'], df_bet['Best_odd'], df_bet['Winning_bet'], df_bet.index)
    
    #4 - betting_workflow_Threshold_Fixed(threshold, fixed, initial_bet, prob_computed, odds, winning, match_id): 
    gain4, match4 = betting_workflow_Threshold_Fixed(alpha , fixed_bet, initial_bankroll, df_bet['Match_Prediction_proba'], df_bet['Best_odd'], df_bet['Winning_bet'], df_bet.index)
    
    
    ax1.plot(range(0,len(gain1)), gain1, '--k',label='R= {:.1f}, Kelly'.format(R))
    ax1.annotate("B={0:.0f},G={1:.1f}".format(len(gain1),gain1[-1]), xy=(len(gain1), gain1[-1]), xytext=(20, -20), fontsize=16 , textcoords='offset points', ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.plot(range(0,len(gain2)), gain2, '-k', label="R={:.1f}, Mise fixe = {:.1f}".format(R,fixed_bet)) 
    ax1.annotate("B={0:.0f},G={1:.1f}".format(len(gain2),gain2[-1]), xy=(len(gain2), gain2[-1]), xytext=(20, -20), fontsize=16 , textcoords='offset points', ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.plot(range(0,len(gain3)), gain3, '--r', label="Seuil={:.1f}, Kelly".format(alpha)) 
    ax1.annotate("B={0:.0f},G={1:.1f}".format(len(gain3),gain3[-1]), xy=(len(gain3), gain3[-1]), xytext=(20, -20), fontsize=16 , textcoords='offset points', ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.plot(range(0,len(gain4)), gain4, '-r', label="Seuil={:.1f}, Mise fixe = {:.1f}".format(alpha,fixed_bet)) 
    ax1.annotate("B={0:.0f},G={1:.1f}".format(len(gain4),gain4[-1]), xy=(len(gain4), gain4[-1]), xytext=(20, -20), fontsize=16 , textcoords='offset points', ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.set_xticks(range(0,400,50))
    ax1.axhline(y=0  , xmin=0, xmax=1, color ='red', linestyle='--', linewidth=1.2)
    ax1.axhline(y=100, xmin=0, xmax=1, color ='black', linestyle='-', linewidth=1.2)
    ax1.set_ylabel('Solde', fontsize=16)
    ax1.set_xlabel('Nombre de paris', fontsize=16)
    st.pyplot(fig1);


############ Graph optimum
    st.subheader("Optimum")
    st.write(r''' R = 1.1,  $\lambda = 0.1$, critère de Kelly ''')

    param_var = [1] # variation of beta
    R = 1.1
    limit = 0.1 # One cannot bet above this bankroll proportion
    initial_bankroll = 100

    fig, ax = plt.subplots(figsize=(15,8))

    for i in param_var:
        #betting_workflow_R_Kelly(R, beta, limit, initial_bet, prob_computed, odds, winning, match_id)
        gain, match = betting_workflow_R_Kelly(R , i, limit, initial_bankroll, df_bet['Match_Prediction_proba'], df_bet['Best_odd'], df_bet['Winning_bet'], df_bet.index)
        ax.plot(range(0,len(gain)), gain,'--k') 
        ax.annotate("B={0:.0f},G={1:.1f}".format(len(gain),gain[-1]), xy=(len(gain), gain[-1]), xytext=(20, -20), fontsize=16 , textcoords='offset points', ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax.set_xticks(range(0,200,50))
    plt.xlim((0, 60))
    ax.axhline(y=0  , xmin=0, xmax=1, color ='red', linestyle='--', linewidth=1.2)
    ax.axhline(y=100, xmin=0, xmax=1, color ='black', linestyle='-', linewidth=1.2)
    ax.set_title('R = 1.1, Lambda = 0.1', fontsize=16)
    ax.set_ylabel('Solde', fontsize=16)
    ax.set_xlabel('Nombre de paris', fontsize=16)
    st.pyplot(fig);
       
        
        
        
############ Sure-Bets
    st.subheader("Sure-Bets")
    st.write("On parie sur tous les évenements possibles sous certaines conditions, le gain du pari gagnant doit être supérieur aux pertes des deux paris perdants.")
    st.write(r''' $$ \frac{1}{\max_k o(A)} + \frac{1}{\max_k o(D)} + \frac{1}{\max_k o(H)} < 1 $$ ''')
    
    S_bet         = st.slider('Mise fixe:', 0.0, 50.0, 10.0)
    def sure_bet(fixed, initial_bet, df):  
        gain = []
        match = []
        gain.append(initial_bet)

        bookmaker_list = ['B365', 'LB', 'PS', 'WH', 'VC', 'PSC']

        for i in range(len(df.index)):

            A = [] # odds for event A
            H = [] # odds for event H
            D = [] # odds for event D
            for j in bookmaker_list: # Check for sure bet
                A.append(df.iloc[i, df.columns.get_loc(j+'A')])
                H.append(df.iloc[i, df.columns.get_loc(j+'D')])
                D.append(df.iloc[i, df.columns.get_loc(j+'H')])

            if ((1/np.max(A) + 1/np.max(D) + 1/np.max(H))<1): # Check if one has a "sure bet"
                if (df['FTR'].iloc[i] =='A'):
                    g = gain[-1] + fixed*(np.max(A)-1) 
                elif (df['FTR'].iloc[i] =='H'):
                    g = gain[-1] + fixed*(np.max(H)-1) 
                else:
                    g = gain[-1] + fixed*(np.max(D)-1) 
                gain.append(g)    
                match.append(df.index[i])
        return gain , match 

    param_var = [S_bet]
    initial_bankroll = 100

    fig = plt.figure(figsize = (15, 10))
    ax = fig.add_subplot(111) 

    for i in param_var:
        #sure_bet(fixed, initial_bet, df): 
        gain, match = sure_bet(i, initial_bankroll, df_bet)
        ax.plot(range(0,len(gain)), gain, '--g',label="Mise fixe ={:.2f}".format(i)) 
        ax.annotate("Bets={0:.1f},Sold={1:.1f}".format(len(gain),gain[-1]), xy=(len(gain), gain[-1]), xytext=(20, -20), textcoords='offset points', ha='center', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), fontsize=16)
    ax.set_xticks(range(0,300,50))
    ax.axhline(y=100, xmin=0, xmax=1, color ='black', linestyle='-', linewidth=1.2)
    ax.set_ylabel('Solde', fontsize=16)
    ax.set_xlabel('Nombre de paris', fontsize=16)
    ax.legend(loc='upper left')
    # 202 Sure bets have been found for all 4 seasons. By definition the curve never decreases, 
    # the highest bets give the highest gain
    st.pyplot(fig);
    st.write("Nombre de paris identifiés: {}".format(len(gain)))
    
    
    
    
############ Conclusion
    st.subheader("Conclusion")

    st.markdown("* Bon résultats avec optimum R et Kelly.")
    st.markdown("* A confirmer avec un nombre de matches plus important pour confirmer la tendance.")
    st.markdown("Battre les bookmakers en terme de gain ne necessite pas un meilleur modèle mais une bonne stratégie de paris.")








