import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score   
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report 

def app():

    st.header("Bookmakerspy - Modélisation")

    st.subheader("Comparaison de modèles")
    
    # fonction qui créée les train et test
    @st.cache
    def get_datasets():

        df = pd.read_csv('./data/df_results.csv', index_col = 0)
        df = df.reset_index(drop=True).set_index('match_id')
        
        feats_list = [ # Features for the 'home' team
        'home_team_rating', 'home_won_contest', 'home_possession_percentage', 'home_total_throws', 'home_blocked_scoring_att', 
        'home_total_scoring_att', 'home_total_tackle', 'home_aerial_won', 'home_aerial_lost', 'home_accurate_pass', 
        'home_total_pass', 'home_won_corners', 'home_shot_off_target', 'home_ontarget_scoring_att','home_total_offside', 
        'home_post_scoring_att', 'home_att_pen_goal', 'home_penalty_save', 'HF', 'HY', 'HR', 'home_pass', 
        'goalkeeper_home_player_rating', 'defender_home_player_rating', 'midfielder_home_player_rating', 'forward_home_player_rating', 'FTHG_mean',       
                # Features for the 'away' team
        'away_team_rating', 'away_won_contest', 'away_possession_percentage', 'away_total_throws', 'away_blocked_scoring_att',
        'away_total_scoring_att', 'away_total_tackle', 'away_aerial_won', 'away_aerial_lost', 'away_accurate_pass', 
        'away_total_pass', 'away_won_corners', 'away_shot_off_target', 'away_ontarget_scoring_att', 'away_total_offside', 
        'away_post_scoring_att', 'away_att_pen_goal', 'away_penalty_save', 'AF', 'AY', 'AR', 'away_pass',
        'goalkeeper_away_player_rating', 'defender_away_player_rating', 'midfielder_away_player_rating', 'forward_away_player_rating', 'FTAG_mean',
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

        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)

        # Dataset _NR
        X_train_NR = X_train
        X_test_NR = X_test

        # Dataset _FS
        to_drop = ['FTAG_mean', 'defender_home_player_rating', 'Diff_mil_home_mid_away', 'Diff_def_home_mid_away', 'home_possession_percentage', 'HY', 'Diff_def_home_fwd_away', 'away_post_scoring_att', 'away_pass', 'home_aerial_won', 'away_ontarget_scoring_att', 'forward_home_player_rating', 'away_total_offside', 'defender_away_player_rating', 'home_total_scoring_att', 'Diff_mil_home_att_away', 'away_team_rating', 'home_total_throws', 'Diff_mil_home_def_away', 'home_team_rating', 'home_aerial_lost', 'Diff_Goal', 'FTHG_mean', 'home_blocked_scoring_att', 'home_pass', 'away_accurate_pass', 'midfielder_home_player_rating']
        X_train_FS = X_train.drop(to_drop, axis=1)
        X_test_FS  = X_test.drop(to_drop, axis=1)

        # Dataset _R
        pca = PCA(n_components = 0.9)
        X_train_R = pca.fit_transform(X_train)
        X_test_R = pca.transform(X_test)

        res = [[X_train_NR,X_test_NR],[X_train_FS,X_test_FS],[X_train_R,X_test_R],y_test,y_train]
        return res
    
    # fonction qui créée les dashboards
    def create_dashboard(metricsNR,metricsFS,metricsR):

        st.caption('Dataset non réduit')

        col1,col2,col3 = st.columns([1,.5,2])

        f1_score_nr = [metricsNR[3]['A']['f1-score'],metricsNR[3]['D']['f1-score'],metricsNR[3]['H']['f1-score']]
        conf_matrix_nr = confusion_matrix(y_test, metricsNR[2])
        
        with col1:
            st.write("F1 Score")
            fig, ax = plt.subplots()
            ax.bar(x=['Away','Draw','Home'],height=f1_score_nr)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            st.pyplot(fig)

            st.metric(label="Accuracy Train", value=round(metricsNR[4], 2))
            st.metric(label="Accuracy Test", value=round(metricsNR[5], 2))
            

        with col3:
            st.write("Matrice de confusion")
            fig, ax = plt.subplots()
            ax.matshow(conf_matrix_nr, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix_nr.shape[0]):
                for j in range(conf_matrix_nr.shape[1]):
                    ax.text(x=j,y=i,s=conf_matrix_nr[i, j], va='center', ha='center',fontsize=18)
            
            labels = ['Away','Draw','Home']
            ax.set_xticklabels(['']+labels)
            ax.set_yticklabels(['']+labels)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.xaxis.set_label_position('top')
            plt.xlabel('Predicted', fontsize=16)
            plt.ylabel('Real', fontsize=16)
            st.pyplot(fig)

    
    # création des datasets train et test
    with st.spinner('Mise en place des données train et test...'):
        X_train_NR = get_datasets()[0][0]
        X_test_NR = get_datasets()[0][1]

        X_train_FS = get_datasets()[1][0]
        X_test_FS = get_datasets()[1][1]

        X_train_R = get_datasets()[2][0]
        X_test_R = get_datasets()[2][1]

        y_test = get_datasets()[3]
        y_train = get_datasets()[4]
    
    # création des modèles de régression logistique
    @st.cache
    def get_data_logreg():
        parametres = {'C':[0.05,0.1,1,3],'l1_ratio': [0.01, 0.1, 0.2, 0.5, 0.99]}

        clf_NR = linear_model.LogisticRegression(penalty = 'elasticnet', solver = 'saga',max_iter = 2000)
        clf_FS = linear_model.LogisticRegression(penalty = 'elasticnet', solver = 'saga',max_iter = 2000)
        clf_R  = linear_model.LogisticRegression(penalty = 'elasticnet', solver = 'saga',max_iter = 2000)

        grid_clf_NR = GridSearchCV(estimator=clf_NR, param_grid=parametres)
        grid_clf_FS = GridSearchCV(estimator=clf_FS, param_grid=parametres)
        grid_clf_R = GridSearchCV(estimator=clf_R, param_grid=parametres)

        grid_clf_NR.fit(X_train_NR,y_train)
        grid_clf_FS.fit(X_train_FS,y_train)
        grid_clf_R.fit(X_train_R,y_train)

        y_pred_cfl_NR = grid_clf_NR.predict(X_test_NR)
        y_pred_cfl_FS = grid_clf_FS.predict(X_test_FS)
        y_pred_cfl_R  = grid_clf_R.predict(X_test_R)

        report_NR = classification_report(y_test, pd.DataFrame(y_pred_cfl_NR),output_dict=True)
        accuracy_train_NR = grid_clf_NR.score(X_train_NR, y_train)
        accuracy_test_NR = grid_clf_NR.score(X_test_NR, y_test)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_cfl_FS),output_dict=True)
        accuracy_train_FS = grid_clf_FS.score(X_train_FS, y_train)
        accuracy_test_FS = grid_clf_FS.score(X_test_FS, y_test)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_cfl_R),output_dict=True)
        accuracy_train_R = grid_clf_R.score(X_train_R, y_train)
        accuracy_test_R = grid_clf_R.score(X_test_R, y_test)
        
        return [[clf_NR,grid_clf_NR,y_pred_cfl_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[clf_FS,grid_clf_FS,y_pred_cfl_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[clf_R,grid_clf_R,y_pred_cfl_R,report_R,accuracy_train_R,accuracy_test_R]]

    
    option = st.selectbox(
     'Choisir un modèle',
     ('Régression logistique', 'Home phone', 'Mobile phone'))

    if option == 'Régression logistique':

        with st.spinner('Calculs en cours'):
           
            logregNR = get_data_logreg()[0]
            logregFS = get_data_logreg()[1]
            logregR = get_data_logreg()[2]

            create_dashboard(logregNR,logregFS,logregR)

        




    #print(classification_report(y_test, pd.DataFrame(logregNR[2])))

    
    #lr = LogisticRegression()
    #gnb = GaussianNB()
    #svc = NaivelyCalibratedLinearSVC(C=1.0)
    #rfc = RandomForestClassifier()

    #clf_list = [
    #    (lr, "Logistic"),
    #    (gnb, "Naive Bayes"),
    #    (svc, "SVC"),
    #    (rfc, "Random forest"),
    #]

