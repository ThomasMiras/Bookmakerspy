import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn import model_selection
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score   
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report 

def app():

    st.header("Bookmakerspy - Modélisation")

    st.subheader("Comparaison de modèles")

    # variables style mises en haut car utilisées à différents endroits
    colors = sns.color_palette('pastel')

    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
    
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

        res = [[X_train_NR,X_test_NR],[X_train_FS,X_test_FS],[X_train_R,X_test_R],y_test,y_train,X_train]
        return res
    
    # fonction qui créée les dashboards
    def create_dashboard(metricsNR,metricsFS,metricsR):

        
        col1,col2,col3,col4,col5 = st.columns([1,.2,1,.2,1])

        
        with col1:
            st.write('Dataset non réduit')
            
            f1_score_nr = [metricsNR[3]['A']['f1-score'],metricsNR[3]['D']['f1-score'],metricsNR[3]['H']['f1-score']]
            conf_matrix_nr = confusion_matrix(y_test, metricsNR[2])

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

            d = {'Accuracy Train': [round(metricsNR[4], 2)], 'Accuracy Test': [round(metricsNR[5], 2)]}
            
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))

            st.caption("F1 Score")
            fig, ax = plt.subplots()
            
            ax.bar(x=['Away','Draw','Home'],height=f1_score_nr,color = [colors[0],colors[2],colors[1]])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            st.pyplot(fig)

            
        with col3:
            
            st.write('Dataset sélection de features')

            f1_score_fs = [metricsFS[3]['A']['f1-score'],metricsFS[3]['D']['f1-score'],metricsFS[3]['H']['f1-score']]
            conf_matrix_fs = confusion_matrix(y_test, metricsFS[2])
            

            fig, ax = plt.subplots()
            ax.matshow(conf_matrix_fs, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix_fs.shape[0]):
                for j in range(conf_matrix_fs.shape[1]):
                    ax.text(x=j,y=i,s=conf_matrix_fs[i, j], va='center', ha='center',fontsize=18)
            
            labels = ['Away','Draw','Home']
            ax.set_xticklabels(['']+labels)
            ax.set_yticklabels(['']+labels)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.xaxis.set_label_position('top')
            plt.xlabel('Predicted', fontsize=16)
            plt.ylabel('Real', fontsize=16)
            st.pyplot(fig)

            d = {'Accuracy Train': [round(metricsFS[4], 2)], 'Accuracy Test': [round(metricsFS[5], 2)]}
            
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))

            
            st.caption("F1 Score")
            fig, ax = plt.subplots()
            ax.bar(x=['Away','Draw','Home'],height=f1_score_fs,color = [colors[0],colors[2],colors[1]])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            st.pyplot(fig)
        
        with col5:
        
            st.write('Dataset réduit PCA')

            f1_score_r = [metricsR[3]['A']['f1-score'],metricsR[3]['D']['f1-score'],metricsR[3]['H']['f1-score']]
            conf_matrix_r = confusion_matrix(y_test, metricsR[2])

                
            fig, ax = plt.subplots()
            ax.matshow(conf_matrix_r, cmap=plt.cm.Blues, alpha=0.3)
            
            for i in range(conf_matrix_r.shape[0]):
                for j in range(conf_matrix_r.shape[1]):
                    ax.text(x=j,y=i,s=conf_matrix_r[i, j], va='center', ha='center',fontsize=18)
            
            labels = ['Away','Draw','Home']
            ax.set_xticklabels(['']+labels)
            ax.set_yticklabels(['']+labels)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax.xaxis.set_label_position('top')
            plt.xlabel('Predicted', fontsize=16)
            plt.ylabel('Real', fontsize=16)
            st.pyplot(fig)

            d = {'Accuracy Train': [round(metricsR[4], 2)], 'Accuracy Test': [round(metricsR[5], 2)]}
            
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))   

            st.caption("F1 Score")
            fig, ax = plt.subplots()
            ax.bar(x=['Away','Draw','Home'],height=f1_score_r,color = [colors[0],colors[2],colors[1]])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
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

        X_train = get_datasets()[5]
    
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

    # création des modèles de K plus proche voisins
    @st.cache
    def get_data_knn():
        parametres = {'n_neighbors': [10, 20, 30, 40, 50], 'metric': ['manhattan', 'chebyshev', 'minkowski', 'l1' , 'l2'] }

        knn_NR = neighbors.KNeighborsClassifier()
        knn_FS = neighbors.KNeighborsClassifier()
        knn_R  = neighbors.KNeighborsClassifier()

        grid_knn_NR = GridSearchCV(estimator=knn_NR, param_grid=parametres)
        grid_knn_FS = GridSearchCV(estimator=knn_FS, param_grid=parametres)
        grid_knn_R = GridSearchCV(estimator=knn_R, param_grid=parametres)

        grid_knn_NR.fit(X_train_NR,y_train)
        grid_knn_FS.fit(X_train_FS,y_train)
        grid_knn_R.fit(X_train_R,y_train)

        y_pred_knn_NR = grid_knn_NR.predict(X_test_NR)
        y_pred_knn_FS = grid_knn_FS.predict(X_test_FS)
        y_pred_knn_R  = grid_knn_R.predict(X_test_R)
        
        report_NR = classification_report(y_test, pd.DataFrame(y_pred_knn_NR),output_dict=True)
        accuracy_train_NR = grid_knn_NR.score(X_train_NR, y_train)
        accuracy_test_NR = grid_knn_NR.score(X_test_NR, y_test)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_knn_FS),output_dict=True)
        accuracy_train_FS = grid_knn_FS.score(X_train_FS, y_train)
        accuracy_test_FS = grid_knn_FS.score(X_test_FS, y_test)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_knn_R),output_dict=True)
        accuracy_train_R = grid_knn_R.score(X_train_R, y_train)
        accuracy_test_R = grid_knn_R.score(X_test_R, y_test)
        
        return [[knn_NR,grid_knn_NR,y_pred_knn_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[knn_FS,grid_knn_FS,y_pred_knn_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[knn_R,grid_knn_R,y_pred_knn_R,report_R,accuracy_train_R,accuracy_test_R]]

    # création des modèles de svm
    @st.cache
    def get_data_svm():
        parametres = {'C':[0.1,1,3], 'kernel':['rbf','linear'], 'gamma':[0.005, 0.1, 0.5]}# Cross validation

        clf_svm_NR = svm.SVC(probability=True)
        clf_svm_FS = svm.SVC(probability=True)
        clf_svm_R = svm.SVC(probability=True)

        grid_clf_svm_NR = model_selection.GridSearchCV(estimator=clf_svm_NR, param_grid=parametres)
        grid_clf_svm_FS = model_selection.GridSearchCV(estimator=clf_svm_FS, param_grid=parametres)
        grid_clf_svm_R  = model_selection.GridSearchCV(estimator=clf_svm_R, param_grid=parametres)

        grid_clf_svm_NR.fit(X_train_NR,y_train)
        grid_clf_svm_FS.fit(X_train_FS,y_train)
        grid_clf_svm_R.fit(X_train_R,y_train)


        y_pred_clf_svm_NR = grid_clf_svm_NR.predict(X_test_NR)
        y_pred_clf_svm_FS = grid_clf_svm_FS.predict(X_test_FS)
        y_pred_clf_svm_R  = grid_clf_svm_R.predict(X_test_R)

        report_NR = classification_report(y_test, pd.DataFrame(y_pred_clf_svm_NR),output_dict=True)
        accuracy_train_NR = grid_clf_svm_NR.score(X_train_NR, y_train)
        accuracy_test_NR = grid_clf_svm_NR.score(X_test_NR, y_test)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_clf_svm_FS),output_dict=True)
        accuracy_train_FS = grid_clf_svm_FS.score(X_train_FS, y_train)
        accuracy_test_FS = grid_clf_svm_FS.score(X_test_FS, y_test)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_clf_svm_R),output_dict=True)
        accuracy_train_R = grid_clf_svm_R.score(X_train_R, y_train)
        accuracy_test_R = grid_clf_svm_R.score(X_test_R, y_test)
        
        return [[clf_svm_NR,grid_clf_svm_NR,y_pred_clf_svm_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[clf_svm_FS,grid_clf_svm_FS,y_pred_clf_svm_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[clf_svm_R,grid_clf_svm_R,y_pred_clf_svm_R,report_R,accuracy_train_R,accuracy_test_R]]

    # création des modèles de Decision Tree
    @st.cache(allow_output_mutation=True)
    def get_data_dectree():
        parametres = {'max_depth': [1, 2, 3, 5, 7]}

        dtc_NR = DecisionTreeClassifier()
        dtc_FS = DecisionTreeClassifier()
        dtc_R  = DecisionTreeClassifier()

        grid_dtc_NR = GridSearchCV(estimator=dtc_NR, param_grid=parametres)
        grid_dtc_FS = GridSearchCV(estimator=dtc_FS, param_grid=parametres)
        grid_dtc_R  = GridSearchCV(estimator=dtc_R, param_grid=parametres)

        grid_dtc_NR.fit(X_train_NR,y_train)
        grid_dtc_FS.fit(X_train_FS,y_train)
        grid_dtc_R.fit(X_train_R,y_train)

        y_pred_dtc_NR = grid_dtc_NR.predict(X_test_NR)
        y_pred_dtc_FS = grid_dtc_FS.predict(X_test_FS)
        y_pred_dtc_R  = grid_dtc_R.predict(X_test_R)

        report_NR = classification_report(y_test, pd.DataFrame(y_pred_dtc_NR),output_dict=True)
        accuracy_train_NR = grid_dtc_NR.score(X_train_NR, y_train)
        accuracy_test_NR = grid_dtc_NR.score(X_test_NR, y_test)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_dtc_FS),output_dict=True)
        accuracy_train_FS = grid_dtc_FS.score(X_train_FS, y_train)
        accuracy_test_FS = grid_dtc_FS.score(X_test_FS, y_test)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_dtc_R),output_dict=True)
        accuracy_train_R = grid_dtc_R.score(X_train_R, y_train)
        accuracy_test_R = grid_dtc_R.score(X_test_R, y_test)
        
        return [[dtc_NR,grid_dtc_NR,y_pred_dtc_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[dtc_FS,grid_dtc_FS,y_pred_dtc_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[dtc_R,grid_dtc_R,y_pred_dtc_R,report_R,accuracy_train_R,accuracy_test_R]]

    # création des modèles de Boosting
    @st.cache(allow_output_mutation=True)
    def get_data_boosting():
        dtc_NR = DecisionTreeClassifier()
        dtc_FS = DecisionTreeClassifier()
        dtc_R  = DecisionTreeClassifier()

        ac_NR = AdaBoostClassifier(base_estimator=dtc_NR, n_estimators=400)
        ac_FS = AdaBoostClassifier(base_estimator=dtc_FS, n_estimators=400)
        ac_R  = AdaBoostClassifier(base_estimator=dtc_R,  n_estimators=400)

        ac_NR.fit(X_train_NR,y_train)
        ac_FS.fit(X_train_FS,y_train)
        ac_R.fit(X_train_R,y_train)

        y_pred_ac_NR = ac_NR.predict(X_test_NR)
        y_pred_ac_FS = ac_FS.predict(X_test_FS)
        y_pred_ac_R  = ac_R.predict(X_test_R)

        report_NR = classification_report(y_test, pd.DataFrame(y_pred_ac_NR),output_dict=True)
        accuracy_train_NR = ac_NR.score(X_train_NR, y_train)
        accuracy_test_NR = ac_NR.score(X_test_NR, y_test)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_ac_FS),output_dict=True)
        accuracy_train_FS = ac_FS.score(X_train_FS, y_train)
        accuracy_test_FS = ac_FS.score(X_test_FS, y_test)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_ac_R),output_dict=True)
        accuracy_train_R = ac_R.score(X_train_R, y_train)
        accuracy_test_R = ac_R.score(X_test_R, y_test)
        
        return [[ac_NR,ac_NR,y_pred_ac_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[ac_FS,ac_FS,y_pred_ac_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[ac_R,ac_R,y_pred_ac_R,report_R,accuracy_train_R,accuracy_test_R]]

    # création des modèles de random forest
    @st.cache(allow_output_mutation=True)
    def get_data_rf():
        parametres = {'max_depth': [1, 2, 3, 5, 7, 10],'n_estimators': [10, 30, 50, 100] }

        forest_NR = RandomForestClassifier(random_state=0)
        forest_FS = RandomForestClassifier(random_state=0)
        forest_R = RandomForestClassifier(random_state=0)

        grid_forest_NR = GridSearchCV(estimator=forest_NR, param_grid=parametres)
        grid_forest_FS = GridSearchCV(estimator=forest_FS, param_grid=parametres)
        grid_forest_R = GridSearchCV(estimator=forest_R, param_grid=parametres)

        grid_forest_NR.fit(X_train_NR,y_train)
        grid_forest_FS.fit(X_train_FS,y_train)
        grid_forest_R.fit(X_train_R,y_train)

        y_pred_forest_NR = grid_forest_NR.predict(X_test_NR)
        y_pred_forest_FS = grid_forest_FS.predict(X_test_FS)
        y_pred_forest_R  = grid_forest_R.predict(X_test_R)

        report_NR = classification_report(y_test, pd.DataFrame(y_pred_forest_NR),output_dict=True)
        accuracy_train_NR = grid_forest_NR.score(X_train_NR, y_train)
        accuracy_test_NR = grid_forest_NR.score(X_test_NR, y_test)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_forest_FS),output_dict=True)
        accuracy_train_FS = grid_forest_FS.score(X_train_FS, y_train)
        accuracy_test_FS = grid_forest_FS.score(X_test_FS, y_test)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_forest_R),output_dict=True)
        accuracy_train_R = grid_forest_R.score(X_train_R, y_train)
        accuracy_test_R = grid_forest_R.score(X_test_R, y_test)
        
        return [[forest_NR,grid_forest_NR,y_pred_forest_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[forest_FS,grid_forest_FS,y_pred_forest_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[forest_R,grid_forest_R,y_pred_forest_R,report_R,accuracy_train_R,accuracy_test_R]]

    # création des modèles de xg boost
    @st.cache(allow_output_mutation=True)
    def get_data_xgboost():
        
        y_train_xgb = y_train.replace({'H': 1, 'D': 0, 'A': -1})
        y_test_xgb  =  y_test.replace({'H': 1, 'D': 0, 'A': -1})

        train_xgb = xgb.DMatrix(data=X_train, label=y_train_xgb)

        train_xgb_NR = xgb.DMatrix(data=X_train_NR, label=y_train_xgb)
        test_xgb_NR  = xgb.DMatrix(data=X_test_NR, label=y_test_xgb)

        train_xgb_FS = xgb.DMatrix(data=X_train_FS, label=y_train_xgb)
        test_xgb_FS  = xgb.DMatrix(data=X_test_FS, label=y_test_xgb)

        train_xgb_R = xgb.DMatrix(data=X_train_R, label=y_train_xgb)
        test_xgb_R  = xgb.DMatrix(data=X_test_R, label=y_test_xgb)
        
        param_CV = {'max_depth': range(2, 3, 5), 'num_boost_round': [10, 30, 50, 100], 'learning_rate': [0.005, 0.01, 0.05]}

        xgb_ini_NR = xgb.XGBClassifier(objective='multi:softprob')
        xgb_ini_FS = xgb.XGBClassifier(objective='multi:softprob')
        xgb_ini_R = xgb.XGBClassifier(objective='multi:softprob')

        grid_xgb_NR = GridSearchCV(estimator=xgb_ini_NR, param_grid=param_CV, scoring = 'f1', cv = 4)
        grid_xgb_FS = GridSearchCV(estimator=xgb_ini_FS, param_grid=param_CV, scoring = 'f1', cv = 4)
        grid_xgb_R = GridSearchCV(estimator=xgb_ini_R, param_grid=param_CV, scoring = 'f1', cv = 4)

        grid_xgb_NR.fit(X_train_NR,y_train)
        grid_xgb_FS.fit(X_train_FS,y_train)
        grid_xgb_R.fit(X_train_R,y_train)

        y_pred_u_xgb_NR = grid_xgb_NR.predict(X_test_NR)
        y_pred_u_xgb_FS = grid_xgb_FS.predict(X_test_FS)
        y_pred_u_xgb_R  = grid_xgb_R.predict(X_test_R)

        y_pred_train_u_xgb_NR = grid_xgb_NR.predict(X_train_NR)
        y_pred_train_u_xgb_FS = grid_xgb_FS.predict(X_train_FS)
        y_pred_train_u_xgb_R  = grid_xgb_R.predict(X_train_R)
        
        report_NR = classification_report(y_test, pd.DataFrame(y_pred_u_xgb_NR),output_dict=True)
        accuracy_train_NR = accuracy_score(y_train, y_pred_train_u_xgb_NR)
        accuracy_test_NR = accuracy_score(y_test, y_pred_u_xgb_NR)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_u_xgb_FS),output_dict=True)
        accuracy_train_FS = accuracy_score(y_train, y_pred_train_u_xgb_FS)
        accuracy_test_FS = accuracy_score(y_test, y_pred_u_xgb_FS)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_u_xgb_R),output_dict=True)
        accuracy_train_R = accuracy_score(y_train, y_pred_train_u_xgb_R)
        accuracy_test_R = accuracy_score(y_test, y_pred_u_xgb_R)
        
        return [[xgb_ini_NR,grid_xgb_NR,y_pred_u_xgb_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[xgb_ini_FS,grid_xgb_FS,y_pred_u_xgb_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[xgb_ini_R,grid_xgb_R,y_pred_u_xgb_R,report_R,accuracy_train_R,accuracy_test_R]]

    # création des modèles de voting classifier
    @st.cache(allow_output_mutation=True)
    def get_data_vc():

        parametres = {'max_depth': [1, 2, 3, 5, 7, 10],'n_estimators': [10, 30, 50, 100] }

        vclf_NR = VotingClassifier(estimators=[('cfl', get_data_logreg()[0][1]), ('knn', get_data_knn()[0][1]), ('svm', get_data_svm()[0][1]), 
                                       ('dtc_boost', get_data_boosting()[0][1]), ('Rforest', get_data_rf()[0][1]), ('XGB', get_data_xgboost()[0][1])], voting='soft')
        vclf_FS = VotingClassifier(estimators=[('cfl', get_data_logreg()[1][1]), ('knn', get_data_knn()[1][1]), ('svm', get_data_svm()[1][1]), 
                                       ('dtc_boost', get_data_boosting()[1][1]), ('Rforest', get_data_rf()[1][1]), ('XGB', get_data_xgboost()[1][1])], voting='soft')
        vclf_R  = VotingClassifier(estimators=[ ('cfl', get_data_logreg()[2][1]), ('knn', get_data_knn()[2][1]) , ('svm', get_data_svm()[2][1]),
                                        ('dtc_boost',get_data_boosting()[2][1]), ('Rforest', get_data_rf()[1][1]), ('XGB', get_data_xgboost()[2][1])], voting='soft')

        # Performances:
        vclf_NR.fit(X_train_NR, y_train)
        vclf_FS.fit(X_train_FS, y_train)
        vclf_R.fit(X_train_R, y_train)

        y_pred_vcfl_NR = vclf_NR.predict(X_test_NR)
        y_pred_vcfl_FS = vclf_FS.predict(X_test_FS)
        y_pred_vcfl_R  = vclf_R.predict(X_test_R)

        report_NR = classification_report(y_test, pd.DataFrame(y_pred_vcfl_NR),output_dict=True)
        accuracy_train_NR = vclf_NR.score(X_train_NR, y_train)
        accuracy_test_NR = vclf_NR.score(X_test_NR, y_test)

        report_FS = classification_report(y_test, pd.DataFrame(y_pred_vcfl_FS),output_dict=True)
        accuracy_train_FS = vclf_FS.score(X_train_FS, y_train)
        accuracy_test_FS = vclf_FS.score(X_test_FS, y_test)

        report_R = classification_report(y_test, pd.DataFrame(y_pred_vcfl_R),output_dict=True)
        accuracy_train_R = vclf_R.score(X_train_R, y_train)
        accuracy_test_R = vclf_R.score(X_test_R, y_test)
        
        return [[vclf_NR,vclf_NR,y_pred_vcfl_NR,report_NR,accuracy_train_NR,accuracy_test_NR],[vclf_FS,vclf_FS,y_pred_vcfl_FS,report_FS,accuracy_train_FS,accuracy_test_FS],[vclf_R,vclf_R,y_pred_vcfl_R,report_R,accuracy_train_R,accuracy_test_R]]


    option = st.selectbox(
     'Choisir un modèle',
     ('Régression logistique', 'K plus proches voisins', 'SVM', 'Decision Tree', 'Boosting','Random Forest','XG Boost','Voting Classifier'))

    if option == 'Régression logistique':

        with st.spinner('Calculs en cours'):
           
            logregNR = get_data_logreg()[0]
            logregFS = get_data_logreg()[1]
            logregR = get_data_logreg()[2]

            create_dashboard(logregNR,logregFS,logregR)
    
    if option == 'K plus proches voisins':

        with st.spinner('Calculs en cours'):
           
            knnNR = get_data_knn()[0]
            knnFS = get_data_knn()[1]
            knnR = get_data_knn()[2]

            create_dashboard(knnNR,knnFS,knnR)
    
    if option == 'SVM':

        with st.spinner('Calculs en cours'):
            
            svmNR = get_data_svm()[0]
            svmFS = get_data_svm()[1]
            svmR = get_data_svm()[2]

        create_dashboard(svmNR,svmFS,svmR)
    
    if option == 'Decision Tree':

        with st.spinner('Calculs en cours'):
            
            dectNR = get_data_dectree()[0]
            dectFS = get_data_dectree()[1]
            dectR = get_data_dectree()[2]

        create_dashboard(dectNR,dectFS,dectR)
    
    if option == 'Boosting':

        with st.spinner('Calculs en cours'):
            
            boostingNR = get_data_boosting()[0]
            boostingFS = get_data_boosting()[1]
            boostingR = get_data_boosting()[2]

        create_dashboard(boostingNR,boostingFS,boostingR)

    if option == 'Random Forest':

        with st.spinner('Calculs en cours'):
            
            forestNR = get_data_rf()[0]
            forestFS = get_data_rf()[1]
            forestR = get_data_rf()[2]

        create_dashboard(forestNR,forestFS,forestR)
    
    if option == 'XG Boost':

        with st.spinner('Calculs en cours'):
            
            xgboostNR = get_data_xgboost()[0]
            xgboostFS = get_data_xgboost()[1]
            xgboostR = get_data_xgboost()[2]

        create_dashboard(xgboostNR,xgboostFS,xgboostR)
    
    if option == 'Voting Classifier':

        with st.spinner('Calculs en cours'):
            
            vcNR = get_data_vc()[0]
            vcFS = get_data_vc()[1]
            vcR = get_data_vc()[2]

        create_dashboard(vcNR,vcFS,vcR)

    st.subheader('Résumé des performances')

    models = ['Régression logistique', 'K plus proches voisins', 'SVM', 'Decision Tree', 'Boosting','Random Forest','XG Boost','Voting Classifier']
    models_abbr = ['RL', 'KN', 'SVM', 'DT', 'BO','RF','XGB','VC']
    
    @st.cache
    def get_metrics():
        report_nr = [get_data_logreg()[0][3],get_data_knn()[0][3],get_data_svm()[0][3],get_data_dectree()[0][3],get_data_boosting()[0][3],get_data_rf()[0][3],get_data_xgboost()[0][3],get_data_vc()[0][3]]
        report_fs = [get_data_logreg()[1][3],get_data_knn()[1][3],get_data_svm()[1][3],get_data_dectree()[1][3],get_data_boosting()[1][3],get_data_rf()[1][3],get_data_xgboost()[1][3],get_data_vc()[1][3]]
        report_r = [get_data_logreg()[2][3],get_data_knn()[2][3],get_data_svm()[2][3],get_data_dectree()[2][3],get_data_boosting()[2][3],get_data_rf()[2][3],get_data_xgboost()[2][3],get_data_vc()[2][3]]

        return [report_nr,report_fs,report_r]

    with st.spinner('Calculs en cours'):
    
        st.write("Accuracy")

        accuracy_nr = [d['accuracy'] for d in get_metrics()[0]]
        accuracy_fs = [d['accuracy'] for d in get_metrics()[1]]
        accuracy_r = [d['accuracy'] for d in get_metrics()[2]]

        col1,col2,col3,col4,col5 = st.columns([1,.2,1,.2,1])
        
        with col1:

            st.caption("Dataset non réduit")
            fig, ax = plt.subplots()
            ax.bar(x=models,height=accuracy_nr, color = colors)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            ax.tick_params(axis='x', rotation=90)
            st.pyplot(fig)

            d = {'Modèles':models_abbr,'Accuracy NR': accuracy_nr}
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))   

        
        with col3:
            st.caption("Dataset sélection de features")
            fig, ax = plt.subplots()
            ax.bar(x=models,height=accuracy_fs, color = sns.color_palette('pastel'))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            ax.tick_params(axis='x', rotation=90)
            st.pyplot(fig)

            d = {'Modèles':models_abbr,'Accuracy FS': accuracy_fs}
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))   

        with col5:
            st.caption("Dataset réduit PCA")
            fig, ax = plt.subplots()
            ax.bar(x=models,height=accuracy_r, color = sns.color_palette('pastel'))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            ax.tick_params(axis='x', rotation=90)
            st.pyplot(fig)

            d = {'Modèles':models_abbr,'Accuracy R': accuracy_r}
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(pd.DataFrame(d).style.format("{:.2}"))   
            

        st.write("F1 Score")

        f1_nr = [[d['A']['f1-score'],d['D']['f1-score'],d['H']['f1-score']] for d in get_metrics()[0]]
        f1_fs = [[d['A']['f1-score'],d['D']['f1-score'],d['H']['f1-score']]  for d in get_metrics()[1]]
        f1_r = [[d['A']['f1-score'],d['D']['f1-score'],d['H']['f1-score']]  for d in get_metrics()[2]]

        f1_nr_H = [d['H']['f1-score'] for d in get_metrics()[0]]
        f1_nr_A = [d['A']['f1-score'] for d in get_metrics()[0]]
        f1_nr_D = [d['D']['f1-score'] for d in get_metrics()[0]]      

        f1_fs_H = [d['H']['f1-score'] for d in get_metrics()[1]]
        f1_fs_A = [d['A']['f1-score'] for d in get_metrics()[1]]
        f1_fs_D = [d['D']['f1-score'] for d in get_metrics()[1]]

        f1_r_H = [d['H']['f1-score'] for d in get_metrics()[2]]
        f1_r_A = [d['A']['f1-score'] for d in get_metrics()[2]]
        f1_r_D = [d['D']['f1-score'] for d in get_metrics()[2]]            
        
        col1,col2,col3,col4,col5 = st.columns([1,.2,1,.2,1])

        with col1:
            st.caption("Dataset non réduit")

            x = np.arange(len(models_abbr))  # the label locations
            width = 0.3  # the width of the bars

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - .3, f1_nr_A, width, label='Away',color='#a1c9f4')
            rects2 = ax.bar(x - .1, f1_nr_D, width, label='Draw',color='#8de5a1')
            rects3 = ax.bar(x + .1, f1_nr_H, width, label='Home',color='#ffb482')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('F1 Score')
            ax.set_xticks(x, models_abbr)
            ax.legend()

            #ax.bar_label(rects1, padding=3)
            #ax.bar_label(rects2, padding=3)

            fig.tight_layout()

            st.pyplot(fig)



            st.dataframe(pd.DataFrame(f1_nr,columns=['Away','Draw','Home'],index=models_abbr).style.format("{:.2}").highlight_max(axis=0))

    with col3:
            
        st.caption("Dataset sélection de features")

        x = np.arange(len(models_abbr))  # the label locations
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - .3, f1_fs_A, width, label='Away',color='#a1c9f4')
        rects2 = ax.bar(x - .1, f1_fs_D, width, label='Draw',color='#8de5a1')
        rects3 = ax.bar(x + .1, f1_fs_H, width, label='Home',color='#ffb482')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('F1 Score')
        ax.set_xticks(x, models_abbr)
        ax.legend()

        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        st.pyplot(fig)
        st.dataframe(pd.DataFrame(f1_fs,columns=['Away','Draw','Home'],index=models_abbr).style.format("{:.2}").highlight_max(axis=0))


    with col5:
            
        st.caption("Dataset réduit PCA")

        x = np.arange(len(models_abbr))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - .3, f1_r_A, width, label='Away',color='#a1c9f4')
        rects2 = ax.bar(x - .1, f1_r_D, width, label='Draw',color='#8de5a1')
        rects3 = ax.bar(x + .1, f1_r_H, width, label='Home',color='#ffb482')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('F1 Score')
        ax.set_xticks(x, models_abbr)
        ax.legend()

        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        st.pyplot(fig)
        st.dataframe(pd.DataFrame(f1_r,columns=['Away','Draw','Home'],index=models_abbr).style.format("{:.2}").highlight_max(axis=0))
