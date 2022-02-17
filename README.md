## Project context
This project was completed throughout the June 2021 data science continuous training session with the training organization [Datascientest.com](https://datascientest.com/).</br>
## Goal
Focusing on English Premier League football matches datasets (2014/15-2017/18), this project aims at coming up with a process that could assist sports betting app users getting to a better-informed guess about a betting outcome. In order to progress on this goal, we attempted to address two challenges: predicting the matches outcome and comparing with bookmakers predictions, so as to come up with a betting strategy.
## Models and packages
We experimented and compared the results for several classification machine learning models:</br>
* Logistic regression
* K nearest neighbours
* SVM
* Decision tree and Boosting
* Random Forest
* XG Boost
* Voting classifier
  
We used the packages [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.ai/)
## Datasets
Data used in this project were collected via [Kaggle](https://www.kaggle.com/shubhmamp/english-premier-league-match-data) and [Datahub.io](https://datahub.io/sports-data/english-premier-league#data-cli).</br>
Data collected on Kaggle were used to retrieve information concerning English Premier League football matches for four seasons, from 2014/2015 to 2017/2018. This dataset contains match statistics and players statistics for the matches. We also collected bookmakers’ odds data for the same matches from the dataset linked above and available on datahub.io.
## Repo content description
The current repo contains three notebooks:
* 1_bookmakerspy_data_collection.ipynb</br>
  The first notebook aims at collecting data from the sources and making the necessary adjustments to obtain a dataset that contains match and players stats.</br> 
* 2_bookmakerspy_preprocessing_dataviz.ipynb</br>
  The second notebooks aims at pre-processing the dataset by removing, grouping or adding variables for example. We also replaced each features row of the dataset obtained in the first notebook by the mean of the 3 past matches. The methodology is explained in more details throughout the notebook. An exploration process via visualization is also detailed.
* 3_bookmakerspy_modelisation.ipynb</br>
The third notebook contains the dimension reduction step, as well as the experimentations on various models. It also thoroughly details an approach tested in terms of betting strategy.
## Members of the team
Mariella DE CROUY CHANEL</br>
Thomas MIRAS</br>
Landry TAYAYA</br>
## Our mentors from Datascientest.com
Thibault</br>
Emilie</br>
Antoine



