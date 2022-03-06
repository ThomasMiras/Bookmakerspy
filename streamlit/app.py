# -*- coding: utf-8 -*-

import streamlit as st

# Custom imports
from multipage import MultiPage
from pages import (
    donnees_preprocessing, exploration_dataviz, modeles_ML, reduction_dimension, strategie_paris
)

app = MultiPage()

# Title of the main page
# st.title("Bookmakerspy")

# Add all your application here
app.add_page("1. Données & pre-processing", donnees_preprocessing.app)
app.add_page("2. Exploration & dataviz", exploration_dataviz.app)

app.add_page("4. Réduction de dimension", exploration_dataviz.app)
app.add_page("5. Modèles de Machine Learning", exploration_dataviz.app)
app.add_page("6. Stratégie de paris", exploration_dataviz.app)
# The main app
app.run()