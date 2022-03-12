# -*- coding: utf-8 -*-

import streamlit as st

# Custom imports
from multipage import MultiPage
from pages import (
    donnees_preprocessing, exploration_dataviz, poisson_modelisation
)

app = MultiPage()

# Title of the main page
# st.title("Bookmakerspy")

# Add all your application here
app.add_page("1. Données & pre-processing", donnees_preprocessing.app)
app.add_page("2. Exploration & dataviz", exploration_dataviz.app)
app.add_page("3. Modélisation par un modèle de Poisson", poisson_modelisation.app)

# The main app
app.run()