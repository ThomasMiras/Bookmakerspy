import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style="white")
sns.set_palette("pastel")

def app():

    st.header("Bookmakerspy - Simulation par un modèle de Poisson")
    st.subheader("Rappels sur le nombre de buts par match")