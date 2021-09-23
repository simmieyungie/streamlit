import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
import altair as alt

#decide what data to visualize
#create drop now to show whether to use the old data or new

def app():
    st.write("Choose the Dataset to be visualize")
    data_vis = st.selectbox("Which dataset do you want to visualize", options = ["new_main_data", "main_data"])
    
    #decide whic meta data is to be read in when data to be visualized is chosen
    if data_vis == "main_data":
        meta_read = "data/metadata/column_type_desc.csv"
    else:
        meta_read = "data/metadata/new_data_column_type_desc.csv"
    
    if data_vis + ".csv" not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        read = "data/"+ data_vis + ".csv"
        df_analysis = pd.read_csv(read)
        # df_visual = pd.DataFrame(df_analysis)
        df_visual = df_analysis.copy()
        cols = pd.read_csv(meta_read)
        
        #descriptive statistics
        #numerical data analysis
        st.markdown("### Descriptive Statistics")
        st.dataframe(df_visual.describe().head())
        
        corr = df_visual.corr(method='pearson')
        
        fig2, ax2 = plt.subplots()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # Colors
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0,ax=ax2)
        ax2.set_title("Correlation Matrix")
        st.pyplot(fig2)
        
        
        #categorical columns analysis
        st.markdown("### Categorical Analysis")
        #categorical columns
        cats = cols[cols.type == "categorical"]
        cat_select = st.selectbox("Select Column to visualize", options = cats.column_name)
        st.bar_chart(df_analysis[cat_select].value_counts())
        