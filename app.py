# import streamlit as st
# import pandas as pd
# import os
#
# # Import profiling capability
# # import pandas_profiling
# import ydata_profiling
# from streamlit_pandas_profiling import st_profile_report
#
# # ML stuff


# # scikit-learn==0.23.2
#
# with st.sidebar:
# 	st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
# 	st.title("AutoStreamML")
# 	choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
# 	st.info("This application you allows to build an automated ML pipeline using Streamlit, Pandas Profiling and Pycaret. And it's damn right magic!")
#
#
# #st.write("Hello World!")
#
# if os.path.exists("sourcedata.csv"):
# 	df = pd.read_csv("sourcedata.csv", index_col=None)
#
# if choice == "Upload":
# 	st.title("Upload your data for Modelling")
# 	file = st.file_uploader("Upload your dataset here")
# 	if file:
# 		#do something
# 		df = pd.read_csv(file, index_col=None)
# 		df.to_csv("sourcedata.csv", index=None)
# 		st.dataframe(df)
#
# if choice == "Profiling":
# 	st.title("Automated Exploratory Data Analysis")
# 	profile_report= df.profile_report()
# 	st_profile_report(profile_report)
#
# if choice == "ML":
# 	st.title("Machine Learning to go BRR***")
# 	target = st.selectbox("Select Your Target", df.columns)
# 	setup(df, target=target)
# 	setup_df = pull()
# 	st.info("This is the ML Experiment settings")
# 	st.dataframe(setup_df)
# 	best_model = compare_models()
# 	compare_df = pull()
# 	st.info("This is the ML Model")
# 	st.dataframe(compare_df)
# 	best_model
#
# if choice == "Download":
# 	pass

from operator import index
import streamlit as st
import plotly.express as px
# from pycaret.regression import setup, compare_models, pull, save_model, load_model
from pycaret.classification import setup, compare_models, pull, save_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

# if os.path.exists('./dataset.csv'):
if os.path.exists('./sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")