import streamlit as st
import eda, predict

with st.sidebar:
    st.title('Navigation')
    selection = st.radio('Go to page', ['EDA','Prediction'])

if selection == 'EDA':
    eda.run()

if selection == 'Prediction':
    predict.run()
    predict.run_file()