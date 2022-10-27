from random import shuffle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

st.markdown("""
# Họ và tên: Nguyễn Nhật Trường
# MSSV: 20522087
""")


uploaded_file = st.file_uploader("Chọn dataset")
if uploaded_file is not None:
    # To read file as bytes to local disk:
    bytes_data = uploaded_file.getvalue()
    with open('dataset/' + uploaded_file.name, 'wb') as f:
        f.write(bytes_data)
    
    # Load data
    df = pd.read_csv('dataset/' + uploaded_file.name)

    # Show features
    st.text("Show your DataFrame")
    st.table(df.iloc[:,0:-1])
    list_feature = df.columns
    numOfFeature = len(list_feature)
    checkpoint = st.columns(numOfFeature)

    
    for i in range(numOfFeature):
        with checkpoint[i]:
            # listChoosed = []
            st.checkbox(list_feature[i])
            # if chooseFeature:
            #     listChoosed = list_feature[i]
    
    # st.text(listChoosed)

    # Show target to predict
    st.text("Output") # Profit (Cột cuối)
    st.table(df.iloc[:,-1])

    option = st.selectbox(
    'Chọn cách chia dữ liệu?',
    ('Normal', 'K Fold'))

    st.write('You selected:', option)

    
    if option == "K Fold":
        num_folds = st.number_input("Nhập K: ", min_value = 2)

        # Define K-Fold Cross validation
        k_fold = KFold(n_splits = num_folds, shuffle=True)
        
    
    if option =="Normal":
        train_split = st.select_slider("Nhập kích thước của tập train", options=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    
    

    

    
st.button("Run")



