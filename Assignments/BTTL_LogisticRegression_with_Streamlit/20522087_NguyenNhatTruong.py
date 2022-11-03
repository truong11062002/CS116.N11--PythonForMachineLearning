import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

import numpy as np
from numpy import array
import altair as alt

st.markdown("""
# Họ và tên: Nguyễn Nhật Trường
# MSSV: 20522087
""")


st.title("Logistic Regression with Streamlit")
def load_file():
    file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        st.markdown("""
            # Show the dataset as table data
            """)
        return data
    else:
        st.text("Please upload a csv file")


data = load_file()
st.dataframe(data)

list_feature = data.columns
list_name = []

for i in range(len(data.columns)):
    list_name.append(list_feature[i])

data_backup = data.copy()
features_used = st.multiselect('Please choose the features including target variable that go into the model', data.columns, default=list_name)
# Dataframe feature to be used
data = data.loc[:, features_used]

st.markdown("""
    # Feature of table to be used
    """)
st.dataframe(data)


type = st.sidebar.selectbox('Algorithm type', ('Classification', 'Regression'))
if type == 'Regression':
    chosen_classifier = st.sidebar.selectbox('Please choose a classifier', ('Random Forest', 'Linear Regression', 'Neural Network'))

elif type == 'Classification':
    chosen_classifier = st.sidebar.selectbox('Please choose a classifier', ('Logistic Regression', 'Naive Bayes', 'Neural Network'))
    if chosen_classifier == 'Logistic Regression':
        max_iter = st.sidebar.slider('max iterations', 1, 100, 10)

# Select split data way
option = st.sidebar.selectbox(
'Chọn cách chia dữ liệu?',
('Train/Test split', 'K Fold Cross validation'))
st.write('You selected:', option)

if option == "Train/Test split":
    # Prepare data
    target_options = data_backup.columns # Target option
    chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))

    # Train/test
    X = data.loc[:, data.columns != chosen_target]
    y = data[chosen_target]

    # Split the dataset
    test_size = st.sidebar.number_input('Validation dataset size (rate: 0.0 -> 1.0)', min_value = 0.0,
    max_value=1.0,
    value = 0.2,
    step = 0.1,
    )

    # random seed
    random_seed = st.sidebar.number_input('Set random seed (0 -> ):', value =0, step=1, min_value=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    predict_btn = st.sidebar.button('Predict', key=1)
    

    if predict_btn:
        st.text("Progress:")
        my_bar = st.progress(0)
        
        # Model logistic regression
        logisticRegr = LogisticRegression(max_iter=max_iter)
        logisticRegr.fit(X_train, y_train)

        Y_pred_train = logisticRegr.predict(X_train)
        Y_pred_test = logisticRegr.predict(X_test)

        precision = precision_score(y_test, Y_pred_test, average='macro')
        st.write("Precistion: ", precision)
        recall = recall_score(y_test, Y_pred_test, average='macro')
        st.write("Recall: ",recall)
        f1 = f1_score(y_test, Y_pred_test, average='macro')
        st.write("f1 score: ", f1)
        ll = log_loss(y_test, Y_pred_test)
        st.write("Log loss", ll)

        # Visualize result after evaluate
        fig = plt.figure(figsize = (10, 5))
        plt.bar(['Precistion', 'Recall', 'F1 Score'], [precision, recall, f1], color='maroon', width = 0.4)
        st.pyplot(fig)
        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1)

if option == "K Fold Cross validation":

    # Prepare data
    target_options = data.columns # Target option
    chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))

    # Train/test
    X = data.loc[:, data.columns != chosen_target]
    y = data[chosen_target]

    X = X.to_numpy()
    y = y.to_numpy()

    # Num of folds
    num_folds = st.sidebar.number_input("Nhập K: ", step=1, min_value = 2)
    # random seed
    random_seed = st.sidebar.number_input('Set random seed (0 -> ):', value =0, step=1, min_value=0)

    # Define K-Fold Cross validation
    kf = KFold(n_splits = int(num_folds), shuffle=True, random_state = random_seed)

    precision = []
    recall = []
    f1 = []
    ll = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        # Model logistic regression
        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train, Y_train)

        Y_pred_train = logisticRegr.predict(X_train)
        Y_pred_test = logisticRegr.predict(X_test)

        # Using the r2 value as a validation indicator
        precision.append(precision_score(Y_test, Y_pred_test, average='macro'))
        recall.append(recall_score(Y_test, Y_pred_test, average='macro'))
        f1.append(f1_score(Y_test, Y_pred_test, average='macro'))
        ll.append(log_loss(Y_test, Y_pred_test))

    st.write("Mean precision: ", np.mean(precision))
    st.write("Mean recall: ", np.mean(recall))
    st.write("Mean f1 score: ", np.mean(f1))
    st.write("Log loss score: ", np.mean(ll))

    df_precision = pd.DataFrame(precision, columns=["Score"])
    df_recall = pd.DataFrame(recall, columns=["Score"])
    df_f1 = pd.DataFrame(f1, columns=["Score"])

    df_f1.index = df_f1.index.factorize()[0] + 1
    df_recall.index = df_recall.index.factorize()[0] + 1
    df_precision.index = df_precision.index.factorize()[0] + 1

    df_recall.reset_index(inplace=True)
    df_recall = df_recall.rename(columns = {'index':'Fold'})
    df_precision.reset_index(inplace=True)
    df_precision = df_precision.rename(columns = {'index':'Fold'})
    df_f1.reset_index(inplace=True)
    df_f1 = df_f1.rename(columns = {'index':'Fold'})

    precision, recall, f1 = st.tabs(["Precision", "Recall", "F1"])

    with precision:
        c = alt.Chart(df_precision).mark_bar(size=30).encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
        st.altair_chart(c, use_container_width=False)
    with recall:
        c = alt.Chart(df_recall).mark_bar(size=30).encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
        st.altair_chart(c, use_container_width=False)
    with f1:
        c = alt.Chart(df_f1).mark_bar(size=30).encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
        st.altair_chart(c, use_container_width=False)

    
    predict_btn = st.sidebar.button('Predict', key=2)


