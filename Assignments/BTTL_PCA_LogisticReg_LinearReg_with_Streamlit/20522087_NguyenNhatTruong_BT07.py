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
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import numpy as np
from numpy import array
import altair as alt

st.markdown("""
# Họ và tên: Nguyễn Nhật Trường
# MSSV: 20522087
""")


st.title('PCA applied to Wine dataset')

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


dataset = load_wine()

df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
df["target"] = dataset.target

st.dataframe(df)


list_feature = df.columns
# ---------------------------
list_name = []
for i in range(len(df.columns)):
    list_name.append(list_feature[i])

data_backup = df.copy()


st.markdown("""
    # Feature of table to be used
    """)
st.dataframe(df.iloc[:, :-1])

type = st.sidebar.selectbox('Please choose a dimensionality reduction', ('PCA', 'LDA'))

if type == 'PCA':
    num_pca = st.sidebar.number_input('The minimum value is an integer of 3 or more.', min_value = 3,
    max_value=13,
    value = 3,
    step = 1,
    )

if type == 'LDA':
    st.write("Coming soon")

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

    # Train/test
    X = df.iloc[:, :-1]
    y = df['target']

    # Standardization
    scaler = StandardScaler()
    x_std = scaler.fit_transform(X)


    # -------------- PCA 
    pca = PCA(n_components=num_pca)
    x_pca = pca.fit_transform(x_std)

    st.markdown("""
    # Dataframe after applied PCA
    """)
    st.dataframe(x_pca)
    st.sidebar.markdown(
    r"""
    ### Select the principal components to plot
    ex. Choose '1' for PCA 1
    """
    )
    # Index of PCA, e.g. 1 for PCA 1, 2 for PCA 2, etc..
    idx_x_pca = st.sidebar.selectbox("x axis is the principal component of ", np.arange(1, num_pca+1), 0)
    idx_y_pca = st.sidebar.selectbox("y axis is the principal component of ", np.arange(1, num_pca+1), 1)
    idx_z_pca = st.sidebar.selectbox("z axis is the principal component of ", np.arange(1, num_pca+1), 2)

    # Axis label
    x_lbl, y_lbl, z_lbl = f"PCA {idx_x_pca}", f"PCA {idx_y_pca}", f"PCA {idx_z_pca}"
    # data to plot
    x_plot, y_plot, z_plot = x_pca[:,idx_x_pca-1], x_pca[:,idx_y_pca-1], x_pca[:,idx_z_pca-1]
    # Split the dataset

    trace1 = go.Scatter3d(
    x=x_plot, y=y_plot, z=z_plot,
    mode='markers',
    marker=dict(
        size=5,
        color=y,
        )
    )   

    # Create an object for graph layout
    fig = go.Figure(data=[trace1])
    fig.update_layout(scene = dict(
                        xaxis_title = x_lbl,
                        yaxis_title = y_lbl,
                        zaxis_title = z_lbl),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10),
                        )
    # ----------------------------------------
    st.plotly_chart(fig, use_container_width=True)
    # ----------------------------------------
    test_size = st.sidebar.number_input('Validation dataset size (rate: 0.0 -> 1.0)', min_value = 0.0,
    max_value=1.0,
    value = 0.2,
    step = 0.1,
    )

    # random seed
    random_seed = st.sidebar.number_input('Set random seed (0 -> ):', value =0, step=1, min_value=0)
    X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=test_size, random_state=random_seed)

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

        prediction_prob = logisticRegr.predict_proba(X_test)

        ll = log_loss(y_test, prediction_prob)
        st.write("Log loss", ll)

        # Visualize result after evaluate
        fig = plt.figure(figsize = (10, 5))
        plt.bar(['Precistion', 'Recall', 'F1 Score'], [precision, recall, f1], color='maroon', width = 0.4)
        st.pyplot(fig)
        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1)

if option == "K Fold Cross validation":

    
    # Train/test
    X = df.iloc[:, :-1]
    y = df['target']

    # Standardization
    scaler = StandardScaler()
    x_std = scaler.fit_transform(X)


    # -------------- PCA 
    pca = PCA(n_components=num_pca)
    x_pca = pca.fit_transform(x_std)
    

    st.markdown("""
    # After applied PCA
    """)
    st.dataframe(x_pca)
    st.sidebar.markdown(
    r"""
    ### Select the principal components to plot
    ex. Choose '1' for PCA 1
    """
    )
    # Index of PCA, e.g. 1 for PCA 1, 2 for PCA 2, etc..
    idx_x_pca = st.sidebar.selectbox("x axis is the principal component of ", np.arange(1, num_pca+1), 0)
    idx_y_pca = st.sidebar.selectbox("y axis is the principal component of ", np.arange(1, num_pca+1), 1)
    idx_z_pca = st.sidebar.selectbox("z axis is the principal component of ", np.arange(1, num_pca+1), 2)

    # Axis label
    x_lbl, y_lbl, z_lbl = f"PCA {idx_x_pca}", f"PCA {idx_y_pca}", f"PCA {idx_z_pca}"
    # data to plot
    x_plot, y_plot, z_plot = x_pca[:,idx_x_pca-1], x_pca[:,idx_y_pca-1], x_pca[:,idx_z_pca-1]
    # Split the dataset

    trace1 = go.Scatter3d(
    x=x_plot, y=y_plot, z=z_plot,
    mode='markers',
    marker=dict(
        size=5,
        color=y,
        )
    )   

    # Create an object for graph layout
    fig = go.Figure(data=[trace1])
    fig.update_layout(scene = dict(
                        xaxis_title = x_lbl,
                        yaxis_title = y_lbl,
                        zaxis_title = z_lbl),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10),
                        )
    # ----------------------------------------
    st.plotly_chart(fig, use_container_width=True)
    # ----------------------------------------

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
        X_train, X_test = x_pca[train_index], x_pca[test_index]
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
        prediction_prob = logisticRegr.predict_proba(X_test)
        ll.append(log_loss(Y_test, prediction_prob))

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


