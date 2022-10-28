from secrets import choice
from tkinter import Label
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np

st.markdown("""
# Họ và tên: Nguyễn Nhật Trường
# MSSV: 20522087
""")


st.title("Chọn dataset để thực hiện predict")
uploaded_file = st.file_uploader("")
if uploaded_file is None:
    st.stop()

if uploaded_file is not None:
    # To read file as bytes to local disk:
    bytes_data = uploaded_file.getvalue()
    with open('dataset/' + uploaded_file.name, 'wb') as f:
        f.write(bytes_data)
    
    
# Load data and show dataset
df = pd.read_csv('dataset/' + uploaded_file.name)

# Show the table data
if st.checkbox('Show the dataset as table data'):
	st.dataframe(df)


col1, col2 = st.columns(2)
st.text("Show features name")
# Get features name
for col in df.columns:
    st.write(col)
with col1: 
    # Show data train
    st.text("Show data train")
    X_train = df.iloc[:, 0:-1]
    st.write(X_train)
with col2:    
    st.text("Data test")
    # target column
    target_column = df.iloc[:, -1]
    st.write(target_column)

# list data
list_feature = df.columns
numOfFeature = len(list_feature)
# Select features to predict
if st.checkbox('Show the relation between "Target" vs each variable'):
	checked_variable = st.selectbox('Select one variable:', list_feature)

    # Plot
	fig, ax = plt.subplots(figsize=(5, 3))
	ax.scatter(x=df[checked_variable], y=target_column)
	plt.xlabel(checked_variable)
	plt.ylabel("y_target")
	st.pyplot(fig)



# Select the variables NOT to be used
Features_chosen = []
Features_NonUsed = st.multiselect(
	'Select the variables NOT to be used', 
	list_feature)
st.write("Features not to be used", Features_NonUsed)

# Remove column this unused variable from the dataset 
df = df.drop(columns=Features_NonUsed)


left_column, right_column = st.columns(2)
bool_log = left_column.radio(
			'Perform the logarithmic transformation?', 
			('No','Yes')
			)

df_log, Log_Features = df.copy(), []
if bool_log == 'Yes':
	Log_Features = right_column.multiselect(
					'Select the variables you perform the logarithmic transformation', 
					df.columns
					)
	# Perform logarithmic transformation
	df_log[Log_Features] = np.log(df_log[Log_Features])

left_column, right_column = st.columns(2)
bool_std = left_column.radio(
			'Perform the standardization?', 
			('No','Yes')
			)

df_std = df_log.copy()
if bool_std == 'Yes':
	Std_Features_chosen = []
	Std_Features_NonUsed = right_column.multiselect(
					'Select the variables NOT to be standardized (categorical variables)', 
					df_log.drop(columns=target_column).columns
					)
	for name in df_log.drop(columns=target_column).columns:
		if name in Std_Features_NonUsed:
			continue
		else:
			Std_Features_chosen.append(name)
	# Perform standardization
	scaler = preprocessing.StandardScaler()
	scaler.fit(df_std[Std_Features_chosen])
	df_std[Std_Features_chosen] = scaler.transform(df_std[Std_Features_chosen])


# test size 
option = st.selectbox(
'Chọn cách chia dữ liệu?',
('Normal', 'K Fold Cross validation'))

st.write('You selected:', option)


if option == "Normal":
    
    # Split the dataset
    left_column, right_column = st.columns(2)
    test_size = left_column.number_input('Validation dataset size (rate: 0.0 -> 1.0)', min_value = 0.0,
    max_value=1.0,
    value = 0.2,
    step = 0.1,
    )

    # random seed
    random_seed = right_column.number_input('Set random seed (0 -> ):', value =0, step=1, min_value=0)

    X = df_std.iloc[:,: -1]
    y = df_std.iloc[:, -1]

    # One hot Encoder
    X_new = X.copy()
    
    for i in range(numOfFeature):
        if list_feature[i] == "Position":
            X_new = X_new.drop(columns=['Position'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)
        if list_feature[i] == "State":
            X_new = X_new.drop(columns=['State'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)

    st.dataframe(X_new)

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X_new , y, test_size = test_size, random_state=random_seed)


    # Model linear regression
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Validation
    Y_pred_train = regressor.predict(X_train)
    Y_pred_test = regressor.predict(X_test)

    # Using the r2 value as a validation indicator
    r2 = r2_score(Y_test, Y_pred_test)
    mse = mean_squared_error(Y_test, Y_pred_test, squared=False)
    mae = mean_absolute_error(Y_test, Y_pred_test)
    if st.button("Run"):
        st.write(f'R2 score: {r2:.2f}')
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'Mean Absolute Error: {mae:.2f}')


if option == "K Fold Cross validation":
    num_folds = st.number_input("Nhập K: ", step=1, min_value = 2)

    # Define K-Fold Cross validation
    k_fold = KFold(n_splits = int(num_folds), shuffle=True, random_state = None)

    X = df_std.iloc[:,: -1]
    y = df_std.iloc[:, -1]

    X_new = X.copy()
    
    for i in range(numOfFeature):
        if list_feature[i] == "Position":
            X_new = X_new.drop(columns=['Position'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)
        if list_feature[i] == "State":
            X_new = X_new.drop(columns=['State'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)

    # Using Linear regression model to predict
    model = LinearRegression()
    scores = cross_val_score(model, X_new, y, cv=k_fold)
    btn = st.button("Run")  

    if btn:
        st.write("Score by Cross-Validation (K-Fold): ", scores)
        st.write("Mean: ", scores.mean())
        st.write("Standard Deviation: ", scores.std())

    




    



