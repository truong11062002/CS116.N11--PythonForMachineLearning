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
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import array
import altair as alt

st.markdown("""
# Họ và tên: Nguyễn Nhật Trường
# MSSV: 20522087
""")


st.title("CHỌN DATASET ĐỂ THỰC HIỆN QUÁ TRÌNH PREDICT (CSV)")
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

st.markdown("""
    # Show name features
    """)
# Get features name
for col in df.columns:
    st.write(col)
with col1: 
    # Show data train
    st.markdown("""
    # Data train
    """)
    X_train = df.iloc[:, 0:-1]
    st.write(X_train)
with col2:    
    st.markdown("""
    # Data test
    """)
    # target column
    target_column = df.iloc[:, -1]
    st.write(target_column)

# list data
list_feature = df.columns
numOfFeature = len(list_feature)
# Plot some feature
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

# Giúp cho dữ liệu tuyến tính hơn và không bị chênh lệch quá cao
df_std = df_log.copy()

# Bị bug và đang trong quá trình chỉnh sửa (Coming soon....)
# if bool_std == 'Yes':
# 	Std_Features_chosen = []
# 	Std_Features_NonUsed = right_column.multiselect(
# 					'Select the variables NOT to be standardized (categorical variables)', 
# 					df_log.drop(columns=target_column).columns
# 					)
# 	for name in df_log.drop(columns=target_column).columns:
# 		if name in Std_Features_NonUsed:
# 			continue
# 		else:
# 			Std_Features_chosen.append(name)
# 	# Perform standardization
# 	scaler = StandardScaler()
# 	scaler.fit(df_std[Std_Features_chosen])
# 	df_std[Std_Features_chosen] = scaler.transform(df_std[Std_Features_chosen])


# Select split data way
option = st.selectbox(
'Chọn cách chia dữ liệu?',
('Train/Test split', 'K Fold Cross validation'))
st.write('You selected:', option)


if option == "Train/Test split":
    
    # Split the dataset
    left_column, right_column = st.columns(2)
    test_size = left_column.number_input('Validation dataset size (rate: 0.0 -> 1.0)', min_value = 0.0,
    max_value=1.0,
    value = 0.2,
    step = 0.1,
    )

    # random seed
    random_seed = right_column.number_input('Set random seed (0 -> ):', value =0, step=1, min_value=0)

    # Set train and test
    X = df_std.iloc[:,: -1]
    y = df_std.iloc[:, -1]

    X_new = X.copy()
    # One hot encoder
    for i in range(numOfFeature):
        if list_feature[i] == "Position":
            X_column = X_new['Position'].to_frame()
            X_new = X_new.drop(columns=['Position'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X_column).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)

        if list_feature[i] == "State":
            X_column = X_new['State'].to_frame()
            X_new = X_new.drop(columns=['State'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X_column).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)
    

    st.markdown("""
    # Dataframe after One-Hot Encoder
    """)

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
    
    # On click run train/test split
    btn_train_test = st.button("Run with train/test split", key="1")
    if btn_train_test:
        st.write(f'R2 score: {r2:.2f}')
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'Mean Absolute Error: {mae:.2f}')
        st.title("Plot the result")
        left_column, right_column = st.columns(2)
        show_train = left_column.radio(
                        'Show the training dataset:', 
                        ('Yes','No')
                        )
        show_val = right_column.radio(
                        'Show the test dataset:', 
                        ('Yes','No')
                        )

        # default axis range
        y_max_train = max([max(Y_train), max(Y_pred_train)])
        y_max_val = max([max(Y_test), max(Y_pred_test)])
        y_max = int(max([y_max_train, y_max_val])) 

        # interactive axis range
        left_column, right_column = st.columns(2)
        x_min = left_column.number_input('x_min:',value=0,step=1)
        x_max = right_column.number_input('x_max:',value=y_max,step=1)
        left_column, right_column = st.columns(2)
        y_min = left_column.number_input('y_min:',value=0,step=1)
        y_max = right_column.number_input('y_max:',value=y_max,step=1)


        fig = plt.figure(figsize=(3, 3))
        if show_train == 'Yes':
            plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
        if show_val == 'Yes':
            plt.scatter(Y_test, Y_pred_test,lw=0.1,color="b",label="test data")
        plt.xlabel("y_target",fontsize=8)
        plt.ylabel("y_target of prediction",fontsize=8)
        plt.xlim(int(x_min), int(x_max)+5)
        plt.ylim(int(y_min), int(y_max)+5)
        plt.legend(fontsize=6)
        plt.tick_params(labelsize=6)
        st.pyplot(fig)
    
    
if option == "K Fold Cross validation":
    left_column, right_column = st.columns(2)

    # Num of folds
    num_folds = left_column.number_input("Nhập K: ", step=1, min_value = 2)

    # random seed
    random_seed = right_column.number_input('Set random seed (0 -> ):', value =0, step=1, min_value=0)

    # Define K-Fold Cross validation
    kf = KFold(n_splits = int(num_folds), shuffle=True, random_state = random_seed)
    
    # Split data
    feature = df_std.iloc[:,: -1] 
    target = df_std.iloc[:, -1]

    X_new = feature.copy()

    # One hot encoder
    for i in range(numOfFeature):
        if list_feature[i] == "Position":
            X_column = X_new['Position'].to_frame()
            X_new = X_new.drop(columns=['Position'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X_column).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)

        if list_feature[i] == "State":
            X_column = X_new['State'].to_frame()
            X_new = X_new.drop(columns=['State'])
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(X_column).astype(int).toarray())
            X_new = pd.concat([X_new, ohe_df], axis=1)

    X = X_new.to_numpy()
    y = target.to_numpy()
    btn_kf = st.button("Run with k-fold", key="2")
    if btn_kf:
        r2_list = []
        mse_list = []
        mae_list = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = y[train_index], y[test_index]

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
            
            r2_list.append(r2)
            mse_list.append(mse)
            mae_list.append(mae)

            
            # list_fold.append(k)

        df_r2 = pd.DataFrame(r2_list, columns=["Score"])
        df_mse = pd.DataFrame(mse_list, columns=["Score"])
        df_mae = pd.DataFrame(mae_list, columns=["Score"])

        df_mae.index = df_mae.index.factorize()[0] + 1
        df_mse.index = df_mse.index.factorize()[0] + 1
        df_r2.index = df_r2.index.factorize()[0] + 1

        df_mae.reset_index(inplace=True)
        df_mae = df_mae.rename(columns = {'index':'Fold'})
        df_mse.reset_index(inplace=True)
        df_mse = df_mse.rename(columns = {'index':'Fold'})
        df_r2.reset_index(inplace=True)
        df_r2 = df_r2.rename(columns = {'index':'Fold'})

        r2, mse, mae = st.tabs(["R2", "MSE", "MAE"])

        with r2:
            c = alt.Chart(df_r2).mark_bar(size=30).encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
            st.altair_chart(c, use_container_width=False)
        with mse:
            c = alt.Chart(df_mse).mark_bar(size=30).encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
            st.altair_chart(c, use_container_width=False)
        with mae:
            c = alt.Chart(df_mae).mark_bar(size=30).encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
            st.altair_chart(c, use_container_width=False)