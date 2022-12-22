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
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from numpy import array
from sklearn.preprocessing import StandardScaler
import altair as alt

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

dataset = load_file()

st.dataframe(dataset)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split( 
                        X,y,test_size = 0.30, random_state = 101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# train the model on train set without using GridSearchCV 
model = SVC() 
model.fit(X_train, y_train)

# print prediction results 
predictions = model.predict(X_test) 
#report = metrics.classification_report(y_test, predictions)

def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    
    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)
    
    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total
    
    class_report_df['avg / total'] = avg

    return class_report_df.T

print(classification_report(y_true=y_test, y_pred=predictions, digits=6))
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1) 
   
# fitting the model for grid search 
grid.fit(X_train, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 
grid_predictions = grid.predict(X_test)
# print classification report 
print(classification_report(y_true=y_test, y_pred=grid_predictions, digits=6))
