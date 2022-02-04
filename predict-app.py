import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.write("""
# Loan Eligible Prediction App
Application to predict the **Loan Eligible** based personal and information data input!
""")
st.write('---')
train = pd.read_csv('loan-train.csv')
st.write('Data Shape: ' + str(train.shape[0]) + ' rows and ' + str(train.shape[1]) + ' columns.')
st.dataframe(train)
test = pd.read_csv('loan-test.csv')
st.write('Data Shape: ' + str(test.shape[0]) + ' rows and ' + str(test.shape[1]) + ' columns.')
st.dataframe(test)

for i in ('Gender', 'Married', 'Dependents', 'Education'):
    st.write(train[i].value_counts(),"\n")
col1, col2 = st.columns(2)
with col1:
    sns.countplot(x='Gender', data=train)
    plt.show()
    st.pyplot()
with col2:
    sns.countplot(train.Married)
    plt.show()
    st.pyplot()

col3, col4 = st.columns(2)
with col3:
    sns.countplot(x='Dependents', data=train)
    plt.show()
    st.pyplot()
with col4:
    sns.countplot(train.Education)
    plt.show()
    st.pyplot()

cleaned_data_train = train.drop(columns=['Loan_ID'], axis=1)
cleaned_data_train = cleaned_data_train.dropna()
cleaned_data_train.reset_index(drop=True, inplace=True)
cleaned_data_encode = cleaned_data_train.copy()
for i in cleaned_data_encode.columns:
    if cleaned_data_encode[i].dtype == np.int64:
        continue
    cleaned_data_encode[i] = LabelEncoder().fit_transform(cleaned_data_encode[i])

lr = LogisticRegression().fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
st.write(classification_report(y_train, y_train_pred))

lr = LogisticRegression().fit(x_test, y_test)
y_test_pred = lr.predict(x_test)
st.write(classification_report(y_test, y_test_pred))

conf_train = pd.DataFrame((confusion_matrix(y_train,y_train_pred)),('No','Yes'),('No','Yes'))
st.write(conf_train)

plt.figure(figsize=(6,5))
hmp = sns.heatmap(conf_train, annot=True, annot_kws={'size':11}, fmt='d', cmap='YlGnBu')
hmp.yaxis.set_ticklabels(hmp.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
hmp.xaxis.set_ticklabels(hmp.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix Data Train')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()
st.pyplot()


conf_test = pd.DataFrame((confusion_matrix(y_test,y_test_pred)),('No','Yes'),('No','Yes'))
st.write(conf_test)

plt.figure(figsize=(6,5))
hmp = sns.heatmap(conf_test, annot=True, annot_kws={'size':11}, fmt='d', cmap='YlGnBu')
hmp.yaxis.set_ticklabels(hmp.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
hmp.xaxis.set_ticklabels(hmp.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix Data Test')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('LOAN ELIGIBLE PREDICT')

pickle_in = open("lr.pkl", 'rb')
classifier = pickle.load(pickle_in)

st.sidebar.header(' please, input your data in below: ')
name = st.sidebar.text_input('Input Your Name: ')
Gender = st.sidebar.number_input('Gender: Male (1) Female (0)')
Married = st.sidebar.number_input('Married: Yes (1) No (0)')
Dependents = st.sidebar.number_input('Dependents: 1/0/2/3')
Education = st.sidebar.number_input('Education: Graduate (0) Not Graduate (1)')
Self_Employed = st.sidebar.number_input('Self_Employed: No (0) Yes (1)')
ApplicantIncome = st.sidebar.number_input('ApplicantIncome: ')
CoapplicantIncome = st.sidebar.number_input('CoapplicantIncome: ')
LoanAmount = st.sidebar.number_input('LoanAmount: ')
Loan_Amount_Term = st.sidebar.number_input('Loan_Amount_Term: ')
Credit_History = st.sidebar.number_input('Credit_History: ')
Property_Area = st.sidebar.number_input('Property_Area: Rural (0) Urban (2) Semiurban (1)')
submit = st.sidebar.button('Predict')

if submit:
    prediction = classifier.predict([[Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area]])
    if prediction == 0:
        st.write('Maaf', name, ', anda tidak memenuhi syarat untuk pengajuan hutang')
    else:
        st.write('Selamat', name, ', anda memenuhi syarat untuk pengajuan hutang')
