import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.balloons()
st.write("""
# Loan Eligible Prediction App
Application to predict the **Loan Eligible** based personal and information data input!
""")
st.write('---')

pickle_in = open("lr.pkl", 'rb')
classifier = pickle.load(pickle_in)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('LOAN ELIGIBLE PREDICT')
st.sidebar.subheader(' please, input your data in below: ')
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
input = np.array([[Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area]])
input_data = pd.DataFrame(input, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area'], index=['Input'])
st.write('''
### Result Your Input Data
''')
st.dataframe(input_data)
if submit:
    prediction = classifier.predict([[Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area]])
    if prediction == 0:
        code = 'Sorry {}, you are not eligible to apply for loan!'.format(name)
        st.code(code,language='python')
    else:
        code1 = 'Congrats {}, you are eligible to apply for loan!'.format(name)
        st.code(code1, language='python')
        
st.write('---')
st.write('''
### Display Dataset for Modeling **Loan Eligible Prediction**
''')
train = pd.read_csv('loan-train.csv')
st.write('Data Shape: ' + str(train.shape[0]) + ' rows and ' + str(train.shape[1]) + ' columns.')
st.dataframe(train)
    
col1, col2 = st.columns(2)
with col1:
    st.write('''
### Distribution of Gender
''')
    sns.countplot(x='Gender', data=train)
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(train['Gender'].value_counts())
with col2:
    st.write('''
### Distribution of Married Status
''')
    sns.countplot(train.Married)
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(train['Married'].value_counts())
col3, col4 = st.columns(2)
with col3:
    st.write('''
### Distribution of Dependents
''')
    sns.countplot(x='Dependents', data=train)
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(train['Dependents'].value_counts())
with col4:
    st.write('''
### Distribution of Education
''')
    sns.countplot(train.Education)
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(train['Education'].value_counts())
