import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

st.write("""
# Loan Eligible Prediction App
This app predicts the **Loan Eligible**!
""")
st.write('---')
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('LOAN ELIGIBLE PREDICT')

df = pd.read_csv('loan-eligible.csv', sep=';')
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LogisticRegression(C=1)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
#saving model
pickle_out = open("lr.pkl", "wb")
pickle.dump(lr, pickle_out)
pickle_out.close()

pickle_in = open("lr.pkl", 'rb')
classifier = pickle.load(pickle_in)

st.sidebar.header('Click hide if u want hide form')
if not st.sidebar.checkbox("Hide", True, key="1"):
    st.title('Loan Eligible Prediction, input your data in below: ')
    name = st.text_input('Input Your Name: ')
    Gender = st.number_input('Gender: Male (1) Female (0)')
    Married = st.number_input('Married: Yes (1) No (0)')
    Dependents = st.number_input('Dependents: 1/0/2/3')
    Education = st.number_input('Education: Graduate (0) Not Graduate (1)')
    Self_Employed = st.number_input('Self_Employed: No (0) Yes (1)')
    ApplicantIncome = st.number_input('ApplicantIncome: ')
    CoapplicantIncome = st.number_input('CoapplicantIncome: ')
    LoanAmount = st.number_input('LoanAmount: ')
    Loan_Amount_Term = st.number_input('Loan_Amount_Term: ')
    Credit_History = st.number_input('Credit_History: ')
    Property_Area = st.number_input('Property_Area: Rural (0) Urban (2) Semiurban (1)')
submit = st.button('Predict')
if submit:
    prediction = classifier.predict([[Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area]])
    if prediction == 0:
        st.write('Maaf', name, ', anda tidak memenuhi syarat untuk pengajuan hutang')
    else:
        st.write('Selamat', name, ', anda memenuhi syarat untuk pengajuan hutang')
