import streamlit as st
import pickle

st.write("""
# Loan Eligible Prediction App
This app predicts the **Loan Eligible**!
""")
st.write('---')
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('LOAN ELIGIBLE PREDICT')

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
