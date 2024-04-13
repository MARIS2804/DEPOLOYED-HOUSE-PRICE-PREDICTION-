import streamlit as st 
import pandas as pd
from sklearn.linear_model import LinearRegression
a=LinearRegression()
df=pd.read_csv('clean_data.csv')
df['BATH']=df['bathroom'].fillna((int)(df['bathroom'].mean()))
df['AGE']=df['age'].fillna((int)(df['age'].mean()))
df=df[['price','area','bhk','BATH','AGE']]
X=df[['area','bhk','BATH','AGE']]
y=df.price
a.fit(X,y)

st.title("HOUSE PRICE PREDICTION")
area=st.slider("GIVE AREA OF THE HOUSE",300,6700)
bhk=st.slider("GIVE BHK OF THE HOUSE",1,8)
bath=st.slider("GIVE NO. OF BATHROOM",1,7)
age=st.number_input("GIVE AGE OF THE HOUSE")
if st.button("PREDICT"):
    k=a.predict([[area,bhk,bath,age]])*1000
    st.write("PREDICTED HOUSE PRICE IS",k)
