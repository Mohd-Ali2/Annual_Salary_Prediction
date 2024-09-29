import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


st.title("Annual Income Prediction")

# Categorical mappings (match them to how they were encoded during training)
workclass_map = {'Not Defiend': 0, ' Federal-gov': 1, ' Local-gov': 2, ' Never-worked': 3, ' Private': 4, ' Self-emp-inc': 5, ' Self-emp-not-inc': 6, ' State-gov': 7, ' Without-pay': 8}
education_map = {' 10th': 0, ' 11th': 1, ' 12th': 2, ' 1st-4th': 3, ' 5th-6th': 4, ' 7th-8th': 5, ' 9th': 6, ' Assoc-acdm': 7, ' Assoc-voc': 8, ' Bachelors': 9, ' Doctorate': 10, ' HS-grad': 11, ' Masters': 12, ' Preschool': 13, ' Prof-school': 14, ' Some-college': 15}
marital_status_map = {' Divorced': 0, ' Married-AF-spouse': 1, ' Married-civ-spouse': 2, ' Married-spouse-absent': 3, ' Never-married': 4, ' Separated': 5, ' Widowed': 6}
occupation_map = {' ?': 0, ' Adm-clerical': 1, ' Armed-Forces': 2, ' Craft-repair': 3, ' Exec-managerial': 4, ' Farming-fishing': 5, ' Handlers-cleaners': 6, ' Machine-op-inspct': 7, ' Other-service': 8, ' Priv-house-serv': 9, ' Prof-specialty': 10, ' Protective-serv': 11, ' Sales': 12, ' Tech-support': 13, ' Transport-moving': 14}
relationship_map = {' Husband': 0, ' Not-in-family': 1, ' Other-relative': 2, ' Own-child': 3, ' Unmarried': 4, ' Wife': 5}
race_map = {' Amer-Indian-Eskimo': 0, ' Asian-Pac-Islander': 1, ' Black': 2, ' Other': 3, ' White': 4}
sex_map = {'Male': 0, 'Female': 1}
native_country_map = {' ?': 0, ' Cambodia': 1, ' Canada': 2, ' China': 3, ' Columbia': 4, ' Cuba': 5, ' Dominican-Republic': 6, ' Ecuador': 7, ' El-Salvador': 8, ' England': 9, ' France': 10, ' Germany': 11, ' Greece': 12, ' Guatemala': 13, ' Haiti': 14, ' Holand-Netherlands': 15, ' Honduras': 16, ' Hong': 17, ' Hungary': 18, ' India': 19, ' Iran': 20, ' Ireland': 21, ' Italy': 22, ' Jamaica': 23, ' Japan': 24, ' Laos': 25, ' Mexico': 26, ' Nicaragua': 27, ' Outlying-US(Guam-USVI-etc)': 28, ' Peru': 29, ' Philippines': 30, ' Poland': 31, ' Portugal': 32, ' Puerto-Rico': 33, ' Scotland': 34, ' South': 35, ' Taiwan': 36, ' Thailand': 37, ' Trinadad&Tobago': 38, ' United-States': 39, ' Vietnam': 40, ' Yugoslavia': 41}

# User inputs
Age = st.number_input('Age', min_value=0, max_value=90)
Workclass = st.selectbox('Workclass', list(workclass_map.keys()))
Fnlwgt = st.number_input('Fnlwgt', min_value=0, max_value=1000000)
Education = st.selectbox('Education', list(education_map.keys()))
Education_num = st.number_input('Education-num', min_value=0, max_value=16)
Marital_status = st.selectbox('Marital-status', list(marital_status_map.keys()))
Occupation = st.selectbox('Occupation', list(occupation_map.keys()))
Relationship = st.selectbox('Relationship', list(relationship_map.keys()))
Race = st.selectbox('Race', list(race_map.keys()))
Sex = st.selectbox('Sex', list(sex_map.keys()))
Capital_gain = st.number_input('Capital-gain', min_value=0, max_value=100000)
Capital_loss = st.number_input('Capital-loss', min_value=0, max_value=100000)
Hours_per_week = st.number_input('Hours-per-week', min_value=0, max_value=168)
Native_country = st.selectbox('Native-country', list(native_country_map.keys()))

# Convert categorical inputs to numeric values using the defined mappings
Workclass = workclass_map[Workclass]
Education = education_map[Education]
Marital_status = marital_status_map[Marital_status]
Occupation = occupation_map[Occupation]
Relationship = relationship_map[Relationship]
Race = race_map[Race]
Sex = sex_map[Sex]
Native_country = native_country_map[Native_country]

# Prepare input for model
input_data = np.array([[Age, Workclass, Fnlwgt, Education, Education_num, Marital_status, Occupation, 
                        Relationship, Race, Sex, Capital_gain, Capital_loss, Hours_per_week, Native_country]])

# Scale the input data
scaled_values = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_values)

    # Display prediction
if st.button('Predict'):
    if prediction == 0:
        st.write("Predicted Annual Income is <=50K")
    else:
        st.write("Predicted Annual Income is >50K")
