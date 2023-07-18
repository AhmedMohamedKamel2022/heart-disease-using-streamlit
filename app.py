import time
import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction App")
def main():

    right, left = st.columns(2)
    with right:
        st.header('Heart Disease Prediction App')
        st.write('---')
        st.markdown("""\n
        ##### The goal of this application is to classify the patient has a heart disease or not based on the characteristics or data of this patient.
        ---

        ##### To predict whether the patient has a heart disease or not, just follow these steps:
        ##### 1. Enter information describing the patient's data.
        ##### 2. Press the "Predict" button and wait for the result.
        """)
    
    with left:
        st.image('images/doctor.png', width=260)

    featuress = pd.read_csv("data/new_data.csv")
    target = featuress.drop(columns=['HeartDisease'])

    def user_input_features():
        st.sidebar.write('# Enter Your Data...')

        Sex = st.sidebar.radio('Sex', 
                                options=(Sex for Sex in featuress.Sex.unique()))

        Smoking = st.sidebar.radio("Smoking", 
                                    options=(Smoking for Smoking in featuress.Smoking.unique()))

        Stroke = st.sidebar.radio("Stroke", 
                                    options=(Stroke for Stroke in featuress.Stroke.unique()))                   

        AlcoholDrinking = st.sidebar.radio("AlcoholDrinking",
                                    options=(AlcoholDrinking for AlcoholDrinking in featuress.AlcoholDrinking.unique()))

        DiffWalking = st.sidebar.radio("DiffWalking", 
                                    options=(DiffWalking for DiffWalking in featuress.DiffWalking.unique()))
                                    
        Diabetic = st.sidebar.radio("Diabetic", 
                                    options=(Diabetic for Diabetic in featuress.Diabetic.unique()))

        PhysicalActivity = st.sidebar.radio("PhysicalActivity", 
                                    options=(PhysicalActivity for PhysicalActivity in featuress.PhysicalActivity.unique()))

        Asthma = st.sidebar.radio("Asthma", 
                                    options=(Asthma for Asthma in featuress.Asthma.unique()))     

        KidneyDisease = st.sidebar.radio("KidneyDisease", 
                                    options=(KidneyDisease for KidneyDisease in featuress.KidneyDisease.unique()))

        SkinCancer = st.sidebar.radio("SkinCancer", 
                                    options=(SkinCancer for SkinCancer in featuress.SkinCancer.unique()))     

        BMI = st.sidebar.selectbox("BMI",
                                    options=(BMI for BMI in featuress.BMI.unique()))

        AgeCategory = st.sidebar.selectbox("AgeCategory",
                                    options=(AgeCategory for AgeCategory in featuress.AgeCategory.unique()))                    

        GenHealth = st.sidebar.selectbox("GenHealth",
                                    options=(GenHealth for GenHealth in featuress.GenHealth.unique()))

        Race = st.sidebar.selectbox("Race",
                                    options=(Race for Race in featuress.Race.unique()))
        
        SleepTime = st.sidebar.number_input('SleepTime', min_value=0, max_value=24)
        
        PhysicalHealth = st.sidebar.number_input('PhysicalHealth', min_value=0, max_value=30)

        MentalHealth = st.sidebar.number_input('MentalHealth', min_value=0, max_value=30)

        data = {
            "BMI": BMI,
            "Smoking": Smoking,
            "AlcoholDrinking": AlcoholDrinking,
            "Stroke": Stroke,
            "PhysicalHealth": PhysicalHealth,
            "MentalHealth": MentalHealth,
            "DiffWalking": DiffWalking,
            "Sex": Sex,
            "AgeCategory": AgeCategory,
            "Race": Race,
            "Diabetic": Diabetic,
            "PhysicalActivity": PhysicalActivity,
            "GenHealth": GenHealth,
            "SleepTime": SleepTime, 
            "Asthma": Asthma,
            "KidneyDisease": KidneyDisease,
            "SkinCancer": SkinCancer
            }

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    df = pd.concat([input_df,target],axis=0)

    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    model = joblib.load("Models/model.h5")
    scaler = joblib.load("Models/scaler.h5")

    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)

    if st.sidebar.button('Predict'):
    
        my_bar = st.sidebar.progress(0)
        for percent_complete in range(100):
            time.sleep(.001)
            my_bar.progress(percent_complete + 1)

        if prediction[0] == 0:
            st.sidebar.success("# The patient does not have heart disease")
        else:
            st.sidebar.warning("# The patient has heart disease")

if __name__ == "__main__":
    main()
