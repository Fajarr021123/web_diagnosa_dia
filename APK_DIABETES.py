import streamlit as st

st.write("""
# APLIKASI MENDIAGNOSA DIABETES
"""
)
import pandas as pd
dataset=pd.read_csv("diabetes.csv")

dataset = dataset.drop('Pregnancies', axis='columns') #Menghapus kolom pada Pregnancies drop dikarenakan banyak pasien yang laki-laki
dataset = dataset.drop('DiabetesPedigreeFunction', axis='columns') #menghapus kolom pada DiabetesPedigreeFunction dengan drop

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


x = dataset[['BMI','Glucose','BloodPressure','SkinThickness','Insulin','Age']]# 6 atribut yang digunakan untuk perbandingan
y = dataset['Outcome']#sebagai target

# melakukan split dataset training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# membuat KNN Classifier
knn = KNeighborsClassifier(n_neighbors=9)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(X_test)

# Model Accuracy, how ofter is the classifier correct?
print("Accuracy:" , knn.score(X_test, y_test))

lookup_diabetes_name = dict(zip(dataset.Outcome.unique(), dataset.Lable.unique()))   
print(lookup_diabetes_name)

st.write("Prediksi Diabetes")
nama=st.text_input("Masukan Nama Anda:")
BMI=st.number_input("masukan BMI(Body Mask Index):")
Glucose=st.number_input("masukan Glucose:")
BloodPressure=st.number_input("masukan BloodPressure:")
SkinThickness=st.number_input("masukan SkinThickness(4.90-21.00 normal):")
Insulin=st.number_input("masukan Insulin:")
Age=st.number_input("masukan Age:")

Test=st.button("Cek Diabetes")
deases_prediction = knn.predict([[BMI,Glucose, BloodPressure,SkinThickness,Insulin,Age]])
prediksi=lookup_diabetes_name[deases_prediction[0]]
if Test : 
    st.success(f"{nama} anda terbukti {prediksi}")






import warnings
warnings.filterwarnings('ignore', message="X does not have valid feature names, but KNeighborsClassifier was fitted with feature names", category=UserWarning)
