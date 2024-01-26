import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
# import warnings
# warnings.filterwarnings("ignore")

loaded_dt_model = pickle.load(open('C:/Users/emmry/Desktop/My Project Web/trained_dt_model.sav', 'rb'))
loaded_rf_model = pickle.load(open('C:/Users/emmry/Desktop/My Project Web/trained_rf_model.sav', 'rb'))
loaded_lr_model = pickle.load(open('C:/Users/emmry/Desktop/My Project Web/trained_lr_model.sav', 'rb'))
loaded_mlp_model = pickle.load(open('C:/Users/emmry/Desktop/My Project Web/trained_mlp_model.sav', 'rb'))

st.set_page_config(
   page_title="Crop Recommender System",
   page_icon="cloud-hail",
   layout="wide",
   initial_sidebar_state="expanded",)


with st.sidebar:
	selected = option_menu ('Models for recommendation',['Decision Tree Model', 'Random Forest Model', 'Logistic Regression Model',
	 'Multi-Layered Percepton'], icons=['cloud-hail', 'cloud-lightning-rain', 'cloud-hail', 'cloud-lightning-rain'], default_index = 0)
    
## Decision Tree
def dt_crop_recommendation(input_data):
	input_data_as_numpy_array = np.array(input_data)
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
	prediction = loaded_dt_model.predict(input_data_reshaped)
	return prediction

if (selected == 'Decision Tree Model'):

	st.title(':red[Crop Prediction System Using Decision Tree Model] :seedling:', )

	col1, col2, col3 = st.columns(3)
	with col1:
		N = st.text_input('Nitrogen Value')
	with col2:
		P = st.text_input('Phosphorus Value')
	with col3:
		K = st.text_input('Potassium Value')
	with col1:
		Temperature = st.text_input('Temperature Value')
	with col2:
		Humidity = st.text_input('Humidity Value')
	with col3:
		Ph = st.text_input('Soil Ph Value')
	with col1:
		Rainfall = st.text_input('Rainfall Value')	

	recommend = ''

	if st.button('Recommend Crop'):
		recommend = dt_crop_recommendation([N,P,K,Temperature,Humidity,Ph,Rainfall])

	st.success(recommend)

##Random Forest
def rf_crop_recommendation(input_data):
	input_data_as_numpy_array = np.array(input_data)
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
	prediction = loaded_rf_model.predict(input_data_reshaped)
	return prediction

if (selected == 'Random Forest Model'):

	st.title(':green[Crop Prediction System Using Random Forest Model] :shamrock:')

	col1, col2, col3, = st.columns(3)
	with col1:
		N = st.text_input('Nitrogen Value')
	with col2:
		P = st.text_input('Phosphorus Value')
	with col3:
		K = st.text_input('Potassium Value')
	with col1:
		Temperature = st.text_input('Temperature Value')
	with col2:
		Humidity = st.text_input('Humidity Value')
	with col3:
		Ph = st.text_input('Soil Ph Value')
	with col1:
		Rainfall = st.text_input('Rainfall Value')	

	recommend = ''

	if st.button('Recommend Crop'):
		recommend = rf_crop_recommendation([N,P,K,Temperature,Humidity,Ph,Rainfall])

	st.success(recommend)

##Logistic Regression
def lr_crop_recommendation(input_data):
	input_data_as_numpy_array = np.array(input_data)
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
	prediction = loaded_lr_model.predict(input_data_reshaped)
	return prediction

if (selected == 'Logistic Regression Model'):

	st.title(':blue[Crop Prediction System Using Logistic Regression Model] :cactus:')

	col1, col2, col3, = st.columns(3)
	with col1:
		N = st.text_input('Nitrogen Value')
	with col2:
		P = st.text_input('Phosphorus Value')
	with col3:
		K = st.text_input('Potassium Value')
	with col1:
		Temperature = st.text_input('Temperature Value')
	with col2:
		Humidity = st.text_input('Humidity Value')
	with col3:
		Ph = st.text_input('Soil Ph Value')
	with col1:
		Rainfall = st.text_input('Rainfall Value')	

	recommend = ''

	if st.button('Recommend Crop'):
		recommend = lr_crop_recommendation([N,P,K,Temperature,Humidity,Ph,Rainfall])

	st.success(recommend)

##Multi_Layered Percepton
def mlp_crop_recommendation(input_data):
	input_data_as_numpy_array = np.array(input_data)
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
	prediction = loaded_mlp_model.predict(input_data_reshaped)
	return prediction

if (selected == 'Multi-Layered Percepton'):

	st.title(':red[Crop Prediction System Using Multi-Layered Percepton Model] :herb:')

	col1, col2, col3, = st.columns(3)
	with col1:
		N = st.text_input('Nitrogen Value')
	with col2:
		P = st.text_input('Phosphorus Value')
	with col3:
		K = st.text_input('Potassium Value')
	with col1:
		Temperature = st.text_input('Temperature Value')
	with col2: 
		Humidity = st.text_input('Humidity Value')
	with col3:
		Ph = st.text_input('Soil Ph Value')
	with col1:
		Rainfall = st.text_input('Rainfall Value')	

	recommend = ''

	if st.button('Recommend Crop'):
		recommend = mlp_crop_recommendation([N,P,K,Temperature,Humidity,Ph,Rainfall])

	st.success(recommend)
