import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import *

# load model object 
api = joblib.load('./model/PAagent.pkl')

# build model_dict
model_dict = {}
model_dict['V1'] = api

#title
st.title('PA control advice model')

# user select model
model_name = st.selectbox(
	'what model you want to use?',
	('V1',))
st.write('You selected',model_name)

# user input
s = st.number_input('input set point(0997)')
d = st.number_input('input DATA OF USE')
st.write('The user input is ',s,d)

# user press predict button
if st.button('predict'):
	model = model_dict[model_name]
	advice,value = model.get_advice(s,d)
	print(value)
	st.subheader('control advice')
	st.write(advice)
	st.subheader('predict output')
	st.write(pd.DataFrame([value]))
