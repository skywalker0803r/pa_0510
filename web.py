import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import *

model_dict = {}
model_dict['V1'] = joblib.load('./model/PAagent.pkl')

st.title('PA control advice model')

model_name = st.selectbox('what model you want to use?',('V1',))
st.write('You selected',model_name)

request = st.number_input('input set point(0997)')
state = st.number_input('input DATA OF USE')
st.write('The user request {} state {}'.format(request,state))

if st.button('predict'):
	advice,output,stream = model_dict[model_name].get_advice(state,request)
	st.subheader('control advice')
	st.write(advice)
	st.subheader('predict output')
	st.write(pd.DataFrame(output))
	st.subheader('predict stream')
	st.write(pd.DataFrame(stream))
