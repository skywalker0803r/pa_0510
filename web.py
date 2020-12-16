import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import *

def advice_input():
	request = st.number_input('input set point(0997)')
	state1 = st.number_input('input DATA OF USE')
	state2 = st.number_input('input env tempe')

def get_advice():
	state = [state1,state2]
	st.write('The user request {} state {}'.format(request,state))
	advice,output,stream,_,_ = model_dict[model_name].get_advice(state,request)
	st.subheader('control advice')
	st.write(advice)
	st.subheader('predict output')
	st.write(pd.DataFrame(output))
	st.subheader('predict stream')
	st.write(pd.DataFrame(stream))
	feed  = advice.iloc[0,1]
	st.subheader('predict 單耗(出料)')
	st.write(pd.DataFrame(feed/output))
	st.subheader('predict 單耗(蒸氣)')
	st.write(pd.DataFrame(feed/stream))

if __name__ == '__main__':
	model_dict = {}
	model_dict['V1'] = joblib.load('./model/PAagent.pkl')
	st.title('PA control advice model')
	model_name = st.selectbox('what model you want to use?',('V1',))
	st.write('You selected',model_name)

if st.button('predict_advice'):
	advice_input()


	

