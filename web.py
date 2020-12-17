import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import *

tag = pd.read_csv(r'./data/tag_cleaned.csv')[['TAG','chinese']]
tag = dict(zip(tag.TAG,tag.chinese))
model_dict = {}
model_dict['V1'] = joblib.load('./model/PAagent.pkl')
st.title('PA control advice model')
model_name = st.selectbox('what model you want to use?',('V1',))
st.write('You selected',model_name)
request = st.number_input('input set point(0997)')
state1 = st.number_input('input 觸媒使用時間最大1最小0')
state2 = st.number_input('input 環境溫度例如30')

#============給建議=========================================
if st.button('給操作建議'):
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
#===============預測出料======================================
st.title('預測出料')
s1 = st.number_input('input 觸媒使用時間最大1最小0.')
s2 = st.number_input('input 環境溫度例如30.')
s = [s1,s2]
model = model_dict[model_name]
action_df = pd.DataFrame(index=[0],columns=model.action_col)
for a in model.action_col:
    try:
        var = st.number_input('{}'.format(tag[a]))
    except:
        var = st.number_input('{}'.format(a))
    action_df[a] = var
if st.button('執行預測出料'):
    result = model.get_predict(s,action_df)
    output,stream,單耗,蒸氣單耗 = result
    st.subheader('predict output')
    st.write(output)
    st.subheader('predict stream')
    st.write(stream)
    st.subheader('predict 單耗(出料)')
    st.write(單耗)
    st.subheader('predict 單耗(蒸氣)')
    st.write(蒸氣單耗)
    