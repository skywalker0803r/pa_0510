import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import *

st.title('製程出料批次預測系統')

# 選擇模型
model_dict = {'V1':joblib.load('./model/PAagent.pkl')}
model_name = st.selectbox('what model you want to use?',('V1',))
st.write('You selected',model_name)
model = model_dict[model_name]

# 上傳檔案
uploaded_file = st.file_uploader('上傳檔案',type="xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file,index_col=0).dropna(axis=1)
    action = df[model.action_col]
    state = df[['DATA OF USE','MLPAP_TI3-1110.PV']]
    st.title('action')
    st.write(action)
    st.title('state')
    st.write(state)

# 預測
if st.button('預測'):
    st.write('預測中請稍後')
    result_df = pd.DataFrame(index=state.index,columns=['出料','出料','單耗','蒸氣單耗'])
    for i,idx in enumerate(state.index):
        result_df.loc[idx,:] = model.get_predict(state.loc[idx].values.tolist(),action.loc[[idx]])
    st.title('預測結果')
    st.write(result_df)
    st.write('預測結果以保存至data/result.xlsx')
    result_df.to_excel('data/result.xlsx')    