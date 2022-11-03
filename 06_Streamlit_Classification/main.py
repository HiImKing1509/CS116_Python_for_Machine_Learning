# ======================================================================================== Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import time

from styles import styles, text_success

# ===== function =====
def checkbox_container(data):
    cols = st.columns(5)
    if cols[0].button('Select All'):
        for i in data:
            st.session_state['dynamic_checkbox_' + i] = True
        st.experimental_rerun()
    if cols[1].button('UnSelect All'):
        for i in data:
            st.session_state['dynamic_checkbox_' + i] = False
        st.experimental_rerun()
    data_cols = st.columns(len(data))
    index = 0
    for i in data:
        with data_cols[index % 5]:
            st.checkbox(i, key='dynamic_checkbox_' + i)
            index += 1

# ======================================================================================== Application

st.set_page_config(layout="wide")
# st.markdown(styles.streamlit_style, unsafe_allow_html=True)

# ============================================ Header introduce ============================================
st.markdown(
    """
        # Name: Huynh Viet Tuan Kiet
        # ID: 20521494
    """
)

# ============================================ Load data ============================================
st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
st.markdown(
    """
        ## Load data
    """
)
uploaded_file = st.file_uploader("Upload dataset")
if uploaded_file is not None:
    st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
    is_loaded = True
    bytes_data = uploaded_file.getvalue()
    img_path = './' + uploaded_file.name
    # =============== Read data ==============
    data = pd.read_csv(uploaded_file)
    st.markdown(text_success(uploaded_file.name), unsafe_allow_html=True)
    st.dataframe(data, height=600, use_container_width=True)
    
    # ============================================ Feature Selection Train ============================================
    st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
    
    data_columns = data.columns.values.tolist()
    data_features_train = [False for _ in range(len(data.columns.values.tolist()))]
    
    features_train = []
    col1, col2 = st.columns(2)
    col1.markdown(
            """
                ## Select features training
            """
    )
    
    with col1:
        check_feature = data_columns
        if data_columns:
            feature_select = []
            for choice in st.session_state.keys():
                if choice.startswith('dynamic_checkbox_') and st.session_state[choice]:
                    feature_select.append(choice.replace('dynamic_checkbox_',''))
        
        if st.button('Select All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = True
            st.experimental_rerun()
        if st.button('UnSelect All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = False
            st.experimental_rerun()
        
        for i in data:
            st.checkbox(i, key='dynamic_checkbox_' + i)
        feature_choose = data.loc[:, feature_select]
        feature_not_choose = data.loc[:, ~data.columns.isin(feature_select)]
        
    # for i in range(0, len(data_columns)):
    #     with col1: 
    #         checkbox_feature = st.checkbox(f"{data_columns[i]}", key=i)
    #         if checkbox_feature:
    #             features_train.append(i)
    #             data_features_train[i] = True
    #         else: 
    #             data_features_train[i] = False
    # ============================================ Feature Selection Test ============================================
    col2.markdown(
            """
                ## Select features predict
            """
    )
    with col2:    
        data_features_label = []
        
        for i in range(len(data.columns.values.tolist())):
            if data_features_train[i] == False:
                data_features_label.append(data_columns[i])
                
        features_label = None
        
        selectbox_label = st.selectbox(
            "",
            tuple(data_features_label)
        )
        if selectbox_label:
            features_label = data_columns.index(selectbox_label)
    
    
    
    # ============================================ Train model ============================================
    if len(features_train) != 0 and features_label != None:
        # =============== Get train and test data ==============
        st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
        
        train_features = data.iloc[:, features_train]
        test_label = data.iloc[:, features_label].to_frame()
        col1, col2, col3 = st.columns([7, 1, 2])
        with col1:
            col1.markdown(
                    """
                        ### Data train
                    """, unsafe_allow_html=True)
            col1.dataframe(train_features, use_container_width=True)
        with col2: st.write(' ')
        with col3:
            col3.markdown(
                    """
                        ### Data test
                    """, unsafe_allow_html=True)
            col3.dataframe(test_label, use_container_width=True)
            
        