# ======================================================================================== Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import time
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
        # Load data
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
    st.markdown(
        """
            # Select features
        """
    )
    st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
    
    
    data_columns = data.columns.values.tolist()
    data_features_train = [False for _ in range(len(data.columns.values.tolist()))]
    
    st.markdown(
            """
                ## Select features training
            """
    )

    check_feature = data_columns
    if data_columns:
        feature_select = []
        for choice in st.session_state.keys():
            if choice.startswith('dynamic_checkbox_') and st.session_state[choice]:
                feature_select.append(choice.replace('dynamic_checkbox_',''))
    col1, col2, col3 = st.columns([1, 1, 10])
    with col1: 
        if st.button('Select All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = True
            st.experimental_rerun()
    with col2:
        if st.button('Unselect All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = False
            st.experimental_rerun()
    with col3:
        st.write("")    
    cols = st.columns(5)
    index = 0
    for i in data:
        with cols[index % 5]:
            st.checkbox(i, key='dynamic_checkbox_' + i)
        index += 1

    features_train = data.loc[:, feature_select]
    features_not_train = data.loc[:, ~data.columns.isin(feature_select)]
    # ============================================ Feature Selection Test ============================================
    st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 7])
    with col1:
        col1.markdown(
                """
                    ## Select features predict
                """
        )
        
        selectbox_label = st.radio(
            "",
            tuple(features_not_train)
        )
        features_label = selectbox_label
    with col2:
        col2.markdown(
            """
                ## Data counts and statistics
            """
        )
        if selectbox_label != None:
            df1 = data[selectbox_label].value_counts()
            st.bar_chart(df1, use_container_width=True)
            st.dataframe(data.loc[:, features_label].to_frame().style.background_gradient(cmap='bone'), use_container_width=True)
    
    # ============================================ Train model ============================================
    st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
    if selectbox_label != None:
        st.markdown(
            """
                # One-hot Encoder
            """
        )
        
        st.write(dict(zip(data[selectbox_label].cat.codes, data[selectbox_label])))
        if data[selectbox_label].dtypes == 'object':   
            data[selectbox_label] = data[selectbox_label].astype('category').cat.codes
            
        feature_select = [feature for feature in feature_select if len(data[feature].unique()) <= len(data) * 0.75]
        data = data.loc[:, feature_select + [features_label]]
        feature_select_obj = [feature for feature in feature_select if data[feature].dtypes == 'object']
        f_data = pd.get_dummies(
            data = data,
            columns = feature_select_obj,
            prefix = "f"
        )
        
        # Get X, y
        X = f_data[f_data.columns.difference([selectbox_label])].to_numpy()
        y = f_data[selectbox_label].to_numpy()
    
    # ============================================ Model selection ============================================
        st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
        st.markdown(
            """
                # Model selection
            """
        )
        
        col1, col2, col3 = st.columns([3, 1, 6])
        with col1:
            button_train_test_split = st.selectbox(
                "",
                ('Train test split', 'K-Fold'),
                key=100
            )
        with col2:
            st.write("")
        with col3:
            if button_train_test_split == 'Train test split':
                slider_train_test_split = st.slider('Training rate', 0.0, 0.99, 0.8)
                st.markdown(f"Train size accounts for {slider_train_test_split * 100}% dataset")
                input_random_state = st.number_input('Random state', min_value=0)
                st.markdown(f"Random state equal {input_random_state}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=slider_train_test_split, random_state=input_random_state)
            else:
                split_value = st.number_input("", min_value=2)
                st.write('You selected:', split_value)
        
        # ============================================ Model ============================================
        st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
        st.markdown(
            """
                # Model
            """
        )
        st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([4, 1, 5])
        with col1:
            option_model = st.selectbox(
                'How would you like to be selected?',
                ('Logistic Regression', 'Decision Tree Classifier', 'KNeighbors Classifier', 'GaussianNB', 'Random Forest Classifier')
            )
                    
            button_train = st.button("Run")
            if button_train:
                y_train = y_train.reshape(-1)
                y_test = y_test.reshape(-1)
                
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                
                model = None
                if option_model == 'Logistic Regression':
                    model = LogisticRegression()
                elif option_model == 'Decision Tree Classifier':
                    model = DecisionTreeClassifier()
                elif option_model == 'KNeighbors Classifier':
                    model = KNeighborsClassifier()
                elif option_model == 'GaussianNB':
                    model = GaussianNB()
                else:
                    model = RandomForestClassifier()
                hist = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.text_input(label="", value=accuracy_score(y_pred, y_test))
        with col2:
            st.write("")
        with col3:
            pred_df = pd.DataFrame({'Actual Value': y_test,'Predicted Value': y_pred,'Difference': y_test != y_pred})
            st.dataframe(pred_df, use_container_width=True)