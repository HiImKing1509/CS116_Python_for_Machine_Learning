# ======================================================================================== Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import time
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from styles import styles, text_success

# ===== function =====

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

# session_state
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
    st.session_state['model'] = None
    # st.session_state['score'] = [0, 0]
    # st.session_state['cfs_matrix'] = None

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
    data_ = data.copy()
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

    if len(feature_select) == 0:
        st.info("Select features to continue")
        st.stop()
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
            tuple(features_not_train),
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
            print(df1)
            print(type(df1))
            st.bar_chart(df1, use_container_width=True)
            st.dataframe(data.loc[:, features_label].to_frame().style.background_gradient(cmap='bone'), use_container_width=True)
            
    # ============================================ Train model ============================================
    if selectbox_label != None:
        
        flag_label_is_obj = False
        
        # if data[selectbox_label].dtypes == 'object':
        #     flag_label_is_obj = True
        #     y_original = data[selectbox_label].copy()
        #     data[selectbox_label] = data[selectbox_label].astype('category').cat.codes
        #     mapping_label = {}
        #     for i in range (0, len(y_original)):
        #         mapping_label.update({data[selectbox_label][i].tolist() : y_original[i]})
        
        # feature_select = [feature for feature in feature_select if len(data[feature].unique()) <= len(data) * 0.75]
        data = data.loc[:, feature_select + [features_label]]
        feature_select_obj = [feature for feature in feature_select if data[feature].dtypes == 'object']
        f_data = pd.get_dummies(
            data = data,
            columns = feature_select_obj,
            prefix = "f"
        )
        
        # Get X, y
        
        XX = f_data[f_data.columns.difference([selectbox_label])]
        yy = f_data[selectbox_label]
        
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
            )
        with col2:
            st.write("")
        with col3:
            if button_train_test_split == 'Train test split':
                slider_train_test_split = st.slider('Training rate', 0.0, 0.99, 0.8)
                st.markdown(f"Train size accounts for {slider_train_test_split * 100}% dataset")
                input_random_state = st.number_input('Random state', min_value=0)
                st.markdown(f"Random state equal {input_random_state}")
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
                ('Logistic Regression', 'Decision Tree Classifier', 'KNeighbors Classifier', 'GaussianNB', 'Random Forest Classifier', 'XGBoost Classifier')
            )
                    
            model = None
            train = False
            if st.button("Run"):
                train = True
                st.session_state['trained'] = True
                
                if option_model == 'Logistic Regression':
                    model = LogisticRegression()
                elif option_model == 'Decision Tree Classifier':
                    model = DecisionTreeClassifier()
                elif option_model == 'KNeighbors Classifier':
                    model = KNeighborsClassifier()
                elif option_model == 'GaussianNB':
                    model = GaussianNB()
                elif option_model == 'Random Forest Classifier':
                    model = RandomForestClassifier()
                else:
                    model = xgb.XGBClassifier()
                    
                y_pred_max = None,
                y_test_max = None
                if button_train_test_split == "Train test split":
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=slider_train_test_split, random_state=input_random_state)
                    y_train = y_train.reshape(-1)
                    y_test = y_test.reshape(-1)
                    
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)
                    hist = model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc_score = accuracy_score(y_pred, y_test)
                    y_pred_max = y_pred.copy()
                    y_test_max = y_test.copy()
                else:
                    i = 0
                    fold = []
                    acc_score = 0
                    
                    kf = KFold(n_splits = int(split_value))
                    for train_index, test_index in kf.split(X):
                        my_bar = st.progress(0)
                        # Split data
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        # k-fold train model
                        hist = model.fit(X_train, y_train)
                        
                        # k-fold test model
                        y_pred = model.predict(X_test)
                        
                        acc_score_pred = accuracy_score(y_pred, y_test)
                        if acc_score_pred > acc_score:
                            acc_score = acc_score_pred
                            y_pred_max = y_pred
                            y_test_max = y_test
                            
                        
                        st.text(f"{i+1} Fold successfully")
                        
                        for percent_complete in range(0, 100, 20):
                            time.sleep(0.1)
                            my_bar.progress(percent_complete + 20)
                        
                        fold.append(f"Fold {i}")
                        i += 1
                
                st.session_state['model'] = model
                
                ps = precision_score(y_test_max, y_pred_max, average='weighted')
                rs = recall_score(y_test_max, y_pred_max, average='weighted')
                fs = f1_score(y_test_max, y_pred_max, average='weighted')
                st.markdown(
                    f"""
                        ### Accuracy score: `{acc_score}`\n
                        ### Precision score: `{ps}`\n
                        ### Recall score: `{rs}`\n
                        ### F1 score: `{fs}`\n
                    """
                )
                st.bar_chart(pd.Series([ps, rs, fs], index =['Precision score', 'Recall score', 'F1 score']), use_container_width=True) 
                
                with col3:
                    pred_df = pd.DataFrame({'Actual Value': y_test_max, 'Predicted Value': y_pred_max, 'Same': y_test_max == y_pred_max})
                    # if flag_label_is_obj == True:
                    #     for i in range(0, len(y_pred)):
                    #         pred_df["Actual Value"][i] = mapping_label[i] 
                    #         pred_df["Predicted Value"][i] = mapping_label[i]
                    st.dataframe(pred_df, use_container_width=True)
                    count = [pred_df["Actual Value"][i] == pred_df["Predicted Value"][i] for i in range(0, len(pred_df))].count(True)
                    st.markdown(
                            f"""
                                ### `{count} / {len(pred_df)}` samples are predicted exactly
                            """
                    )
        if not st.session_state['trained']:
            st.info("Train model to continue")
            st.stop()
        with col2:
            st.write("")
        # ============================================ Test on real data ============================================
        st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
        st.markdown(
            """
                ## Test on real data
            """
        )
              
        feature_real = []
        
        cols = st.columns(3)
        
        index = 0
        for i in features_train.columns:
            with cols[index % 3]:
                if data_[i].dtype == 'object':
                    enter_state = st.selectbox(
                        f'Enter {i}',
                        tuple(data_[i].unique().tolist())
                    )
                elif data_[i].dtype == 'int64':
                    text_input = st.number_input(f"Enter {i}", min_value=0)
                else:
                    text_input = st.text_input(f"Enter {i}", placeholder=i)
            
                if data_[i].dtype != 'object' and text_input != '':
                    feature_real.append(text_input)
                if data_[i].dtype == 'object':
                    
                    feature_real.append(enter_state)
            index += 1
                       
        if st.button("Make Predict"):
            try:
                model = st.session_state['model']
                predict_arr = np.array(feature_real)
                predict_arr = np.reshape(predict_arr, (1, -1))
                predict_arr_ = np.asarray(predict_arr, dtype=float)
                y_pred_real = model.predict(predict_arr)
                st.markdown(
                    f"""
                        ### Predict: `{y_pred_real[0]}`
                    """
                )
            except Exception as e:
                st.write(e)
        else:
            st.info("Error!!!")
else:
    st.info("Upload dataset to continue")