import numpy as np
import pandas as pd
import streamlit as st
import time

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import min_values
from k_fold import k_fold_evaluation

from styles import styles, text_success

st.set_page_config(layout="wide", page_title='Wine PCA')
# st.markdown(styles.streamlit_style, unsafe_allow_html=True)

# ============================================ Header introduce ============================================
st.markdown(
    """
        # Name: Huynh Viet Tuan Kiet
        # ID: 20521494
    """
)

if 'model' not in st.session_state:
    st.session_state['trained'] = False
    st.session_state['model'] = None

# ============================================ Load data ============================================
st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
st.markdown("""# Load data""")

dataset = load_wine()
X = dataset['data']
y = dataset['target']

df = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']], columns= dataset['feature_names'] + ['customer_segment'])
st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
st.dataframe(df, height=600, use_container_width=True)

# ============================================ Load data ============================================
st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
col1, col2, col3 = st.columns([4, 1, 5])
with col1:
    st.markdown("""# Data Normalization""")
    
    selectbox_normalization = st.selectbox(
        'Select preprocessing data method',
        ('None', 'MinMaxScaler', 'StandardScaler')
    )
    
    if selectbox_normalization == 'MinMaxScaler':
        X_scale = MinMaxScaler().fit_transform(X)
    elif selectbox_normalization == 'StandardScaler':
        X_scale = StandardScaler().fit_transform(X)
    else:
        X_scale = X
            
with col2: st.write("")
with col3:
    st.markdown("""# Feature Dimensionality Reduction""")
    slider_n_components = st.slider('Select the number of features', 1, min_values(dataset), max(int(min_values(dataset) / 2), 1))
    pca = PCA(n_components=slider_n_components)
    X_pca = pca.fit_transform(X_scale)

st.write(X.shape)
st.write(X_pca.shape)


# ============================================ Model Selection ============================================
st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
st.markdown("""# Model Selection""")
col1, col2, col3 = st.columns([3, 1, 6])
with col1:
    button_model_selection = st.selectbox(
        "",
        ('Train test split', 'K-Fold'),
    )
with col2:
    st.write("")
with col3:
    if button_model_selection == 'Train test split':
        slider_train_test_split = st.slider('Training rate', 0.0, 0.99, 0.8)
        st.markdown(f" ### Train size accounts for {slider_train_test_split * 100}% dataset")
        input_random_state = st.number_input('Random state', min_value=0)
        st.markdown(f" ### Random state equal {input_random_state}")
    else:
        split_value = st.number_input("", min_value=2)
        st.markdown(f""" ### You selected {split_value}""")

# ============================================ Train Model ============================================
st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
st.markdown("""# Model""")
col1, col2, col3 = st.columns([4, 1, 5])
with col1:
    option_model = st.selectbox(
        'How would you like to be selected?',
        ('Logistic Regression', 'Decision Tree Classifier', 'KNeighbors Classifier', 'GaussianNB', 'Random Forest Classifier')
    )
    
    if st.button("Train model"):
        st.session_state['trained'] = True
        # selected method
        if option_model == 'Logistic Regression':
            model = LogisticRegression(random_state=0)
        elif option_model == 'Decision Tree Classifier':
            model = DecisionTreeClassifier()
        elif option_model == 'KNeighbors Classifier':
            model = KNeighborsClassifier()
        elif option_model == 'GaussianNB':
            model = GaussianNB()
        else:
            model = RandomForestClassifier()
        
        # selected model
        if button_model_selection == "Train test split":
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, train_size=slider_train_test_split, random_state=input_random_state)
            hist = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_score = accuracy_score(y_pred, y_test)
            ps = precision_score(y_test, y_pred, average='weighted')
            rs = recall_score(y_test, y_pred, average='weighted')
            fs = f1_score(y_test, y_pred, average='weighted')
            
            st.markdown(
                f"""
                    ### Accuracy score: `{acc_score}`\n
                    ### Precision score: `{ps}`\n
                    ### Recall score: `{rs}`\n
                    ### F1 score: `{fs}`\n
                """
            )
            with col3:
                st.markdown("""### Data""")
                st.bar_chart(pd.Series([ps, rs, fs], index =['Precision score', 'Recall score', 'F1 score']), use_container_width=True) 
        else:
            # KFold
            
            i = 0
            acc_score_all = []
            ps_list = []
            rs_list = []
            fs_list = []
            
            kf = KFold(n_splits=int(split_value), random_state=None)

            for train_index , test_index in kf.split(X_pca):
                my_bar = st.progress(0)
                X_train , X_test = X_pca[train_index,:], X_pca[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc_score_all.append(accuracy_score(y_pred , y_test))                
                ps_list.append(precision_score(y_test, y_pred, average='weighted'))
                rs_list.append(recall_score(y_test, y_pred, average='weighted'))
                fs_list.append(f1_score(y_test, y_pred, average='weighted'))
                
                st.text(f"{i+1} Fold successfully")
                for percent_complete in range(0, 100, 20):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 20)
                i += 1
                
            with col3:
                st.markdown("""### Data""")
                tabs = st.tabs([f"Fold {i+1}" for i in range(0, int(split_value))])
                for i in range(0, int(split_value)):    
                    with tabs[i]:
                        st.bar_chart(pd.Series([ps_list[i], rs_list[i], fs_list[i]], index =['Precision score', 'Recall score', 'F1 score']), use_container_width=True)
                        st.markdown(
                            f"""
                                ##### Accuracy score: `{acc_score_all[i]}`\n
                                ##### Precision score: `{ps_list[i]}`\n
                                ##### Recall score: `{rs_list[i]}`\n
                                ##### F1 score: `{fs_list[i]}`\n
                            """
                        )
                st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
                st.markdown("""# K Fold evaluation / number of features""")
                list_features = [i for i in range(1, min_values(dataset))]
                f1_score_list = k_fold_evaluation(model, X_scale, y, int(split_value), list_features)
                chart_data = pd.DataFrame(f1_score_list)
                st.bar_chart(chart_data)
                st.markdown(f"""### Optimal dimensionality is `{f1_score_list.index(max(f1_score_list))}`""")
                
                                                    
            acc_score = sum(acc_score_all) / int(split_value)
            ps_mean = sum(ps_list) / int(split_value)
            rs_mean = sum(rs_list) / int(split_value)
            fs_mean = sum(fs_list) / int(split_value)
            
            st.markdown(
                f"""
                    ### Mean Accuracy score: `{acc_score}`\n
                    ### Mean Precision score: `{ps_mean}`\n
                    ### Mean Recall score: `{rs_mean}`\n
                    ### Mean F1 score: `{fs_mean}`\n
                """
            )
        with col2:
            st.write("")
        st.session_state['model'] = model
if not st.session_state['trained']:
    st.info("Train model to continue")
    st.stop()

# ============================================ Test on real data ============================================
st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
st.markdown(""" # Test data""")

predict_data = {}
cols = st.columns(3)
for (index, feature) in enumerate(dataset['feature_names']):            
    number = cols[index % 3].number_input(feature)
    predict_data[feature] = number
if (st.button("Predict")):
    predict = []
    for feature in dataset['feature_names']:
        predict.append(predict_data[feature])
    matrix = pca.transform(np.array([predict]))
    res = st.session_state['model'].predict(matrix)
    st.markdown(f""" #### Type of Customer: {res[0]}""")