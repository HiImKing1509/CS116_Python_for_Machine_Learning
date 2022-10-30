# ======================================================================================== Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import time

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Style
from streamlit_style import styles, test_result
from streamlit_resources import plot_kfold

# ======================================================================================== Application

st.set_page_config(layout="wide")
st.markdown(styles.streamlit_style, unsafe_allow_html=True)

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
    st.markdown(
        """
            ### About this file
        """
    )
    st.text("This dataset has data collected from New York, California and Florida about 50 business Startups "'17 in each state'". The variables used in the dataset are Profit, R&D spending, Administration Spending, and Marketing Spending.")
    st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
    is_loaded = True
    bytes_data = uploaded_file.getvalue()
    img_path = './' + uploaded_file.name
    text_load_dataset_success = f"""
        <style>
            p.uploaded_file {{
                    color: Green;
                    font-size: 16px;
                    font-weight: 900;
                }}
        </style> 
        <p class="uploaded_file">{uploaded_file.name} is uploaded</p>
    """
    
    # =============== Read data ==============
    data = pd.read_csv(uploaded_file)
    dataset = data.copy()
    # ========================================
    
    st.markdown(text_load_dataset_success, unsafe_allow_html=True)
    dataset_style = dataset.copy()
    dataset_style = dataset_style.style.background_gradient(cmap='Blues')
    st.dataframe(dataset_style, height=600, use_container_width=True)

# ============================================ Feature Selection ============================================
    st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
    st.markdown(
        """
            ## Select feature
        """
    )

    features = []
    col1, col2 = st.columns([1, 19])
    with col1: feature_1 = st.checkbox("", key=1)
    with col2: 
        if feature_1: 
            st.markdown(styles.selected_style_checked("R&D Spend"), unsafe_allow_html=True)
            features.append(0)
        else: st.markdown(styles.selected_style_unchecked("R&D Spend"), unsafe_allow_html=True)

    col1, col2 = st.columns([1, 19])
    with col1: feature_2 = st.checkbox("", key=2)
    with col2: 
        if feature_2: 
            st.markdown(styles.selected_style_checked("Administration"), unsafe_allow_html=True)
            features.append(1)
        else: st.markdown(styles.selected_style_unchecked("Administration"), unsafe_allow_html=True)

    col1, col2 = st.columns([1, 19])
    with col1: feature_3 = st.checkbox("", key=3)
    with col2: 
        if feature_3: 
            st.markdown(styles.selected_style_checked("Marketing Spend"), unsafe_allow_html=True)
            features.append(2)
        else: st.markdown(styles.selected_style_unchecked("Marketing Spend"), unsafe_allow_html=True)

    col1, col2 = st.columns([1, 19])
    with col1: feature_4 = st.checkbox("", key=4)
    with col2: 
        if feature_4: 
            st.markdown(styles.selected_style_checked("State"), unsafe_allow_html=True)
            features.append(3)
        else: st.markdown(styles.selected_style_unchecked("State"), unsafe_allow_html=True)

    # =============== Get train and test data ==============
    st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
    
    train_features = data.iloc[:, features]
    test_label = data.iloc[:, -1].to_frame()
    col1, col2, col3 = st.columns([7, 1, 2])
    with col1:
        col1.markdown(
                """
                    ### Data train
                """, unsafe_allow_html=True)
        col1.dataframe(train_features.style.background_gradient(cmap='Blues'), use_container_width=True)
    with col2: st.write(' ')
    with col3:
        col3.markdown(
                """
                    ### Data test
                """, unsafe_allow_html=True)
        col3.dataframe(test_label.style.background_gradient(cmap='Blues'), use_container_width=True)
        
    # ======================================================

    # ============================================ Model visualization ========================================
    st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
    st.markdown(
        """
            ## Visualization
        """
    )

    # =============== Data Visualization ===============
    col1, col2, col3 = st.columns([3, 1, 6])
    with col1: 
        button_visualization = st.selectbox(
            "",
            ('Pairlot dataset', 'Heatmap dataset', 'Outliers detection in the target variable')
        )
    with col2:
        st.write(' ')
    with col3: 
        if button_visualization == "Pairlot dataset":
            fig = sns.pairplot(dataset)
            st.pyplot(fig, caption='Visualize pairlot dataset', use_column_width=True)
            plt.close()
        elif button_visualization == "Heatmap dataset":
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), annot=True, cmap='Blues')
            st.write(fig, unsafe_allow_html=True)
            plt.close()
        else:
            fig, ax = plt.subplots()
            outliers = ['Profit']
            plt.rcParams['figure.figsize'] = [8,8]
            sns.boxplot(data=dataset[outliers], orient="v", palette="Set2" , width=0.7)
            
            plt.title("Outliers Variable Distribution")
            plt.ylabel("Profit Range")
            plt.xlabel("Continuous Variable")
            st.write(fig, unsafe_allow_html=True)
            plt.close()

    # ============================================ Data preprocessing ============================================
    st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
    if len(features) == 0:
        st.text('No features is selected')
    else:
        st.markdown(
            """
                ## Data preprocessing
            """
        )
        
        st.text("For label encoding, we need to import LabelEncoder as shown below. Then we create an object of this class that is used to call fit_transform() method to encode the state column of the given datasets.")
        flag_encoder = False
        
        if 3 in features:
            lb = LabelEncoder()
            data['State'] = data['State'].astype('category').cat.codes
            st.dataframe(data.iloc[:, features].style.background_gradient(cmap='Purples'), use_container_width=True)
            flag_encoder = True

        # ============================================ Model selection ============================================
        st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
        st.markdown(
            """
                ## Model selection
            """
        )
        dt = data.to_numpy()
        X = dt[:, features]
        y = dt[:, -1]
        
        col1, col2 = st.columns([3, 17])
        with col1:
            button_train_test_split = st.radio(
                "",
                ('Train test split', 'K-Fold')
            )
        with col2:
            if button_train_test_split == 'Train test split':
                slider_train_test_split = st.slider('Training rate', 0.0, 0.99, 0.8)
                st.markdown(f"Train size accounts for {slider_train_test_split * 100}% dataset")
                input_random_state = st.number_input('Random state', min_value=0)
                st.markdown(f"Random state equal {input_random_state}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=slider_train_test_split, random_state=input_random_state)
            else:
                split_value = st.selectbox(
                    'How many fold splitted',
                    (2, 3, 4, 5, 6, 7, 8, 9, 10)
                )
                st.write('You selected:', split_value)
        

        # ============================================ Model ============================================
        st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
        st.markdown(
            """
                ## Model
            """
        )
        st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([4, 1, 5])
        # ==================== Model ==========================
        with col1:
            st.markdown(
                """
                    ### Train model
                """
            )
            button_train = st.checkbox("Run")
            arr_score_train_test_split = None
            df_train_test_split = None
            if button_train:
                model = None 
                if button_train_test_split == 'Train test split':
                    my_bar = st.progress(0)
                    model = LinearRegression()
                    hist = model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    tt_mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    tt_r2_score = r2_score(y_test, y_pred) * 100
                    tt_mae = mean_absolute_error(y_test, y_pred)
                    for percent_complete in range(0, 100, 20):
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 20)
                else:
                    kf_r2_score = 0
                    kf_mse = 0
                    kf_mae = 0
                    i = 0
                    
                    kf_r2_arr = []
                    kf_mse_arr = []
                    kf_mae_arr = []
                    fold = []
                    
                    kf = KFold(n_splits = int(split_value))
                    print(type(X))
                    for train_index, test_index in kf.split(X):
                        my_bar = st.progress(0)
                        # Split data
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        # k-fold train model
                        model = LinearRegression()
                        hist = model.fit(X_train, y_train)
                        
                        # k-fold test model
                        y_pred = model.predict(X_test)
                        
                        # r2_score
                        r2 = r2_score(y_test, y_pred)
                        kf_r2_arr.append(r2)
                        
                        # mse score
                        mse = np.sqrt(mean_squared_error(y_test, y_pred))
                        kf_mse_arr.append(mse)
                        
                        # mae score
                        mae = mean_absolute_error(y_test, y_pred)
                        kf_mae_arr.append(mae)
                        
                        st.text(f"{i + 1} Fold successfully")
                        
                        for percent_complete in range(0, 100, 20):
                            time.sleep(0.1)
                            my_bar.progress(percent_complete + 20)
                        
                        fold.append(f"Fold {i}")
                        i += 1
                train_success = f"""
                        <style>
                            p.uploaded_file {{
                                    color: Green;
                                    fon-size: 16px;
                                    font-weight: 900;
                                }}
                        </style> 
                        <p class="uploaded_file">Model has been trained successfully</p>
                    """
                st.markdown(train_success, unsafe_allow_html=True)
        # ============================================ Model predicted ============================================
                with col2:
                    st.write("")
                with col3:
                    st.markdown(
                        """
                            ### Result in test set
                        """
                    )
                    
                    test_success = f"""
                            <style>
                                p.uploaded_file {{
                                        color: Green;
                                        font-size: 16px;
                                        font-weight: 900;
                                    }}
                            </style> 
                            <p class="uploaded_file">Model has been tested successfully</p>
                        """
                    st.markdown(test_success, unsafe_allow_html=True)
                        
                    if button_train_test_split == 'Train test split':
                        st.write(test_result("Mean square error", tt_mse), unsafe_allow_html=True)
                        st.write(test_result("R2 Score", tt_r2_score), unsafe_allow_html=True)
                        st.write(test_result("Mean Absolute Error", tt_mae), unsafe_allow_html=True)
                    else:
                        final_kf_mse = sum(kf_mse_arr) / len(kf_mse_arr)
                        final_kf_r2_score = sum(kf_r2_arr) / len(kf_r2_arr)
                        final_kf_mae = sum(kf_mae_arr) / len(kf_mae_arr)
                        st.write(test_result("Mean square error", final_kf_mse), unsafe_allow_html=True)
                        st.write(test_result("R2 Score", final_kf_r2_score), unsafe_allow_html=True)
                        st.write(test_result("Mean Absolute Error", final_kf_mae), unsafe_allow_html=True)
                        
                        tab1, tab2, tab3 = st.tabs(["MSE", "R2 Score", "MAE"])
                        with tab1:
                            st.header("MSE")
                            st.pyplot(plot_kfold(kf_mse_arr, fold, 'r'))
                        with tab2:
                            st.header("R2 Score")
                            st.pyplot(plot_kfold(kf_r2_arr, fold, 'b'))
                        with tab3:
                            st.header("MAE")
                            st.pyplot(plot_kfold(kf_mae_arr, fold, 'g'))


                st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
                st.markdown(
                    """
                        ### Predicted Values
                    """
                )
                pred_df = pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
                st.dataframe(pred_df, use_container_width=True)
                
                st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
                st.markdown(
                    """
                        ### Visualize predicted and actually values
                    """
                )
                

        # ============================================ Test on real data ============================================
                st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
                st.markdown(
                    """
                        ## Test on real data
                    """
                )
                
                text_0 = ''
                text_1 = ''
                text_2 = ''
                
                predict = []
                if 0 in features:
                    text_0 = st.text_input(f"Enter {data.columns[0]}", placeholder=data.columns[0])
                if 1 in features:
                    text_1 = st.text_input(f"Enter {data.columns[1]}", placeholder=data.columns[1])
                if 2 in features:
                    text_2 = st.text_input(f"Enter {data.columns[2]}", placeholder=data.columns[2])
                if 3 in features:
                    enter_state = st.selectbox(
                        f'Enter {data.columns[3]}',
                        ("None", "New York", "California", "Florida")
                    )
                
                if 0 in features and text_0 != '':
                    predict.append(text_0)
                if 1 in features and text_1 != '':
                    predict.append(text_1)
                if 2 in features and text_2 != '':
                    predict.append(text_2)
                if 3 in features:
                    if enter_state == "New York":
                        predict.append(2)
                    elif enter_state == "California":
                        predict.append(0)
                    elif enter_state == "Florida":
                        predict.append(1)
                relity_data = st.button("Run")
                if relity_data:
                    if len(predict) == len(features):
                        predict_arr = np.array(predict)                    
                        predict_arr = np.reshape(predict_arr, (1, -1))
                        y_pred_real = model.predict(predict_arr)
                        st.write(test_result("Profit", y_pred_real[0]), unsafe_allow_html=True)
                    else:
                        if 0 in features and text_0 == '':
                            st.write("Enter value for R&D Spend")
                        if 1 in features and text_1 == '':
                            st.write("Enter value for Administration")
                        if 2 in features and text_1 == '':
                            st.write("Enter value for Marketing Spend")
                        if 3 in features and enter_state == "None":
                            st.write("Select value for State")

