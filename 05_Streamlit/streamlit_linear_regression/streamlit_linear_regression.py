# ======================================================================================== Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mpld3
import streamlit.components.v1 as components
from PIL import Image
import cv2

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Style
from streamlit_style import styles
from streamlit_resources import is_loaded, icon_check_success

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
    data = pd.read_csv(uploaded_file.name)
    dataset = data.copy()
    # ========================================
    
    col1, col2 = st.columns([1, 30])
    col1.image(icon_check_success, width=24)
    col2.markdown(text_load_dataset_success, unsafe_allow_html=True)
    # st.text(uploaded_file.name + ' is loaded')
    # st.write(bytes_data)
    # with open(img_path, 'wb') as f:
    #     f.write(bytes_data)
    dataset_style = dataset.copy()
    dataset_style = dataset_style.style.background_gradient(cmap='Purples_r')
    st.dataframe(dataset, height=600, use_container_width=True)

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
    test_label = data.iloc[:, -1]
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
        
    # ======================================================

    # ============================================ Model visualization ========================================
    st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
    st.markdown(
        """
            ## Visualization
        """
    )

    # =============== Data Visualization ===============
    sns.pairplot(dataset)
    plt.savefig('./images/pairlot')
    # ==================================================

    button_visualization = st.button("Pairlot dataset")

    if button_visualization:
        image_pairlot = Image.open('./images/pairlot.png')
        col1, col2, col3 = st.columns([1,6,1])
        with col1: st.write(' ')
        with col2: st.image(image_pairlot, caption='Visualize pairlot dataset')
        with col3: st.write(' ')

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
        
        flag_encoder = False
        
        if 3 in features:
            lb = LabelEncoder()
            data['State'] = data['State'].astype('category').cat.codes
            st.dataframe(data.iloc[:, features], use_container_width=True)
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
                st.write('You selected train test split')
                slider_train_test_split = st.slider('Training rate', 0.0, 0.99, 0.8)
                st.markdown(f"Train size accounts for {slider_train_test_split * 100}% dataset")
                input_random_state = st.number_input('Random state', min_value=0)
                st.markdown(f"Random state equal {input_random_state}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=slider_train_test_split, random_state=input_random_state)
            else:
                st.write("You selected K-Fold")
        

        # ============================================ Model ============================================
        st.markdown(styles.lines_section_separate_style, unsafe_allow_html=True)
        st.markdown(
            """
                ## Model
            """
        )
        st.markdown(styles.lines_separate_style, unsafe_allow_html=True)
        col1, col2 = st.columns([65, 35])
        # ==================== Model ==========================
        with col1:
            model = LinearRegression()
            hist = model.fit(X_train, y_train)
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
            st.image(icon_check_success, width=24)
            st.markdown(train_success, unsafe_allow_html=True)
        # ============================================ Model predicted ============================================
        with col2:
            y_pred = model.predict(X_test)
            test_success = f"""
                <style>
                    p.uploaded_file {{
                            color: Green;
                            font-size: 16px;
                            font-weight: 900;
                        }}
                </style> 
                <p class="uploaded_file">Model has been trained successfully</p>
            """
            st.image(icon_check_success, width=24)
            st.markdown(test_success, unsafe_allow_html=True)

        # # Step 6: Testing model

        # print(np.sqrt(mean_squared_error(y_test, y_pred)))
        # print(r2_score(y_test, y_pred) * 100)
        # print(mean_absolute_error(y_test, y_pred))

        # st.text_input("Result:", r2_score(y_test, y_pred) * 100)