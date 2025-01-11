import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

# Custom CSS for dark theme with modern UI elements
st.markdown(
    """
    <style>
        /* Dark Background and Fonts */
        .stApp {
            background-color: #121212; 
            color: #ffffff; 
            font-family: 'Roboto', sans-serif;
        }

        /* Headers */
        .stTitle {
            color: #ffffff;
            font-size: 32px;
            font-weight: bold;
            padding-bottom: 20px;
        }
        .stSubheader {
            color: #dddddd;
            font-size: 24px;
            font-weight: semi-bold;
        }

        /* DataFrame Styling */
        .stDataFrame {
            background-color: #1e1e1e; 
            color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            padding: 16px;
        }

        /* Button Styling */
        .stButton > button {
            background-color: #2196F3; 
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 12px 24px;
            margin-top: 12px;
            transition: background-color 0.3s ease-in-out;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #0b79d0;
        }

        /* Selectbox Styling */
        .stSelectbox > div {
            background-color: #2a2a2a; 
            color: #ffffff;
            border-radius: 6px;
            padding: 12px;
        }

        /* Plot Styling */
        .stPlot > div {
            background-color: #1e1e1e; 
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }

        /* Heatmap Styling */
        .stMarkdown > div {
            background-color: #121212; 
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.title("Enhanced AutoML Classification Application")

# File upload
uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Loaded:")
    st.dataframe(data.head())
    
    # Display basic info
    st.write("### Descriptive Statistics:")
    st.dataframe(data.describe())
    
    # Data Visualization
    st.write("### Data Visualizations:")
    
    st.subheader("Pair Plot of the Dataset")
    st.pyplot(sns.pairplot(data, diag_kind='kde'))
    
    st.subheader("Histogram of Features")
    num_features = data.select_dtypes(include=np.number).columns
    fig, axes = plt.subplots(nrows=len(num_features)//2, ncols=2, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(num_features):
        sns.histplot(data[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Histogram of {col}')
    plt.tight_layout()
    st.pyplot(fig)

    # Preprocessing for Classification
    st.write("### Preprocessing the data for classification...")
    
    # Handle missing values
    for col in data.select_dtypes(include=np.number).columns:
        data[col].fillna(data[col].mean(), inplace=True)

    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    target_column = data.columns[-1]  # Target column
    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Classifiers
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Bagging Classifier": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42)
    }

    for name, model in classifiers.items():
        st.write(f"### Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model - Show Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### {name} Accuracy: {accuracy * 100:.2f}%")
        
        # Confusion Matrix with Advanced Styling
        cm = confusion_matrix(y_test, y_pred)
        st.write(f"### {name} Confusion Matrix")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=np.unique(y), yticklabels=np.unique(y), linewidths=1, linecolor='black')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        st.pyplot(fig)  # Display the plot using the created figure
