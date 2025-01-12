import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

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
        transition: background-color 0.3s ease-in-out, transform 0.2s;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stButton > button:hover {
        background-color: #0b79d0;
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
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

    /* Advanced Table Styling */
    .stDataFrame th, .stDataFrame td {
        padding: 12px 16px;
        border: 1px solid #444;
        background-color: #2a2a2a;
        border-radius: 6px;
    }

    .stDataFrame tr:hover {
        background-color: #333;
        cursor: pointer;
    }

    /* Header Styling */
    .stDataFrame thead tr {
        background-color: #333;
        color: white;
        font-weight: bold;
    }

    /* Input Text Styling */
    .stTextInput > div {
        background-color: #2a2a2a;
        border: 1px solid #444;
        border-radius: 6px;
        color: #ffffff;
        padding: 12px;
    }

    /* Card Component Styling */
    .stCard {
        background-color: #1e1e1e;
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        padding: 20px;
        margin: 20px 0;
        transition: transform 0.2s;
    }
    .stCard:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }

    /* Modal Styling */
    .stDialog {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        animation: fadeIn 0.3s ease-in-out;
    }

    /* Keyframes for fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }

    /* Responsive Images */
    .stImage img {
        width: 100%;
        height: auto;
        border-radius: 10px;
    }

    /* Checkbox Styling */
    .stCheckbox {
        color: #ffffff;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: #2196F3;
    }

    /* Link Styling */
    a {
        color: #2196F3;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    /* Toast Notifications */
    .toast-notification {
        background-color: #2a2a2a;
        color: white;
        padding: 12px;
        border-radius: 6px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        animation: fadeInToast 0.4s ease-in;
    }

    /* Keyframes for Toast fade-in */
    @keyframes fadeInToast {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Scrollbar Customization */
    .stDataFrame::-webkit-scrollbar {
        width: 8px;
    }

    .stDataFrame::-webkit-scrollbar-track {
        background: #1e1e1e;
    }

    .stDataFrame::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 4px;
    }

    /* Interactive Table: Row Hover Effect */
    .stDataFrame tr:hover {
        background-color: #333;
        cursor: pointer;
    }

    /* Tooltip Styling */
    .stTooltip {
        background-color: #2a2a2a;
        color: #ffffff;
        padding: 8px 12px;
        border-radius: 6px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }

    /* Icon Tooltip Styling */
    .stIconButton:hover .stTooltip {
        display: block;
    }

</style>

    """,
    unsafe_allow_html=True
)

# App Header
st.title("Prediction for ML Classification Dataset")

# Classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# File upload with error handling
uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load the dataset with error handling
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        
        if data.empty:
            raise ValueError("The uploaded dataset is empty. Please upload a valid dataset.")

        st.write("### Dataset Loaded:")
        st.dataframe(data.head())
        
        # Display basic info
        st.write("### Descriptive Statistics:")
        st.dataframe(data.describe())
        
        # Data Visualization
        st.write("### Data Visualizations:")
        
        # Histograms of numerical features
        st.subheader("Histograms of Features")
        num_features = data.select_dtypes(include=np.number).columns
        fig, axes = plt.subplots(nrows=(len(num_features) + 1)//2, ncols=2, figsize=(16, 12))
        axes = axes.flatten()
        for i, col in enumerate(num_features):
            sns.histplot(data[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Histogram of {col}')
        plt.tight_layout()
        st.pyplot(fig)

        # Boxplot - User selects column
        st.subheader("Box Plot")
        box_column = st.selectbox("Select a numerical column for box plot:", data.select_dtypes(include=[np.number]).columns)
        if box_column:
            st.write(f"### Box Plot for {box_column}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data[box_column], ax=ax, palette='Set3')
            ax.set_title(f'Box Plot of {box_column}')
            st.pyplot(fig)

        # Bar Chart - User selects column
        st.subheader("Bar Chart")
        selected_bar_col = st.selectbox("Select a column for bar chart:", data.columns)
        if selected_bar_col:
            value_counts = data[selected_bar_col].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='plasma')
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', size=12, color='black')
            plt.title(f'Bar Chart of {selected_bar_col}')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Preprocessing for Classification
        st.write("### Perform Model Training below")
        
        # Handle missing values with intelligent imputation
        for col in data.select_dtypes(include=np.number).columns:
            data[col].fillna(data[col].mean(), inplace=True)

        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        target_column = st.selectbox("## Select Target Column for Classification:", data.columns)
        if target_column:
            X = data.drop(columns=[target_column])  # Features
            y = data[target_column]  # Target
        
            # Split data with stratification to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Scaling features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Select Algorithms for Training
            st.write("### Select Algorithms for Training and Evaluation")
            selected_algorithms = st.multiselect(
                "Choose one or more algorithms to train and evaluate:",
                options=list(classifiers.keys())
            )

            if selected_algorithms:
                results = []
                for name in selected_algorithms:
                    model = classifiers[name]
                    st.write(f"### Training {name}")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Evaluate the model - Show Accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = classification_report(y_test, y_pred, output_dict=True)[str(0)]['precision']
                    recall = classification_report(y_test, y_pred, output_dict=True)[str(0)]['recall']
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    results.append({
                        "Model": name,
                        "Accuracy": f"{accuracy * 100:.2f}%",
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1
                    })
                    
                    # Confusion Matrix with Advanced Styling
                    cm = confusion_matrix(y_test, y_pred)
                    st.write(f"### {name} Confusion Matrix")
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                                xticklabels=np.unique(y), yticklabels=np.unique(y), linewidths=1, linecolor='black')
                    plt.title(f'{name} Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=45)
                    
                    st.pyplot(fig)  # Display the plot using the created figure
                    
                    # Classification Report in the form of table
                    st.write(f"### {name} Classification Report")
                    clf_report_dict = classification_report(y_test, y_pred, output_dict=True)
                    clf_report_df = pd.DataFrame(clf_report_dict).transpose()
                    st.dataframe(clf_report_df)
                
                # Display results in a table
                results_df = pd.DataFrame(results)
                st.write("### Classification Results:")
                st.table(results_df)
                
        if uploaded_file and selected_algorithms:  # Added condition to check if algorithms are selected
            # Save the trained model to a file
            st.write("### Download Trained Model")
            model = classifiers[selected_algorithms[0]]
            model.fit(X_train, y_train)
            model_filename = 'trained_model.sav'
            pd.to_pickle(model, model_filename)
            st.download_button(
                label="Download Trained Model",
                data=open(model_filename, 'rb'),
                file_name=model_filename,
                mime='application/octet-stream'
            )
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
