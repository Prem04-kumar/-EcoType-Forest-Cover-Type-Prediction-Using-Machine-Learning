**🌲 EcoType: Forest Cover Type Prediction Using Machine Learning


#📌 Project Overview
EcoType is a machine learning classification project that predicts the forest cover type of a geographical area using cartographic and environmental features such as elevation, slope, soil type, and distance measures. The project supports environmental monitoring, forestry management, and land-use planning by providing an automated and reliable prediction system.

🎯 Problem Statement
To develop a machine learning classification model that accurately predicts the forest cover type based on cartographic variables, enabling efficient forest resource management and ecological analysis.

🌿 Domain
Environmental Data & Geospatial Predictive Modeling

📚 Skills & Technologies Used
Exploratory Data Analysis (EDA)
Data Cleaning & Preprocessing
Skewness Detection & Handling
Feature Engineering
Class Imbalance Handling (SMOTE)
Classification Models
Model Evaluation
Streamlit Application Development
Model Deployment
Libraries & Tools: Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn, Streamlit, Joblib

📊 Dataset Information
Source: Forest Cover Type Dataset
Size: 145,890 rows × 12 columns
Target Variable: Cover_Type (7 classes)
🔍 Exploratory Data Analysis (EDA)
EDA was performed in a separate Jupyter notebook to understand feature distributions, skewness, class imbalance, correlations, and feature importance.

Notebook:

notebooks/cover_type.ipynb
⚙️ Data Preprocessing
Verified no missing values
Detected skewed features using skewness metrics
Applied transformations where required
Encoded target variable
Ensured consistent feature selection
⚖️ Class Imbalance Handling
SMOTE (Synthetic Minority Oversampling Technique) was applied on the training dataset to balance class distribution.

🧠 Model Building & Evaluation
Models trained:

•Logistic Regression

•Decision Tree

•K-Nearest Neighbors (KNN)

•Random Forest

•XGBoost

Evaluation metrics:

Accuracy
Confusion Matrix
Classification Report
📈 Model Comparison Summary
Model	Accuracy

•Logistic Regression	0.64

•Decision Tree	0.93

•KNN	0.88

•Random Forest	0.95

•XGBoost	0.94

Best Model Selected: Random Forest

Notebook:

notebooks/cover_type.ipynb

💾 Model Saving
Saved artifacts using joblib:

random_forest.pkl
features.pkl
label_encoder.pkl

🌐 Streamlit Application
A Streamlit web application was developed for single-instance prediction using manual numeric inputs.

Run the app:"C:\Users\A Prem kumar\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run "C:\VSCODE\app.py"

📁 Project Structure
Eco_Type_Forest_Prediction/
│
├── data/
│   └── cover_type.csv
│
├── notebooks/
│   └──cover_type.ipynb
│
├── app.py
├── random_forest.pkl
├── features.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md

▶️ How to Run the Project
Follow the steps below to run the project locally.

•1️⃣ Clone the Repository
git clone <your-github-repo-link>
cd Eco_Type_Forest_Prediction

•2️⃣ Create and Activate Virtual Environment
Windows

python -m venv venv
venv\Scripts\activate
Mac / Linux

python3 -m venv venv
source venv/bin/activate

•3️⃣ Install Required Dependencies
pip install -r requirements.txt



•4️⃣ Run Model Training (One-Time)
This step trains the final model and saves it as .pkl files.

forest_cover_model.pkl
selected_features.pkl
label_encoder.pkl

•5️⃣ Run the Streamlit Application
"C:\Users\A Prem kumar\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run "C:\VSCODE\app.py"
The application will open in your browser and allow you to:

Enter feature values manually
Predict the forest cover type

✅ Notes
Ensure Python 3.8+ is installed
Model training is done only once
Streamlit app uses the saved model for prediction
🏁 Conclusion
EcoType demonstrates a complete end-to-end machine learning pipeline—from data analysis and model comparison to deployment—providing a practical solution for forest cover type prediction.

👤 Author

Prem Kumar.A
