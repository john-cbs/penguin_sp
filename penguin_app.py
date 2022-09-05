# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

#the method of prediction
@st.cache()
def prediction (model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
  pred_out = model.predict(np.array([island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]).reshape(1,-1))
  if pred_out ==0:
    return "Adelie"
  elif pred_out ==1:
    return 'Chinstrap'
  elif pred_out ==2:
    return 'Gentoo'
  else :return pred_out



   #streamlit code
st.title("Penguin Species Prediction App")

#showing the types of species
if st.checkbox("show types of penguins"):
  st.write(df.species.unique())
  
st.write("The output will be here once you press the predict button with required values:\n")

#sliders for input values
bill_length_mm = st.sidebar.slider("bill_length_mm", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
bill_depth_mm = st.sidebar.slider("bill_depth_mm", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
flipper_length_mm = st.sidebar.slider("flipper_length_mm", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
body_mass_g = st.sidebar.slider("body_mass_g", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))

#selection boxes for sex and island columns
sex = st.sidebar.selectbox('sex', set(df.sex.unique()))
island = st.sidebar.selectbox('island', set(df.island.unique()))

# classifier selection box
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

#The predict button
if st.button("Predict"):
  if classifier=='Support Vector Machine':
    sp_type = prediction (svc_model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
    score =svc_model.score(X_train, y_train)
  elif classifier=='Logistic Regression':
    sp_type = prediction (log_reg,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
    score =log_reg.score(X_train, y_train)
  elif classifier=='Random Forest Classifier':
    sp_type = prediction (rf_clf,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
    score =rf_clf.score(X_train, y_train)

  st.write("Species predicted:", sp_type)
  st.write("Accuracy score of this model is:", score*100,"%")







