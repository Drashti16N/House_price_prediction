import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import  r2_score
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
import category_encoders as ce
import numpy as np

# Load the dataset
data = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/sales/house_price/house_price.csv')# Replace 'your_dataset.csv' with the actual dataset path
data_copy = data.copy()
data_copy['availability'] = data_copy['availability'].apply(lambda x: 1 if x == 'Ready To Move' else 0)

# Perform ordinal encoding to area type column assigning 1,2,3 and 4 to Carpet Area, Plot Area, Built-up Area and Super built-up Area respectivelly:
from sklearn.preprocessing import OrdinalEncoder
ordinal_mapping = {
    'Super built-up  Area': 4,
    'Built-up  Area': 3,
    'Plot  Area': 2,
    'Carpet  Area': 1
}

# instance of the OrdinalEncoder with the defined mapping
ordinal_encoder = OrdinalEncoder(categories=[sorted(ordinal_mapping, key=ordinal_mapping.get, reverse=True)])

# Apply ordinal encoding to the area_type column
data_copy['area_type'] = ordinal_encoder.fit_transform(data_copy[['area_type']])

# Perform target encoding on "society" and "location" column of data frame:
encoder = ce.TargetEncoder(cols=['society', 'location'])

# Fit and transform the data
df_encoded = encoder.fit_transform(data_copy, data_copy['price'])

# Replace the original columns with the encoded ones
df_encoded[['society', 'location']] = df_encoded[['society', 'location']]

data_copy['price'] = np.log(data_copy['price'])
x = df_encoded.drop('price',axis = 1)
# Define features and target variable
y = data_copy['price']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

XGBoost = XGBRegressor()
# Train the Random Forest model
XGBoost.fit(X_train_scaled, y_train)

# Predictions
y_pred = XGBoost.predict(X_test_scaled)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Print the R2 score for Random Forest model
st.write(f'R2 Score for XGBoost Model: {r2}')

# Streamlit app for predicting crop yield
import streamlit as st
import pandas as pd

st.title('House Price Prediction App')

# Input fields for features
Area_type = st.selectbox('area_type', data['area_type'].unique())
Availability = st.selectbox('availability', data['availability'].unique())
Location = st.selectbox('location', data['location'].unique())
Society = st.selectbox('society', data['society'].unique())
Total_sqft = st.number_input('total_sqft', value=0.0)
Bath = st.number_input('bath', value=0.0)
Balcony = st.number_input('balcony', value=0.0)
Size_n = st.number_input('size_n', value=0.0)


# Prepare input features
input_features = pd.DataFrame({
    'area_type': [Area_type],
    'availability': [Availability],
    'location': [Location],
    'society': [Society],
    'total_sqft': [Total_sqft],
    'bath': [Bath],
    'balcony': [Balcony],
    'size_n': [Size_n]
})

# Predict button
if st.button('Predict'):
    # Predict using the trained model
    prediction = XGBoost.predict(input_features)
    st.success(f'Predicted House Price: {prediction[0]}')