import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline    
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor

df = pd.read_csv('Dataset_Fertiliser_Final.csv')
df.dropna(inplace=True)
df=df.drop(columns=['Region'])
print(df.head())

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in '{column}':")
    print(unique_values)
    print("\n")
    
X=df[['State','City','N','P','K','Soil_Type','Crop_Type','Moisture','Humidity','Temperature','pH']]
y=df[['Fertilizer_Amount','Fertilizer_Recommendation']]

le = LabelEncoder()
df['Fertilizer_Recommendation'] = le.fit_transform(df['Fertilizer_Recommendation'])

numerical_features = ['N', 'P', 'K', 'Moisture', 'Humidity', 'Temperature', 'pH']
categorical_features = ['State', 'City', 'Soil_Type', 'Crop_Type']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_amt = y_train['Fertilizer_Amount']
y_train_rec = y_train['Fertilizer_Recommendation']
y_test_amt = y_test['Fertilizer_Amount']
y_test_rec = y_test['Fertilizer_Recommendation']

le = LabelEncoder()
combined_rec = pd.concat([y_train_rec, y_test_rec])
le.fit(combined_rec)

y_train_rec_encoded = le.transform(y_train_rec)
y_test_rec_encoded = le.transform(y_test_rec)

pipeline_amt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50,max_depth=10,min_samples_leaf=4,min_samples_split=10,random_state=40))
])

pipeline_rec = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=10))
])

pipeline_amt.fit(X_train, y_train_amt)
pipeline_rec.fit(X_train, y_train_rec_encoded)

y_pred_amt = pipeline_amt.predict(X_test)
y_pred_rec = pipeline_rec.predict(X_test)

y_pred_rec_decoded = le.inverse_transform(y_pred_rec.astype(int))

mse_amt = mean_squared_error(y_test_amt, y_pred_amt)
r2_amt = r2_score(y_test_amt, y_pred_amt)
mse_rec = mean_squared_error(y_test_rec_encoded, y_pred_rec)
r2_rec = r2_score(y_test_rec_encoded, y_pred_rec)

input_data = {
    'State': ['West Bengal'],
    'City': ['Durgapur'],
    'N': [62],
    'P': [59],
    'K': [45],
    'Soil_Type': ['Loamy Soil'],
    'Crop_Type': ['Wheat'],
    'Moisture': [20.0],
    'Humidity': [40.0],
    'Temperature': [30.0],
    'pH': [6.5]
}

input_df = pd.DataFrame(input_data)
predicted_amt = pipeline_amt.predict(input_df)
predicted_rec_encoded = pipeline_rec.predict(input_df)
predicted_rec = le.inverse_transform(predicted_rec_encoded.astype(int))

print("Predicted Fertilizer Amount:", round(predicted_amt[0],2))
print("Predicted Fertilizer Recommendation:", predicted_rec[0])

import numpy as np

# Calculate the standard deviation of the predicted fertilizer amounts
std_dev_amt = np.std(y_pred_amt)
print("Standard Deviation of Predicted Fertilizer Amounts:", round(std_dev_amt, 2))
# Calculate the standard deviation of the actual fertilizer amounts
std_dev_actual_amt = np.std(y_test_amt)
print("Standard Deviation of Actual Fertilizer Amounts:", round(std_dev_actual_amt, 2))


predictions = []
for i in range(100):
    input_data['N'][0] += np.random.uniform(-2, 2)  # Slightly vary the 'N' value for this example
    input_df = pd.DataFrame(input_data)
    predicted_amt = pipeline_amt.predict(input_df)
    predictions.append(predicted_amt[0])

# Calculate the standard deviation of these predictions
std_dev_custom_pred = np.std(predictions)
print("Standard Deviation of Custom Predictions:", round(std_dev_custom_pred, 2))
