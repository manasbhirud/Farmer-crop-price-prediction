import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv(r"D:\Farmer Project ML\crop_Price_prediction\combined_dataset.csv")
print(data.head())

print(data.isnull().sum())

# Convert 'Arrival_Date' to datetime format
data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], format='%d-%m-%Y')

# Extract year, month, and day from the 'Arrival_Date'
data['Year'] = data['Arrival_Date'].dt.year
data['Month'] = data['Arrival_Date'].dt.month
data['Day'] = data['Arrival_Date'].dt.day

# Drop the original 'Arrival_Date' column
data.drop('Arrival_Date', axis=1, inplace=True)

# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Display the first few rows of the preprocessed dataset
print(data.head())


Q1,Q3 = data.Modal_Price.quantile([0.25,0.75])
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

data_new = data[(data["Modal_Price"]>lower)&(data["Modal_Price"]<upper)]
# Cap outliers instead of removing them
lower_bound = data_new["Modal_Price"].quantile(0.01)
upper_bound = data_new["Modal_Price"].quantile(0.99)

data_new["Modal_Price"] = np.clip(data_new["Modal_Price"], lower_bound, upper_bound)

print(data_new.head())
print(data_new["Modal_Price"].describe())
# Define features (X) and target variable (y)
X = data_new.drop(['Min_Price', 'Max_Price', 'Modal_Price'], axis=1)
y = data_new['Modal_Price']  # Using Modal_Price as the target variable

# Display the first few rows of X and y
print(X.head())
print(y.head())

# plt.boxplot(data_new["Modal_Price"])
# plt.show()


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Squared Error: {mse}')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print(f'R-squared: {r2}')

# Save the trained model to a file
joblib.dump(rf_regressor, 'crop_price_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')  
# # Load the model from the file (if needed)
# rf_regressor = joblib.load('crop_price_prediction_model.pkl')