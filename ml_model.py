import pandas as pd
import joblib

# Load the saved model and label encoders
rf_regressor = joblib.load('crop_price_prediction_model.pkl')  # Load the trained model
label_encoders = joblib.load('label_encoders.pkl')  # Load the label encoders

# Function to preprocess user input and make predictions
def predict_crop_price(state, district, market, commodity, year, month, day):
    # Encode the user input using the loaded label encoders
    state_encoded = label_encoders['State'].transform([state])[0]
    district_encoded = label_encoders['District'].transform([district])[0]
    market_encoded = label_encoders['Market'].transform([market])[0]
    commodity_encoded = label_encoders['Commodity'].transform([commodity])[0]

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'State': [state_encoded],
        'District': [district_encoded],
        'Market': [market_encoded],
        'Commodity': [commodity_encoded],
        'Variety': [0],  # Assuming 'Other' as default (you can modify this if needed)
        'Grade': [0],    # Assuming 'FAQ' as default (you can modify this if needed)
        'Commodity_Code': [0],  # Assuming default value (you can modify this if needed)
        'Year': [year],
        'Month': [month],
        'Day': [day]
    })

    # Predict the modal price
    predicted_price = rf_regressor.predict(user_input)
    return predicted_price[0]

# Example usage of the function
state = 'Maharashtra'
district = 'Jalgaon'
market = 'Chalisgaon'
commodity = 'Bajra(Pearl Millet/Cumbu)'
year = 2025
month = 9
day = 20

predicted_price = predict_crop_price(state, district, market, commodity, year, month, day)
print(f'Predicted Modal Price for {commodity} in {market} on {day}-{month}-{year}: {predicted_price}')