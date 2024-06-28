from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS
import pickle

# Load the persisted model from a file
with open("car_sale_model.bin", "rb") as f:
    model = pickle.load(f)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Define the required features and the encoded features
required_features = ["odometer", "horse_power", "manufactured_date", "vehicle_type", "transmission_type", "fuel_type", "brand", "Season"]
encoded_features = ["odometer", "horse_power", "manufactured_date", "vehicle_type_compact car", "vehicle_type_convertible", 
                    "vehicle_type_other", "vehicle_type_sedan", "vehicle_type_station wagon", "vehicle_type_suv", 
                    "vehicle_type_van", "transmission_type_Automatic", "transmission_type_Manual", "fuel_type_CNG", 
                    "fuel_type_LPG", "fuel_type_diesel", "fuel_type_gasoline", "fuel_type_hybrid", "brand_alfa_romeo", 
                    "brand_audi", "brand_bmw", "brand_chevrolet", "brand_chrysler", "brand_citroen", "brand_dacia", "brand_daewoo", 
                    "brand_daihatsu", "brand_fiat", "brand_ford", "brand_honda", "brand_hyundai", "brand_jaguar", "brand_jeep", 
                    "brand_kia", "brand_lancia", "brand_land_rover", "brand_mazda", "brand_mercedes_benz", "brand_mini", 
                    "brand_mitsubishi", "brand_nissan", "brand_opel", "brand_peugeot", "brand_porsche", "brand_renault", 
                    "brand_rover", "brand_saab", "brand_seat", "brand_skoda", "brand_smart", "brand_sonstige_autos", 
                    "brand_subaru", "brand_suzuki", "brand_toyota", "brand_volkswagen", "brand_volvo", "Season_Spring", 
                    "Season_Winter"]

# Sample transformation function
def transform_input(data):
    # Initialize a list with zeros for all encoded features
    transformed = np.zeros(len(encoded_features))
    
    # Map vehicle types
    vehicle_type_map = {
        "compact_car": "vehicle_type_compact car",
        "convertible": "vehicle_type_convertible",
        "other": "vehicle_type_other",
        "sedan": "vehicle_type_sedan",
        "station_wagon": "vehicle_type_station wagon",
        "suv": "vehicle_type_suv",
        "van": "vehicle_type_van"
    }
    
    # Map transmission types
    transmission_type_map = {
        "Automatic": "transmission_type_Automatic",
        "Manual": "transmission_type_Manual"
    }
    
    # Map fuel types
    fuel_type_map = {
        "CNG": "fuel_type_CNG",
        "LPG": "fuel_type_LPG",
        "diesel": "fuel_type_diesel",
        "gasoline": "fuel_type_gasoline",
        "hybrid": "fuel_type_hybrid"
    }
    
    # Map brands
    brand_map = {brand: f"brand_{brand.lower().replace(' ', '_')}" for brand in [
        "alfa_romeo", "audi", "bmw", "chevrolet", "chrysler", "citroen", "dacia", "daewoo", 
        "daihatsu", "fiat", "ford", "honda", "hyundai", "jaguar", "jeep", 
        "kia", "lancia", "land_rover", "mazda", "mercedes_benz", "mini", 
        "mitsubishi", "nissan", "opel", "peugeot", "porsche", "renault", 
        "rover", "saab", "seat", "skoda", "smart", "sonstige_autos", 
        "subaru", "suzuki", "toyota", "volkswagen", "volvo"
    ]}
    
    # Map seasons
    season_map = {
        "Spring": "Season_Spring",
        "Winter": "Season_Winter"
    }
    
    # Assign the basic features
    transformed[encoded_features.index("odometer")] = data["odometer"]
    transformed[encoded_features.index("horse_power")] = data["horse_power"]
    transformed[encoded_features.index("manufactured_date")] = data["manufactured_date"]
    
    # Encode categorical features
    transformed[encoded_features.index(vehicle_type_map[data["vehicle_type"]])] = 1
    transformed[encoded_features.index(transmission_type_map[data["transmission_type"]])] = 1
    transformed[encoded_features.index(fuel_type_map[data["fuel_type"]])] = 1
    transformed[encoded_features.index(brand_map[data["brand"]])] = 1
    transformed[encoded_features.index(season_map[data["Season"]])] = 1
    
    # Encode Holiday feature (assuming 0 or 1)
    # if data["Holiday"]:
    #     transformed[encoded_features.index("Holiday")] = 1
    
    return transformed

# Define a route for the root URL that returns a welcome message
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello, Welcome to the car price prediction!'})

# Define a route for predicting car prices
@app.route("/predict", methods=["POST"])
def predict_price():
    # Get JSON data from the request
    data = request.get_json()

    # Check if required features are present
    if not all(key in data for key in required_features):
        return jsonify({"error": "Missing required features"}), 400

    # Transform input data
    transformed_input = transform_input(data)

    # Convert to DataFrame for model prediction
    transformed_input_df = pd.DataFrame([transformed_input], columns=encoded_features)

    # Predict the price using the model
    prediction = model.predict(transformed_input_df)[0]

    # Return the predicted price as a JSON response
    return jsonify({"predicted_price": prediction})

# Run the Flask application
# if __name__ == "__main__":
#     app.run()
