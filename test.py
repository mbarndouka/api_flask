import requests

url = "http://localhost:5000/predict"

# Replace with your actual car data
data = {
  "features": {
                "vehicle_type": "sedan",
                "transmission_type": "automatic",
                "fuel_type": "petrol",
                "repaired": "no",
                "brand": "Toyota",
                "odometer": 150000,
                "season": "summer",
                "horse_power": 150,
                "Holiday": "no",
                "manufactured_date": "1998-01-01"
            }
}

# Send the POST request and get the response
response = requests.post(url, json=data)

# Check for successful response
if response.status_code == 200:
  # Convert JSON response to Python dictionary
  data = response.json()
  predicted_price = data["price"]
  print(f"Predicted price: ${predicted_price}")
else:
  print(f"Error: {response.status_code}")