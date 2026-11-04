# ---------- TOOLS I NEED ----------


import streamlit as st        # builds the web page (st = nickname for streamlit)
import pandas as pd           # handles tables of data (pd = nickname for pandas)
from joblib import load       # reads a saved model file back in
import matplotlib.pyplot as plt   # draws charts (plt = nickname)


# ---------- PAGE HEADING ----------


st.title("Car Price Predictor (Canada)")
st.write("Enter the details of a used car to get an estimated price")


# ---------- LOAD THE TRAINED MODEL (the "model" I made earlier) ----------

# The line below tells Streamlit: remember this, don't reload it on every click

@st.cache_resource
def get_car_price_model():
    return load('model/model.joblib')

car_price_model = get_car_price_model()    # the () runs the function


# ---------- LOAD THE CAR DATA (only used to fill the dropdowns) ----------

@st.cache_data
def get_car_data():
    return pd.read_csv('data/used_car_canada_clean.csv')

car_data = get_car_data()



# ---------- DROPDOWN 1: MAKE ----------

all_makes = sorted(car_data['make'].unique())      # every make once, A to Z
chosen_make = st.selectbox("Make", all_makes)      # the user picks one


# ---------- DROPDOWN 2: MODEL ----------

cars_of_chosen_make = car_data[car_data['make'] == chosen_make]   # keep only that make
models_available = sorted(cars_of_chosen_make['model'].unique()) # its models, A to Z
chosen_model = st.selectbox("Model", models_available)       # the user picks one


# Number inputs      

chosen_miles = st.number_input("Miles driven", min_value=0, max_value=500000, value=60000, step=1000)

chosen_year = st.number_input(
    "Year",
    min_value=int(car_data['year'].min()),      #oldest year in the data
    max_value=int(car_data['year'].max()),       #newest year in the data
    value= 2017,
    step=1 
)

chosen_engine_size = st.number_input("Engine size (litres)", min_value=0.6, max_value=8.4, value=2.0, step=0.1)

#More Dropdown
### The choices  for these dropdown are pulled straight from the data 
chosen_body_type=st.selectbox("Body type", sorted(car_data['body_type'].unique()))
chosen_vehicle_type=st.selectbox("Vehicle type", sorted(car_data["vehicle_type"].unique()))
chosen_drivetrain = st.selectbox("Drivetrain", sorted(car_data['drivetrain'].unique()))
chosen_transmission = st.selectbox("Transmission", sorted(car_data['transmission'].unique()))
chosen_fuel_type = st.selectbox("Fuel type", sorted(car_data['fuel_type'].unique()))
chosen_engine_block = st.selectbox("Engine block", sorted(car_data['engine_block'].unique()))
chosen_state = st.selectbox("Province / state", sorted(car_data['state'].unique()))


# ---------- PREDICT BUTTON ----------

if st.button('Predict price'):
     # Put the answers into a one-row table.
    one_car = pd.DataFrame([{
        'miles': chosen_miles,
        'year': chosen_year,
        'make': chosen_make,
        'model': chosen_model,
        'body_type': chosen_body_type,
        'vehicle_type': chosen_vehicle_type,
        'drivetrain': chosen_drivetrain,
        'transmission': chosen_transmission,
        'fuel_type': chosen_fuel_type,
        'engine_size': chosen_engine_size,
        'engine_block': chosen_engine_block,
        'state': chosen_state,
    }])

    # Ask the model for a price. [0] picks the first (only) answer.
    predicted_price = car_price_model.predict(one_car)[0]

    # Show it, with a comma for thousands and no decimals
    st.subheader(f"Estimated price: ${predicted_price:,.0f}")

        # ---------- CHART: WHERE YOUR CAR SITS AMONG REAL LISTINGS ----------
    # Keep only real cars of the same make AND model
    similar_cars = car_data[(car_data['make'] == chosen_make) & (car_data['model'] == chosen_model)]

    figure, axis = plt.subplots()                     # an empty chart
    axis.scatter(similar_cars['miles'], similar_cars['price'], alpha=0.3, label="Cars for sale")
    axis.scatter(chosen_miles, predicted_price, color="red", s=100, label="Your car (prediction)")
    axis.set_xlim(0, 300000)                          # ignore a few extreme high-mileage cars
    axis.set_xlabel("Miles")
    axis.set_ylabel("Price ($)")
    axis.legend()                                     # the little key explaining the dots
    st.pyplot(figure)                                 # show the chart in the app