import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def recommend_order_delivery(orders_df):
    # Mock machine learning model (you should train a real model with relevant data)
    features = ["num_together", "delivery_person_rating", "age"]
    target = "delivery_time"  # You may need to replace this with the actual target variable

    model = RandomForestRegressor()  # Replace with your trained model

    # Train the model (for demonstration, using the same data as features)
    model.fit(orders_df[features], orders_df[target])

    # Make predictions
    orders_df["predicted_delivery_time"] = model.predict(orders_df[features])

    # Recommend the order with the shortest predicted delivery time
    recommended_order = orders_df.loc[orders_df["predicted_delivery_time"].idxmin()]

    return recommended_order

def main():
    st.title("Food Order Delivery Recommendation Framework")

    # Get the number of orders specified by the user
    num_orders = st.number_input("Enter the number of orders:", min_value=1, step=1)

    # Collect details for each order
    orders = []
    for i in range(num_orders):
        st.subheader(f"Order {i + 1} Details")

        order_date = st.date_input("Order Date", key=f"date_{i}")
        order_time = st.time_input("Order Time", key=f"time_{i}")
        pickup_time = st.time_input("Order Pick Up Time", key=f"pickup_time_{i}")
        order_type = st.selectbox("Type of Order", ["Breakfast", "Lunch", "Dinner"], key=f"order_type_{i}")
        num_together = st.number_input("Number of Orders Together", min_value=1, step=1, key=f"num_together_{i}")
        location = st.text_input("Location (Choose from Maps)", key=f"location_{i}")
        delivery_person_rating = st.slider("Rating of Delivery Person", min_value=1, max_value=5, step=1, key=f"rating_{i}")
        age = st.number_input("Age", min_value=1, step=1, key=f"age_{i}")
        city_type = st.selectbox("City Type", ["Urban", "Suburban", "Rural"], key=f"city_type_{i}")

        order_details = {
            "order_date": order_date,
            "order_time": order_time,
            "pickup_time": pickup_time,
            "order_type": order_type,
            "num_together": num_together,
            "location": location,
            "delivery_person_rating": delivery_person_rating,
            "age": age,
            "city_type": city_type,
        }

        orders.append(order_details)

    # Create a DataFrame from the collected order details
    orders_df = pd.DataFrame(orders)

    # Display the collected order details
    st.subheader("Order Details:")
    st.write(orders_df)

    # Recommend which order should be delivered first using machine learning
    recommended_order = recommend_order_delivery(orders_df)

    # Display the recommended order
    st.subheader("Recommended Order to Deliver First:")
    st.write(recommended_order)

if __name__ == "__main__":
    main()
