import numpy as np
import pickle

# Load the pre-trained linear regression model
with open('kmeans.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_cluster():
    try:
        annual_income = int(input("Enter the Annual Income in thousands(k): "))
        score = float(input("Enter the Spending Score (1-100): "))
       
        # Make predictions using the model
        input_data = np.array([annual_income, score])
        predicted_cluster = model.predict(input_data.reshape(1, -1))[0]
        print(f"Predicted Cluster: {predicted_cluster}")
    
    except ValueError:
        print("Error: Please enter valid numerical values for all input fields.")


if __name__ == "__main__":
    print("----------------------- Customer Grouping --------------------------------")
    predict_cluster()