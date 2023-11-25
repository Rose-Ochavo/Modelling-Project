import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Function to perform data generation
def generate_data():
    np.random.seed(42)
    num_samples = 1000
    feature1 = np.random.normal(0, 1, num_samples)
    feature2 = 2 * feature1 + np.random.normal(0, 1, num_samples)
    target = 3 * feature1 + 4 * feature2 + np.random.normal(0, 1, num_samples)
    data = pd.DataFrame({'Feature1': feature1, 'Feature2': feature2, 'Target': target})
    return data

# Function to perform linear regression simulation
def simulate_linear_regression(data):
    X = data[['Feature1', 'Feature2']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    simulated_outcomes = model.predict(X)
    data['Simulated_Outcome'] = simulated_outcomes
    return data

# Function to perform evaluation and analysis
def evaluate_and_analyze(data):
    mse_simulation = mean_squared_error(data['Target'], data['Simulated_Outcome'])
    return mse_simulation

# Streamlit app
st.title('Modeling and Simulation with Python')
st.write('This app showcases a simple project on modeling and simulation using Python.')

# Generate synthetic data
st.header('Data Generation')
generated_data = generate_data()
st.write('Generated synthetic data:')
st.write(generated_data.head())

# Perform linear regression simulation
st.header('Linear Regression Simulation')
simulated_data = simulate_linear_regression(generated_data)
st.write('Simulated data with linear regression:')
st.write(simulated_data.head())

# Evaluate and analyze
st.header('Evaluation and Analysis')
mse_simulation = evaluate_and_analyze(simulated_data)
st.write(f'Mean Squared Error for Simulation: {mse_simulation}')

# Visualize the distribution of actual and simulated outcomes
st.subheader('Distribution of Actual and Simulated Outcomes')
fig, ax = plt.subplots()
sns.kdeplot(simulated_data['Target'], label='Actual Outcomes', shade=True, ax=ax)
sns.kdeplot(simulated_data['Simulated_Outcome'], label='Simulated Outcomes', shade=True, ax=ax)
st.pyplot(fig)

