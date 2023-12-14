import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def set_random_seed(seed=42):
    np.random.seed(seed)

# Generate synthetic data based on provided class properties
def generate_synthetic_data(class_properties, samples_per_class=300, num_dimensions=2):        
    data = []
    for class_id, props in class_properties.items():
        mean, std = props['mean'], props['std']
        samples = np.random.normal(mean, std, size=(samples_per_class, num_dimensions))
        labels = np.full((samples_per_class, 1), class_id)
        class_data = np.hstack((samples, labels))
        data.append(class_data)

    # Combine data from all classes
    data = np.vstack(data)
    np.random.shuffle(data)
    return data

# Create a Pandas DataFrame from the generated synthetic data
def create_dataframe(data, num_dimensions):
    column_names = [f'Feature_{i+1}' for i in range(num_dimensions)] + ['Class']
    return pd.DataFrame(data, columns=column_names)

# Visualize the synthetic data distribution
def visualize_synthetic_data(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Feature_1', y='Feature_2', data=df, color='blue', palette='viridis')
    plt.title('Synthetic Data Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return fig

# Split the data into training and testing sets
def split_data(df):
    X = df.iloc[:, :-1]
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# This function trains a decision tree classifier using the provided training data.
def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Save the trained model to a file using joblib
def save_model(model, filename="trained_model.joblib"):
    joblib.dump(model, filename)

def load_model(filename="trained_model.joblib"):
    loaded_model = joblib.load(filename)
    return loaded_model

# Visualize decision boundaries and actual data points
def visualize_decision_boundaries(model, X, y, df):
    # Visualization of Decision Boundaries
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual data points
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis')
    plt.title('Actual Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot decision boundaries
    plt.subplot(1, 2, 2)

    # Create a meshgrid to cover the entire feature space
    h = 0.02
    x_min, x_max = X.iloc[:, 0].min() - 6, X.iloc[:, 0].max() + 2
    y_min, y_max = X.iloc[:, 1].min() - 5, X.iloc[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on the meshgrid to obtain decision boundaries
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # Uses the trained model to predict the class for each point in the feature space
    Z = Z.reshape(xx.shape) # Reshapes the predictions back into the shape of the meshgrid.

    # Visualize the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis', alpha=0.5)
    plt.title('Decision Boundaries and Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Class')

    return fig

# Evaluate the trained model on the test set and visualize actual vs predicted data
def evaluate_model(model, X_test, y_test, df):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and display classification report
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    # st.write("\nClassification Report:")
    # st.write(classification_report(y_test, y_pred))

    # Plot actual and predicted data points 
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual data points
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis')
    plt.title('Actual Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot predicted data points
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis', alpha=0.5)
    sns.scatterplot(x=X_test['Feature_1'], y=X_test['Feature_2'], hue=y_pred, marker='X', s=100, palette='Set2', edgecolor='black')
    plt.title('Actual vs Predicted Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    return fig

def main():
    # Set random seed for reproducibility
    set_random_seed()

    # Define class properties for synthetic data generation
    class_properties = {
        0: {'mean': [25, 30], 'std': [5, 4]},
        1: {'mean': [40, 45], 'std': [6, 5]},
        2: {'mean': [55, 60], 'std': [4, 3.5]}
    }

    # Generate synthetic data and create a DataFrame
    data = generate_synthetic_data(class_properties)
    df = create_dataframe(data, num_dimensions=2)

    # Streamlit setup
    st.title("Modeling and Simulation")

    # Visualization of synthetic data distribution
    st.header("Synthetic Data Distribution")
    vsd = visualize_synthetic_data(df)
    st.pyplot(vsd)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Train a decision tree classifier
    clf = train_decision_tree(X_train, y_train)

    # Save the trained model
    save_model(clf)

    # Load the saved model
    loaded_model = load_model()

    # Visualization of decision boundaries and Data Points
    st.header("Decision Boundaries and Data Points")
    vdb = visualize_decision_boundaries(loaded_model, X_test, y_test, df)
    st.pyplot(vdb)

    # Actual vs Predicted data
    st.header("Actual vs Predicted Data")
    fig = evaluate_model(loaded_model, X_test, y_test, df)
    st.pyplot(fig)

if __name__ == "__main__":
    main()