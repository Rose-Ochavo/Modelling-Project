# Import necessary libraries
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set a random seed for reproducibility
def set_random_seed(seed=42):
    """
    This function sets a random seed to ensure reproducibility of random processes.

    Parameters:
    - seed (int): A fixed number used to initialize the random number generator.
    """
    np.random.seed(seed)

# Generate synthetic data based on provided class properties
def generate_synthetic_data(class_properties, samples_per_class=300, num_dimensions=2):
    """
    This function generates synthetic data based on provided characteristics for each class.

    Parameters:
    - class_properties (dict): Characteristics of each class, including mean and standard deviation.
    - samples_per_class (int): Number of synthetic samples to generate for each class.
    - num_dimensions (int): Number of features for each sample.

    Returns:
    - data (numpy.ndarray): Synthetic data with corresponding labels.
    """
    data = []
    for class_id, props in class_properties.items():
        mean, std = props['mean'], props['std']
        samples = np.random.normal(mean, std, size=(samples_per_class, num_dimensions))
        labels = np.full((samples_per_class, 1), class_id)
        class_data = np.hstack((samples, labels))
        data.append(class_data)

    # Combine data from all classes and shuffle
    data = np.vstack(data)
    np.random.shuffle(data)
    return data

# Create a Pandas DataFrame from the generated synthetic data
def create_dataframe(data, num_dimensions):
    """
    This function converts the generated synthetic data into a structured DataFrame.

    Parameters:
    - data (numpy.ndarray): Synthetic data with corresponding labels.
    - num_dimensions (int): Number of features for each sample.

    Returns:
    - df (pandas.DataFrame): A structured DataFrame containing synthetic data.
    """
    column_names = [f'Feature_{i+1}' for i in range(num_dimensions)] + ['Class']
    return pd.DataFrame(data, columns=column_names)

# Visualize the synthetic data distribution
def visualize_synthetic_data(df):
    """
    This function creates a visualization of the synthetic data distribution.

    Parameters:
    - df (pandas.DataFrame): A structured DataFrame containing synthetic data.

    Returns:
    - fig (matplotlib.figure.Figure): A visual representation of the synthetic data distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Feature_1', y='Feature_2', data=df, color='blue', palette='viridis')
    plt.title('Synthetic Data Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return fig

# Split the data into training and testing sets
def split_data(df):
    """
    This function splits the synthetic data into training and testing sets.

    Parameters:
    - df (pandas.DataFrame): A structured DataFrame containing synthetic data.

    Returns:
    - X_train, X_test, y_train, y_test (tuple): Train-test split of features and labels.
    """
    X = df.iloc[:, :-1]
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier using the training data
def train_decision_tree(X_train, y_train):
    """
    This function trains a decision tree classifier using the provided training data.

    Parameters:
    - X_train (pandas.DataFrame): Training features.
    - y_train (pandas.Series): Training labels.

    Returns:
    - clf (sklearn.tree.DecisionTreeClassifier): Trained decision tree classifier.
    """
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Save the trained model to a file using joblib
def save_model(model, filename="trained_model.joblib"):
    """
    This function saves the trained machine learning model to a file using joblib.

    Parameters:
    - model: Trained machine learning model.
    - filename (str): Name of the file to save the model.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Load a pre-trained model from a file using joblib
def load_model(filename="trained_model.joblib"):
    """
    This function loads a pre-trained machine learning model from a file using joblib.

    Parameters:
    - filename (str): Name of the file containing the pre-trained model.

    Returns:
    - loaded_model: Pre-trained machine learning model.
    """
    loaded_model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return loaded_model

# Visualize decision boundaries and actual data points
def visualize_decision_boundaries(model, X, y, df):
    """
    This function visualizes decision boundaries and actual data points based on the trained model.

    Parameters:
    - model: Trained machine learning model.
    - X (pandas.DataFrame): Features for visualization.
    - y (pandas.Series): Labels for visualization.
    - df (pandas.DataFrame): DataFrame containing synthetic data and labels.

    Returns:
    - fig (matplotlib.figure.Figure): A visual representation of decision boundaries and data points.
    """
    # Visualization of Decision Boundaries
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual data points
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis')
    plt.title('Actual Operation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot decision boundaries
    plt.subplot(1, 2, 2)
    h = .02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis', alpha=0.5)
    plt.title('Decision Boundaries and Operation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Class')

    return fig

# Evaluate the trained model on the test set and visualize actual vs predicted data
def evaluate_model(model, X_test, y_test, df):
    """
    This function evaluates the trained model on the test set and visualizes actual vs predicted data.

    Parameters:
    - model: Trained machine learning model.
    - X_test (pandas.DataFrame): Test features.
    - y_test (pandas.Series): Test labels.
    - df (pandas.DataFrame): DataFrame containing synthetic data and labels.

    Returns:
    - fig (matplotlib.figure.Figure): A visual representation of actual vs predicted data.
    """
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and display it
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    # Plot actual and predicted data points
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual data points
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis')
    plt.title('Actual Operation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Plot predicted data points
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis', alpha=0.5)
    sns.scatterplot(x=X_test['Feature_1'], y=X_test['Feature_2'], hue=y_pred, marker='X', s=100, palette='Set2', edgecolor='black')
    plt.title('Actual vs Predicted Operation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    return fig

# Main function to execute the entire workflow
def main():
    """
    This is the main function that orchestrates the entire workflow.
    It generates synthetic data, trains a decision tree classifier, and visualizes the results.
    """
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
    st.title("Decision Tree Visualization App")

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

    # Visualization of decision boundaries and operation data
    st.header("Decision Boundaries and Operation Data")
    vdb = visualize_decision_boundaries(loaded_model, X_test, y_test, df)
    st.pyplot(vdb)

    # Actual vs Predicted operation data
    st.header("Actual vs Predicted Operation Data")
    fig = evaluate_model(loaded_model, X_test, y_test, df)
    st.pyplot(fig)

# Entry point of the script, calling the main function
if __name__ == "__main__":
    main()
