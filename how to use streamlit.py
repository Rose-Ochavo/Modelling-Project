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

def create_dataframe(data, num_dimensions):
    column_names = [f'Feature_{i+1}' for i in range(num_dimensions)] + ['Class']
    return pd.DataFrame(data, columns=column_names)

def visualize_synthetic_data(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Feature_1', y='Feature_2', data=df, color='blue', palette='viridis')
    ax.set_title('Synthetic Data Distribution')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return fig

def main():
    set_random_seed()

    class_properties = {
        0: {'mean': [25, 30], 'std': [5, 4]},
        1: {'mean': [40, 45], 'std': [6, 5]},
        2: {'mean': [55, 60], 'std': [4, 3.5]}
    }

    data = generate_synthetic_data(class_properties)
    df = create_dataframe(data, num_dimensions=2)

    # Use Streamlit to display the plot
    fig = visualize_synthetic_data(df)
    st.pyplot(fig)

if __name__ == "__main__":
    main()