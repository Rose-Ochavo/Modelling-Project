# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data():
    class_properties = {
        0: {'mean': [25, 30], 'std': [5, 4]},
        1: {'mean': [40, 45], 'std': [6, 5]},
        2: {'mean': [55, 60], 'std': [4, 3.5]}
    }

    samples_per_class = 300
    num_dimensions = 2
    num_classes = len(class_properties)

    data = []
    for class_id in class_properties.keys():
        mean = class_properties[class_id]['mean']
        std = class_properties[class_id]['std']
        samples = np.random.normal(mean, std, size=(samples_per_class, num_dimensions))
        labels = np.full((samples_per_class, 1), class_id)
        class_data = np.hstack((samples, labels))
        data.append(class_data)

    data = np.vstack(data)
    np.random.shuffle(data)

    column_names = [f'Feature_{i+1}' for i in range(num_dimensions)] + ['Class']
    df = pd.DataFrame(data, columns=column_names)

    return df

def train_decision_tree(df):
    X = df.iloc[:, :-1]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test

def visualize_synthetic_data(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Feature_1', y='Feature_2', data=df, color='blue', palette='viridis')
    plt.title('Synthetic Data Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    st.pyplot()

def visualize_decision_boundaries(clf, X, y):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis')
    plt.title('Actual Operation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(1, 2, 2)
    h = .02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis', alpha=0.5)

    st.pyplot()

def visualize_actual_vs_predicted(clf, X, y):
    y_pred = clf.predict(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis')
    plt.title('Actual Operation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Feature_1', y='Feature_2', hue='Class', data=df, palette='viridis', alpha=0.5)
    sns.scatterplot(x=X['Feature_1'], y=X['Feature_2'], hue=y_pred, marker='X', s=100, palette='Set2', edgecolor='black')
    plt.title('Actual vs Predicted Operation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    st.pyplot()

def main():
    st.title("Decision Tree Visualization App")

    df = generate_synthetic_data()

    visualize_synthetic_data(df)

    clf, X_test, y_test = train_decision_tree(df)

    visualize_decision_boundaries(clf, X_test, y_test)

    visualize_actual_vs_predicted(clf, X_test, y_test)

    st.write("Model Evaluation:")
    st.write(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))}")
    st.write("Classification Report:")
    st.write(classification_report(y_test, clf.predict(X_test)))

if __name__ == "__main__":
    main()
