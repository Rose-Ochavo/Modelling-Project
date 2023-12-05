# # streamlit_visualization.py
# import streamlit as st
# from test import visualize_synthetic_data, visualize_decision_boundaries, evaluate_model

# def main():
#     st.title("Decision Tree Visualization App")

#     # Call functions from your existing code to visualize data
#     st.header("Synthetic Data Distribution")
#     visualize_synthetic_data(df)

#     st.header("Decision Boundaries and Operation Data")
#     visualize_decision_boundaries()

#     st.header("Actual vs Predicted Operation Data")
#     evaluate_model()

# if __name__ == "__main__":
#     main()

import streamlit as st
from test import main  # Replace 'your_module' with the actual module name

def main():
    st.title("Decision Tree Classifier App")
    st.sidebar.header("Options")

    # Add any additional options or settings using st.sidebar

    run_model_checkbox = st.checkbox("Run Model")

    if run_model_checkbox:
        run_model()

def run_model():
    with st.spinner("Running the model... This may take a while."):
        main()

if __name__ == "__main__":
    main()
