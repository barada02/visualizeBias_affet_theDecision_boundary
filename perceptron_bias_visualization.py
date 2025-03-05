import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set page config
st.set_page_config(page_title="Perceptron Decision Boundary Visualization", layout="wide")

# Create two columns
left_column, right_column = st.columns([1, 3])  # 1:3 ratio for left:right

# Left column - Controls and explanation
with left_column:
    st.title("Perceptron Decision Boundary with Adjustable Bias")
    st.write("""
    This visualization demonstrates how the bias term affects the decision boundary in a perceptron.
    Use the slider below to adjust the bias and observe how it shifts the decision boundary.
    """)
    
    # Initialize random seed for reproducibility
    np.random.seed(42)

    # Generate sample data
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)
    # Adjust the second class to make it more interesting
    X[y == 1] = X[y == 1] + [2, 2]

    # Initialize weights randomly
    weights = np.random.randn(2)

    # Create the bias slider
    bias = st.slider(
        'Adjust Bias',
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Move the slider to change the bias value and see how it affects the decision boundary"
    )
    
    st.markdown("""
    ### How to interpret this visualization:

    1. **Data Points**:
       - Blue dots represent Class 0
       - Red dots represent Class 1

    2. **Decision Boundary**:
       - The green line shows where the perceptron makes its decision
       - Points above the line are classified differently from points below it

    3. **Bias Effect**:
       - Moving the slider changes the bias value
       - Watch how the green line shifts up or down as you adjust the bias
       - This demonstrates how bias allows the decision boundary to move away from the origin
    """)

# Right column - Visualization
with right_column:
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data points
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', color='blue', alpha=0.5)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', color='red', alpha=0.5)

    # Calculate and plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x = np.linspace(x_min, x_max, 100)
    y = -(weights[0]*x + bias)/weights[1]

    ax.plot(x, y, 'g-', label=f'Decision Boundary\nBias = {bias:.2f}')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Keep the plot area fixed
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    # Display the plot in Streamlit
    st.pyplot(fig)
