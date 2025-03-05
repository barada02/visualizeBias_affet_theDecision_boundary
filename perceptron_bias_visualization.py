import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set page config
st.set_page_config(
    page_title="Perceptron Decision Boundary Visualization",
    layout="wide"
)

# Create two columns for the main content
left_column, right_column = st.columns([1, 3])  # 1:3 ratio for left:right

# Left column - Controls
with left_column:
    st.title("Controls")
    
    st.write("""
    Adjust the bias value using the slider below to see how it affects the decision boundary.
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
        help="Move the slider to change the bias value"
    )

# Right column - Visualization
with right_column:
    st.title("Visualization")
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))

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

    # Display the plot
    st.pyplot(fig)

# Interpretation section below the main content
st.markdown("""
---
## How to Interpret This Visualization

This visualization demonstrates how the bias term affects the decision boundary in a perceptron classifier.

### Key Components

#### 1. Data Points
- **Blue dots** (Class 0): First group of data points
- **Red dots** (Class 1): Second group of data points
- The two classes are clearly separated for better visualization

#### 2. Decision Boundary (Green Line)
- Represents where the perceptron makes its classification decision
- Points on different sides of this line are classified into different classes
- The line's position changes as you adjust the bias

#### 3. Bias Effect
- **Moving the slider left** (negative bias): Shifts the boundary upward
- **Moving the slider right** (positive bias): Shifts the boundary downward
- This demonstrates how bias allows the decision boundary to move away from the origin (0,0)

### Why This Matters
The bias term gives the perceptron more flexibility in finding the optimal decision boundary. Without bias, the boundary would be forced to pass through the origin, which might not be optimal for separating the classes.
""")
