import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

# Set page config
st.set_page_config(
    page_title="Perceptron Decision Boundary Visualization",
    layout="wide"
)

# Create two columns for the main content with adjusted ratio
left_column, right_column = st.columns([1, 2])  # Changed to 1:2 ratio

# Left column - Controls
with left_column:
    st.title("Controls")
    
    # Dataset selection
    dataset_type = st.selectbox(
        'Select Dataset',
        ['Height-Weight Example', 'Test Scores Example', 'Nonlinear Example'],
        help="Choose different realistic datasets to visualize"
    )
    
    st.write("""
    Adjust the bias value using the slider below to see how it affects the decision boundary.
    """)
    
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
    
    # Initialize random seed
    np.random.seed(42)
    
    # Generate different types of realistic datasets
    if dataset_type == 'Height-Weight Example':
        # Simulate height-weight data for two age groups
        n_samples = 100
        # Group 1: Teenagers (shorter, lighter)
        X1 = np.random.normal(loc=[160, 55], scale=[5, 7], size=(n_samples//2, 2))
        # Group 2: Adults (taller, heavier)
        X2 = np.random.normal(loc=[175, 75], scale=[8, 10], size=(n_samples//2, 2))
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        feature_names = ['Height (cm)', 'Weight (kg)']
        class_names = ['Teenagers', 'Adults']
        
    elif dataset_type == 'Test Scores Example':
        # Simulate test scores for pass/fail students
        n_samples = 100
        # Failed students (lower scores with more variance)
        X1 = np.random.normal(loc=[60, 55], scale=[15, 12], size=(n_samples//2, 2))
        # Passed students (higher scores with less variance)
        X2 = np.random.normal(loc=[85, 80], scale=[10, 8], size=(n_samples//2, 2))
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        # Clip scores to be between 0 and 100
        X = np.clip(X, 0, 100)
        feature_names = ['Math Score', 'Science Score']
        class_names = ['Failed', 'Passed']
        
    else:  # Nonlinear Example
        # Create a more complex, nonlinear dataset
        X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
        X = X * 2  # Scale up the data for better visualization
        feature_names = ['Feature 1', 'Feature 2']
        class_names = ['Class A', 'Class B']
    
    # Initialize weights randomly
    weights = np.random.randn(2)
    
    # Create the visualization with smaller figure size
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.tight_layout()
    
    # Plot data points
    scatter0 = ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], 
                         label=class_names[0], color='blue', alpha=0.6)
    scatter1 = ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
                         label=class_names[1], color='red', alpha=0.6)
    
    # Calculate and plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x = np.linspace(x_min, x_max, 100)
    y_boundary = -(weights[0]*x + bias)/weights[1]
    
    line = ax.plot(x, y_boundary, 'g-', label=f'Decision Boundary\nBias = {bias:.2f}')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Keep the plot area fixed
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    
    # Display the plot with custom width
    st.pyplot(fig, use_container_width=True)

# Interpretation section below the main content
st.markdown(f"""
---
## How to Interpret This Visualization

This visualization demonstrates how the bias term affects the decision boundary in a perceptron classifier using {dataset_type}.

### Dataset Description
- **{class_names[0]}** (Blue dots): {
    "Lower height and weight measurements" if dataset_type == 'Height-Weight Example'
    else "Students with generally lower test scores" if dataset_type == 'Test Scores Example'
    else "First group in the nonlinear pattern"}
- **{class_names[1]}** (Red dots): {
    "Higher height and weight measurements" if dataset_type == 'Height-Weight Example'
    else "Students with generally higher test scores" if dataset_type == 'Test Scores Example'
    else "Second group in the nonlinear pattern"}

### Key Components

#### 1. Data Points
- Each point represents {
    "a person's height and weight measurements" if dataset_type == 'Height-Weight Example'
    else "a student's math and science scores" if dataset_type == 'Test Scores Example'
    else "a sample in the feature space"}
- The two colors show the different classes
- Notice the natural overlap and variation in the data

#### 2. Decision Boundary (Green Line)
- The line shows where the perceptron makes its classification decision
- Points on different sides are classified into different groups
- The line's position changes as you adjust the bias

#### 3. Bias Effect
- **Moving the slider left** (negative bias): Shifts the boundary upward
- **Moving the slider right** (positive bias): Shifts the boundary downward
- This shows how bias helps find the optimal separation between classes

### Why This Matters
The bias term allows the perceptron to better adapt to real-world data patterns where the optimal separation line may not pass through the origin (0,0).
""")
