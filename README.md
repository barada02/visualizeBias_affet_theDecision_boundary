# Perceptron Decision Boundary Visualization

This project demonstrates how the bias term affects the decision boundary in a perceptron classifier using an interactive visualization built with Streamlit.

## Overview

The visualization shows:
- Two linearly separable classes of data points
- An adjustable decision boundary
- Interactive bias control
- Real-time updates of the boundary position

## Features

1. **Interactive Controls**
   - Bias slider (-5.0 to 5.0)
   - Real-time visualization updates
   - Clear visual feedback

2. **Visualization Components**
   - Blue dots: Class 0 data points
   - Red dots: Class 1 data points
   - Green line: Decision boundary
   - Grid for better reference

3. **Layout**
   - Split view with controls and visualization
   - Detailed interpretation guide
   - Responsive design

## Requirements

```
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
streamlit>=1.20.0
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run perceptron_bias_visualization.py
   ```

## Understanding the Visualization

### Data Points
- The visualization uses synthetic data created using `make_blobs`
- Two distinct classes are generated and slightly separated for clear visualization
- Each point represents a 2D feature vector

### Decision Boundary
- The green line represents the perceptron's decision boundary
- Points on different sides of this line are classified into different classes
- The boundary's position changes based on the bias value

### Bias Effect
- Negative bias: Shifts boundary upward
- Positive bias: Shifts boundary downward
- Zero bias: Boundary passes through origin (0,0)

## Code Structure

- `perceptron_bias_visualization.py`: Main application file
- `requirements.txt`: Required Python packages
- Uses Streamlit for the web interface
- Matplotlib for plotting
- NumPy for numerical operations
- Scikit-learn for data generation

## Educational Value

This visualization helps understand:
1. How bias affects perceptron's decision making
2. The importance of bias in linear classification
3. The relationship between bias and decision boundary position

## Note

This is the basic version in the main repository. A more advanced version with realistic datasets is available in a separate branch.
