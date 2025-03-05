# Visualizing Bias Effect on Perceptron Decision Boundary

## Task Overview
Create an interactive visualization to demonstrate how the bias term affects a perceptron's decision boundary. This visualization serves as an educational tool to understand one of the fundamental concepts in neural networks and machine learning.

## Problem Statement
In machine learning, a perceptron's effectiveness in classification tasks heavily depends on its decision boundary. While the weights determine the angle of this boundary, the bias term enables the boundary to shift in space. Without bias, the decision boundary must pass through the origin (0,0), limiting the perceptron's classification capabilities.

## Approach

### 1. Data Generation
- Used sklearn's `make_blobs` to create synthetic dataset
- Generated two distinct classes with controlled separation
- Applied appropriate scaling and positioning for clear visualization

### 2. Interactive Visualization
- Implemented using Streamlit for web-based interaction
- Created a responsive two-column layout:
  * Left column: Controls (bias slider)
  * Right column: Dynamic visualization
- Added real-time updates of the decision boundary

### 3. User Interface Design
- Clean, minimal interface focusing on the visualization
- Intuitive bias slider with appropriate range (-5 to 5)
- Clear labeling and color coding for classes
- Informative grid and legend

### 4. Educational Components
- Comprehensive explanation section
- Clear documentation of the bias effect
- Visual demonstration of boundary movement

## Implementation Benefits
1. **Interactive Learning**
   - Real-time visualization of bias effects
   - Immediate feedback on parameter changes
   - Hands-on experimentation capability

2. **Clear Visualization**
   - Distinct class separation
   - Well-defined decision boundary
   - Intuitive color coding

3. **Educational Value**
   - Demonstrates key machine learning concept
   - Helps build intuition about perceptron behavior
   - Supports theoretical understanding with visual proof
