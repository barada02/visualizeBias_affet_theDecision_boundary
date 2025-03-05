# Visualizing Bias Effect on Perceptron Decision Boundary

## Task Overview
The task involves creating a visualization to demonstrate how the bias term affects the decision boundary in a perceptron classifier. This visualization is crucial for understanding one of the fundamental concepts in neural networks and machine learning.

## Problem Statement
In machine learning, a perceptron's ability to correctly classify data points depends heavily on its decision boundary. While the weights determine the orientation of this boundary, the bias term allows the boundary to shift in space. Without bias, the decision boundary must pass through the origin (0,0), which limits the perceptron's classification capabilities.

## Approach

### 1. Data Generation
- Created synthetic dataset with two distinct classes using sklearn's `make_blobs`
- Ensured clear separation between classes for better visualization
- Added controlled randomness for realistic data distribution

### 2. Visualization Strategy
- Implemented an interactive matplotlib plot
- Created a slider interface for real-time bias adjustment
- Color-coded data points for clear class distinction
- Added dynamic decision boundary updating

### 3. User Interaction
- Provided bias range from -5 to 5 for comprehensive exploration
- Included real-time boundary updates for immediate feedback
- Added grid and legends for better interpretation

## Implementation Benefits
1. Interactive Learning: Users can directly observe the relationship between bias and decision boundary
2. Clear Visualization: Distinct color coding and clean layout enhance understanding
3. Real-time Updates: Immediate feedback helps in grasping the concept quickly
