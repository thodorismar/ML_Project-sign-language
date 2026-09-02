# ML_Project-sign-language
A machine learning project focused on multi-class image classification using the Sign-MNIST dataset for my first project in the Machine Learning course of my university. The objective is to identify 24 static American Sign Language (ASL) hand gestures from $28 \times 28$ grayscale images. 

Overview
Dataset & Preprocessing: Evaluated on 27,455 samples across 24 classes (letters A–Y, excluding motion-based J and Z). Preprocessing includes $[0, 1]$ min-max pixel normalization, 784-dimensional vector flattening, and an 80/20 stratified train-validation split.  

Model Benchmarking & Tuning: Systematically evaluated several supervised learning algorithms using Accuracy and Macro-F1 scores:  
k-Nearest Neighbors (kNN): Explored neighbor counts ($k \in \{1, 3, 5, 7, 11\}$) and distance metrics (Euclidean, Manhattan, Cosine).  
Support Vector Machines (SVM): Benchmarked Linear and RBF kernels across various $C$ and $\gamma$ values.  
Decision Trees & Random Forests: Evaluated tree depth, leaf limits, splitting criteria (Gini, Entropy), and ensemble sizes.  
AdaBoost & Multi-Layer Perceptrons (MLP): Tested decision-stump boosting and deep fully connected architectures.  
Best Performing Model: A kNN classifier ($k=3$, Euclidean distance) achieved the best balance of accuracy and generalization ($\approx 99.8\%$ Accuracy and Macro-F1) without overfitting, outperforming complex models in inference speed and stability.  
Custom CNN Implementation: A from-scratch convolutional neural network (SimpleCNN) implemented in pure NumPy and integrated with scikit-learn estimators, showcasing manual implementations of 2D cross-correlation, ReLU activation, and max-pooling.  

Tech Stack
Language: Python  
Libraries: scikit-learn, NumPy, Pandas 
