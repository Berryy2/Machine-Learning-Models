title: "🤖 Machine Learning Models — From Scratch"
description: >
  A collection of classic and modern Machine Learning models implemented **from scratch** using Python and NumPy.
  This repository is designed for learners and practitioners who want to deeply understand how ML algorithms work
  behind the scenes — without relying on high-level libraries like scikit-learn.

overview:
  motivation: >
    Most tutorials teach how to use machine learning libraries, but not how they actually work.
    This repository aims to bridge that gap by building each algorithm from first principles,
    including mathematical intuition, cost functions, optimization steps, and evaluation metrics.
  goals:
    - Reinforce core ML math and algorithmic intuition.
    - Implement models using only NumPy for computations.
    - Visualize performance and compare theoretical vs practical understanding.

structure:
  root_directory: "machine-learning-models/"
  contents:
    - linear_regression/
    - logistic_regression/
    - k_means/
    - decision_tree/
    - support_vector_machine/
    - neural_network/
  notes: >
    Each folder contains:
      - Model implementation (.py file)
      - Test/demo script (.py)
      - Model-specific README.yml (theoretical explanation + math + usage)

implemented_models:
  - name: "Linear Regression"
    status: "✅ Completed"
    key_topics:
      - Hypothesis function
      - Gradient Descent
      - Mean Squared Error (MSE)
  - name: "Logistic Regression"
    status: "🚧 In Progress"
    key_topics:
      - Sigmoid Function
      - Binary Cross-Entropy
      - Gradient Updates
  - name: "K-Means Clustering"
    status: "🧩 Planned"
    key_topics:
      - Centroid Initialization
      - Inertia Minimization
      - Iterative Optimization

technologies:
  languages: ["Python 3.x"]
  libraries_used: ["NumPy", "Matplotlib"]
  optional_tools: ["scikit-learn (for dataset generation only)"]

how_to_run:
  requirements:
    - Python 3.10+
    - Install dependencies:
      command: "pip install numpy matplotlib scikit-learn"
  example_usage: |
    # Clone repository
    git clone https://github.com/<your-username>/machine-learning-models.git
    cd machine-learning-models/linear_regression

    # Run model
    python Linear_Regression_test.py

future_plans:
  - Implement Logistic Regression with gradient descent.
  - Add evaluation metrics (R², MAE, Accuracy, etc.).
  - Build neural network from scratch with backpropagation.
  - Visualize learning process using Matplotlib animations.

learning_resources:
  - "Andrew Ng — Machine Learning (Coursera)"
  - "Sebastian Raschka — Python Machine Learning"
  - "Aurelien Geron — Hands-On Machine Learning"
  - "MIT OpenCourseWare — Introduction to Machine Learning"
  - "Wikipedia — Mathematical background for each model"

author:
  name: "Mohamed Maged Elsayed Ahmed Elberry"
  location: "Cairo, Egypt"
  email: "mohamed_berry210@hotmail.com"
  linkedin: "https://linkedin.com/in/mohamed-elberry"
  github: "https://github.com/<your-username>"
  interests: ["Artificial Intelligence", "Machine Learning", "Embedded Systems", "Digital Design"]

license:
  type: "MIT License"
  year: 2025
  holder: "Mohamed Elberry"

tags:
  - "machine learning"
  - "deep learning"
  - "python"
  - "from scratch"
  - "ai"
  - "gradient descent"
  - "data science"
