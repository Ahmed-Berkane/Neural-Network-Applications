# Neural-Network-Applications
This repo is designed to apply diffrent NN aspects 

# Neural-Network-Applications

This project is designed to apply different aspects of Neural Networks, including building a custom logistic regression model and comparing it with scikit-learn implementations. The repo also demonstrates modular design for scalability, testing, and reproducibility.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Data Source](#data-source)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
4. [Modeling](#modeling)  
    - [Model Selection](#model-selection)  
5. [Setup](#setup)  
6. [Project Structure](#project-structure)  
7. [Features](#features)  

---

### 1. Project Overview
   - **Objective**: Implement a logistic regression model using neural network principles.  
   - **Key Components**: Forward propagation, cost function optimization, gradient descent.  
   - **Outcome**: Compare performance of custom NN logistic regression against scikit-learn’s implementation.  

### 2. Data Source
   - Dataset: Synthetic/generated datasets for binary classification tasks.  
   - Target: Binary outcome (0 or 1).  
   - Features: Multiple independent variables simulating classification problems.  

### 3. Exploratory Data Analysis (EDA)
   - Included steps:
      - Data cleaning and preprocessing.  
      - Feature scaling and transformation.  
      - Visualization of class distribution and feature relationships.  

### 4. Modeling
   - Models implemented:
      - Custom Neural Network Logistic Regression (from scratch in NumPy).  
      - Scikit-learn Logistic Regression.  
   - Model comparison:
      - Evaluated accuracy, precision, recall, and F1 score.  
      - Confirmed alignment between custom and sklearn implementations.  

#### Model Selection
   - Final Model: **Custom NN Logistic Regression**  
   - Chosen to demonstrate mathematical foundations of logistic regression and gradient-based learning.  

### 5. Setup
   - **Clone the repository**:
     ```bash
     git clone https://github.com/Ahmed-Berkane/Neural-Network-Applications.git
     ```
   - **Create a virtual environment**:
     ```bash
     python3 -m venv venv

     # On Windows:
     .\venv\Scripts\activate 

     # On macOS/Linux:
     source venv/bin/activate
     ```
   - **Install dependencies**:
     ```bash
     cd Neural-Network-Applications
     pip install -r requirements.txt
     ```
   - **Run the training script**:
     ```bash
     python src/train.py
     ```

---

### 6. Project Structure
Neural-Network-Applications/
│── data/ # Dataset files
│── src/ # Data prep, models, training scripts
│── tests/ # Unit tests
│── requirements.txt # Dependencies





---

### 7. Features
- Custom Neural Network Logistic Regression implementation (NumPy).  
- Scikit-learn Logistic Regression for benchmarking.  
- Modular project structure for scalability.  
- Unit tests with pytest for reproducibility.  
- Extensible design to add more NN architectures in the future.  
── README.md # Project documentation