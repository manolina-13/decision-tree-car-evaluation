# Decision Tree Car Evaluation

This project presents an end-to-end implementation of Decision Tree classification on the `car_evaluation.csv` dataset. The notebook covers data preparation, baseline model training, model evaluation, and performance improvement through hyperparameter tuning.

## Repository Name
`decision-tree-car-evaluation-lab`

## Project Objective
I build and evaluate a Decision Tree classifier for car acceptability prediction, and compare model behavior before and after hyperparameter optimization.

## Project Structure
- `decision_tree_car_evaluation.ipynb` — Main notebook with preprocessing, training, evaluation, and tuning.
- `README.md` — Project documentation and theoretical background.

## Theoretical Background

### 1) Decision Tree Classifier
A Decision Tree is a supervised learning algorithm that recursively splits the feature space into homogeneous regions. At each node, the algorithm selects the feature and split condition that best separates the classes according to an impurity criterion.

Common split criteria include:
- **Gini Impurity** (used in CART):
	$$Gini = 1 - \sum_{i=1}^{K} p_i^2$$
- **Entropy** (used in information gain based trees):
	$$Entropy = -\sum_{i=1}^{K} p_i \log_2(p_i)$$

Where $p_i$ is the proportion of samples of class $i$ in a node and $K$ is the number of classes.

### 2) Why Encoding Is Required
The car evaluation dataset is categorical. Since scikit-learn estimators operate on numeric inputs, categorical categories are encoded into numerical form before model training.

### 3) Overfitting vs Underfitting
- **Underfitting** occurs when the tree is too shallow and cannot capture meaningful patterns.
- **Overfitting** occurs when the tree is overly complex and learns noise from training data.

Controlling tree complexity through hyperparameters is essential for generalization.

### 4) Hyperparameter Tuning
Grid Search with cross-validation is used to select better hyperparameters by systematically evaluating combinations such as:
- `max_depth`
- `min_samples_leaf`
- `criterion` (`gini` / `entropy`)

This improves model robustness and helps balance bias-variance tradeoff.

## Methodology
1. Load and inspect `car_evaluation.csv`.
2. Encode categorical variables.
3. Split data into train/test sets.
4. Train a baseline Decision Tree model.
5. Evaluate using accuracy, confusion matrix, and classification report.
6. Perform hyperparameter tuning with `GridSearchCV`.
7. Re-evaluate and compare the tuned model against baseline.

## Evaluation Metrics
- **Accuracy**: overall proportion of correct predictions.
- **Confusion Matrix**: class-wise prediction distribution.
- **Classification Report**: precision, recall, and F1-score per class.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib

## Author
**Manolina Das**  

