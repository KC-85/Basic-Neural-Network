import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# 🔄 Analyze Data Distribution Function
def analyze_data_distribution(y_train, y_test):
    print("\n💡 Training Data Distribution:")
    print(pd.Series(y_train.flatten()).value_counts())
    print("\n💡 Test Data Distribution:")
    print(pd.Series(y_test.flatten()).value_counts())


# 🔄 Check for Missing Values
def check_missing_values(data):
    print("\n🔍 Checking for missing values...")
    missing_values = data.isna().sum()
    print(missing_values[missing_values > 0])  # Display columns with NaNs


# 🔄 Data Preprocessing Function
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    print("📂 Columns in the original dataset:", data.columns)
    print("🔍 Inspecting columns for non-numeric values:")
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    for col in non_numeric_columns:
        unique_values = data[col].unique()
        print(f"Column '{col}' has non-numeric values: {unique_values[:5]}... (Total Unique: {len(unique_values)})")

    # Drop non-relevant columns
    columns_to_drop = ["location", "hospital_name", "patient_name"]
    for col in columns_to_drop:
        if col in data.columns:
            print(f"❌ Dropping non-relevant column: {col}")
            data = data.drop([col], axis=1)

    # One-hot encode categorical features
    columns_to_encode = ["sex", "cp", "restecg", "thal"]
    data_encoded = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)

    # Separate features and target
    X = data_encoded.drop("num", axis=1).values
    y = data_encoded["num"].values.reshape(-1, 1)

    # Ensure all features are numeric before imputation
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values

    # Impute missing values using the mean strategy
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Define Activation Functions and Their Derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Neural Network Class with Two Hidden Layers and Correct Matrix Shapes
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Initialize weights and biases with correct dimensions
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size1)  # Shape: (input_size, hidden_size1)
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2)  # Shape: (hidden_size1, hidden_size2)
        self.weights_hidden2_output = np.random.randn(hidden_size2, output_size)  # Shape: (hidden_size2, output_size)

        # Initialize biases
        self.bias_hidden1 = np.zeros(hidden_size1)
        self.bias_hidden2 = np.zeros(hidden_size2)
        self.bias_output = np.zeros(output_size)

    def forward(self, X):
        """Forward pass through the network."""
        self.hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_output1 = relu(self.hidden_input1)

        self.hidden_input2 = np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden_output2 = relu(self.hidden_input2)

        final_input = np.dot(self.hidden_output2, self.weights_hidden2_output) + self.bias_output
        final_output = sigmoid(final_input)
        return final_output

    def backward(self, X, y, output, learning_rate=0.001):
        """Backward pass for updating weights and biases."""
        # Calculate errors and deltas for each layer
        output_error = y - output  # Shape should be (samples, 1)
        output_delta = output_error * sigmoid_derivative(output)  # Shape: (samples, 1)

        # Calculate hidden layer errors and deltas
        hidden_error2 = np.dot(output_delta, self.weights_hidden2_output.T)  # Shape should be (samples, hidden_size2)
        hidden_delta2 = hidden_error2 * relu_derivative(self.hidden_output2)

        hidden_error1 = np.dot(hidden_delta2, self.weights_hidden1_hidden2.T)  # Shape should be (samples, hidden_size1)
        hidden_delta1 = hidden_error1 * relu_derivative(self.hidden_output1)

        # Update the weights and biases
        self.weights_hidden2_output += np.dot(self.hidden_output2.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate

        self.weights_hidden1_hidden2 += np.dot(self.hidden_output1.T, hidden_delta2) * learning_rate
        self.bias_hidden2 += np.sum(hidden_delta2, axis=0) * learning_rate

        self.weights_input_hidden1 += np.dot(X.T, hidden_delta1) * learning_rate
        self.bias_hidden1 += np.sum(hidden_delta1, axis=0) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.001):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)



# Evaluate Model Function
def evaluate_model(nn, X_test, y_test):
    predictions = nn.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions == y_test)
    print(f"\n💡 Test Accuracy: {accuracy * 100:.2f}%")


# Main Script
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data("heart.csv")
    analyze_data_distribution(y_train, y_test)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    nn = NeuralNetwork(input_size=X_train_balanced.shape[1], hidden_size1=20, hidden_size2=10, output_size=1)
    nn.train(X_train_balanced, y_train_balanced, epochs=5000, learning_rate=0.001)
    evaluate_model(nn, X_test, y_test)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_balanced, y_train_balanced.flatten())
    log_reg_accuracy = log_reg.score(X_test, y_test)
    print(f"💡 Logistic Regression Test Accuracy: {log_reg_accuracy * 100:.2f}%")
