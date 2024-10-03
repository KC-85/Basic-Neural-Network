import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to inspect non-numeric columns
def inspect_non_numeric_columns(data):
    print("\n🔍 Inspecting columns for non-numeric values...")
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    for col in non_numeric_columns:
        unique_values = data[col].unique()
        print(f"Column '{col}' has non-numeric values: {unique_values[:5]}... (Total Unique: {len(unique_values)})")

# Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    print("📂 Columns in the original dataset:", data.columns)
    inspect_non_numeric_columns(data)
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    if len(non_numeric_columns) > 0:
        print(f"⚠️ Detected non-numeric columns: {list(non_numeric_columns)}.")
        for col in non_numeric_columns:
            if col.lower() in ["location", "hospital_name", "patient_name"]:
                print(f"❌ Dropping column: {col}")
                data = data.drop([col], axis=1)
            else:
                print(f"🔄 Encoding column: {col}")
                data = pd.get_dummies(data, columns=[col], drop_first=True)
    columns_to_encode = ["sex", "cp", "restecg", "thal"]
    existing_columns = [col for col in columns_to_encode if col in data.columns]
    if len(existing_columns) > 0:
        data_encoded = pd.get_dummies(data, columns=existing_columns, drop_first=True)
    else:
        print("⚠️ No columns to encode. Proceeding without encoding.")
        data_encoded = data
    X = data_encoded.drop("num", axis=1).values
    y = data_encoded["num"].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = sigmoid(final_input)
        return final_output

    def backward(self, X, y, output, learning_rate=0.001):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs=15000, learning_rate=0.002):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

X_train, X_test, y_train, y_test = load_and_preprocess_data("heart.csv")
nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=40, output_size=1)
nn.train(X_train, y_train, epochs=5000, learning_rate=0.001)

predictions = nn.predict(X_test)
accuracy = np.mean((predictions > 0.5).astype(int) == y_test)
print(f"\n💡 Test Accuracy: {accuracy * 100:.2f}%")
