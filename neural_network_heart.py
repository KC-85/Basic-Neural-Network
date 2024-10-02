import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Heart Disease dataset
data = pd.read_csv("heart.csv")

# Print unique values in each column to identify categorical features
print("Unique values in each column:")
for col in data.columns:
    print(f"{col}: {data[col].unique()}")

# Encode categorical columns to numeric using pd.get_dummies
# Columns to be encoded: "sex", "cp", "restecg", "thal"
data_encoded = pd.get_dummies(data, columns=["sex", "cp", "restecg", "thal"], drop_first=True)

# Update feature and target matrices
X = data_encoded.drop("num", axis=1).values  # Features
y = data_encoded["num"].values.reshape(-1, 1)  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, X):
        # Hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Output layer
        final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = sigmoid(final_input)
        
        return final_output

    def backward(self, X, y, output, learning_rate=0.01):
        # Calculate the error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass
            self.backward(X, y, output, learning_rate)

            # Print the loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch} Loss: {loss}")

    def predict(self, X):
        return self.forward(X)


# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize and train the neural network
nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=10, output_size=1)
nn.train(X_train, y_train, epochs=5000, learning_rate=0.01)

# Evaluate the model on the test set
predictions = nn.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary (0/1)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
