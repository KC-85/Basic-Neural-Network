import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):
    """
    Load the dataset from a CSV file and preprocess it for training.
    This function handles encoding of categorical columns and data normalization.

    Args:
        filepath (str): Path to the CSV file containing the dataset.

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed feature and label matrices for training and testing.
    """
    # Load the Heart Disease dataset
    data = pd.read_csv(filepath)

    # Print the unique values in each column to identify categorical features
    print("\n🔍 Unique values in each column (Categorical columns identified):")
    for col in data.columns:
        print(f"{col}: {data[col].unique()}")

    # Encode categorical columns using pd.get_dummies
    data_encoded = pd.get_dummies(data, columns=["sex", "cp", "restecg", "thal"], drop_first=True)

    # Separate features and target
    X = data_encoded.drop("num", axis=1).values  # Features
    y = data_encoded["num"].values.reshape(-1, 1)  # Target (reshape to a column vector)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def sigmoid(x):
    """
    Sigmoid activation function.
    Args:
        x (ndarray): Input array.

    Returns:
        ndarray: Transformed array using sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function.
    Args:
        x (ndarray): Input array.

    Returns:
        ndarray: Derivative of sigmoid for each element in the input array.
    """
    return x * (1 - x)


class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize weights and biases for the network layers.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons (1 for binary classification).
        """
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, X):
        """
        Perform a forward pass through the network.
        Args:
            X (ndarray): Input feature matrix.

        Returns:
            ndarray: Network output after passing through all layers.
        """
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = sigmoid(final_input)
        return final_output

    def backward(self, X, y, output, learning_rate=0.01):
        """
        Perform backpropagation to update weights and biases.
        Args:
            X (ndarray): Input feature matrix.
            y (ndarray): True labels.
            output (ndarray): Predicted output.
            learning_rate (float): Learning rate for gradient descent.
        """
        # Calculate output and hidden layer errors
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases using the calculated deltas
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.01):
        """
        Train the network using the training data for a specified number of epochs.
        Args:
            X (ndarray): Input feature matrix for training.
            y (ndarray): True labels for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for gradient descent.
        """
        print("\n🚀 Starting Training...")
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

            # Print the loss every 1000 epochs for tracking progress
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.
        Args:
            X (ndarray): Input feature matrix.

        Returns:
            ndarray: Predicted labels.
        """
        return self.forward(X)


def evaluate_model(nn, X_test, y_test):
    """
    Evaluate the neural network on the test dataset.
    Args:
        nn (NeuralNetwork): Trained neural network.
        X_test (ndarray): Test feature matrix.
        y_test (ndarray): True labels for the test set.

    Prints:
        The test accuracy of the model.
    """
    predictions = nn.predict(X_test)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary (0/1)
    accuracy = np.mean(predictions == y_test)
    print(f"\n💡 Test Accuracy: {accuracy * 100:.2f}%")


# Main script to run the neural network
if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("heart.csv")

    # Initialize and train the neural network
    nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=10, output_size=1)
    nn.train(X_train, y_train, epochs=5000, learning_rate=0.01)

    # Evaluate the model on the test set
    evaluate_model(nn, X_test, y_test)
