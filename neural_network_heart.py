import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go


# 🔄 Activation Functions and Their Derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# 🔄 Dictionary to Select Activation Functions
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative)
}

# 🔄 Binary Cross-Entropy Loss Function and Its Derivative
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-9  # Prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


# 🔄 Neural Network Class with Optimizations and Flexibility
class NeuralNetwork:
    def __init__(self, layers, activation="leaky_relu"):
        print("🔍 Initializing Neural Network...")
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        self.activation = activation_functions[activation][0]
        self.activation_derivative = activation_functions[activation][1]
        
        # Initialize weights and biases using He Initialization
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]))
            self.biases.append(np.zeros((1, layers[i + 1])))

        print(f"✅ Network Initialized with Layers: {self.layers} and Activation: {activation}")

    def forward(self, X):
        print("🔄 Performing Forward Pass...")
        activations = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = np.dot(activations[-1], W) + b
            # Use sigmoid for the last layer (output layer), otherwise use chosen activation
            A = sigmoid(Z) if i == self.num_layers - 2 else self.activation(Z)
            activations.append(A)
        return activations

    def backward(self, X, y, activations, learning_rate=0.001):
        print("🔄 Performing Backward Pass...")
        dA = binary_cross_entropy_derivative(y, activations[-1])  # Loss derivative

        for i in reversed(range(self.num_layers - 1)):
            dZ = dA * (sigmoid_derivative(activations[i + 1]) if i == self.num_layers - 2 else self.activation_derivative(activations[i + 1]))
            dW = np.dot(activations[i].T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)

            # Gradient Clipping
            dW = np.clip(dW, -1, 1)
            db = np.clip(db, -1, 1)

            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

            # Calculate dA for the next layer (moving backwards)
            dA = np.dot(dZ, self.weights[i].T)

    def train(self, X, y, epochs=1000, learning_rate=0.001, verbose=True):
        print("🚀 Starting Training...")
        self.loss_history = []

        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations, learning_rate)
            loss = binary_cross_entropy(y, activations[-1])
            self.loss_history.append(loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
        print("✅ Training Completed!")

    def predict(self, X):
        return self.forward(X)[-1]


# 🔄 Visualization Function Using Plotly
def plot_loss_curve(loss_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Loss'))
    fig.update_layout(title='Loss Curve Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()


# 🔄 Load and Preprocess Data with Improved Debugging and Flexibility
def load_and_preprocess_data(filepath):
    print("🔄 Loading and Preprocessing Data...")
    try:
        data = pd.read_csv(filepath)
        print(f"✅ Data Loaded Successfully! Columns: {list(data.columns)}")

        # Identify all columns with object or categorical data types
        categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()

        # One-hot encode all categorical columns
        data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        # Separate features and target
        X = data_encoded.drop("num", axis=1).values
        y = data_encoded["num"].values.reshape(-1, 1)

        # Scale the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(f"✅ Data Preprocessing Completed! Shape: {X.shape}")
        return X, y
    except FileNotFoundError:
        print("❌ Error: Dataset file not found. Please check the file path.")
        return None, None
    except KeyError:
        print("❌ Error: Target column 'num' not found in the dataset. Please ensure the dataset has the correct columns.")
        return None, None
    except Exception as e:
        print(f"❌ An error occurred during preprocessing: {e}")
        return None, None



# 🔄 Main Script with Debugging and Parameter Flexibility
if __name__ == "__main__":
    print("🔄 Loading Dataset...")
    X, y = load_and_preprocess_data("heart.csv")

    if X is not None and y is not None:
        print("🔄 Initializing and Training Neural Network...")
        nn = NeuralNetwork(layers=[X.shape[1], 10, 5, 1], activation="leaky_relu")
        nn.train(X, y, epochs=1000, learning_rate=0.001)

        print("🔄 Plotting Loss Curve...")
        plot_loss_curve(nn.loss_history)
    else:
        print("❌ Terminating Script: Data Not Loaded Properly.")
