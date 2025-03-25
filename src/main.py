import csv
import random


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = random.uniform(-1, 1)
        self.weights = None

    def init_weights(self, vectors):
        self.weights = [random.uniform(-1, 1) for _ in range(vectors)]

    def predict(self, x):
        net = sum(w_i * x_i for w_i, x_i in zip(self.weights, x)) - self.bias

        if net >= 0:
            return 1
        else:
            return 0

    def update_weights_bias(self, x, label):
        prediction = self.predict(x)
        self.weights = [w_i + self.learning_rate * (label - prediction) * x_i for w_i, x_i in zip(self.weights, x)]
        self.bias -= self.learning_rate * (label - prediction)

    def train(self, train_set, labels):
        vectors = len(train_set[0])
        self.init_weights(vectors)

        for _ in range(self.epochs):
            for x, label in zip(train_set, labels):
                self.update_weights_bias(x, label)


def load_files(filename):
    data, labels = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            *vectors, label = row
            data.append([float(x) for x in vectors])
            labels.append(1 if label == "Iris-versicolor" else 0)
    return data, labels


def user_input(perceptron):
    choice = input("Do you wanna continue for user input (y/n) : ")

    if choice == "y":
        print("Enter vectors seperated by spaces: ")
        u_input = list(map(float, input().strip().split()))
        prediction = perceptron.predict(u_input)
        print(f"Predicted Class: {'Iris-versicolor' if prediction == 1 else 'iris-virginica'}")

    else:
        exit(1)


if __name__ == "__main__":
    print("Enter Train_file and Test_file")
    train_file = input(": ")
    test_file = input(": ")
    train_data, train_labels = load_files(train_file)
    test_data, test_labels = load_files(test_file)

    perceptron = Perceptron(learning_rate=0.01, epochs=100)
    perceptron.train(train_data, train_labels)

    correct = sum(1 for x, label in zip(test_data, test_labels) if perceptron.predict(x) == label)

    accuracy = correct / len(test_data) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    user_input(perceptron)
