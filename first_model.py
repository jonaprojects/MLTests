from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import random

def plot_validation_and_predictions(val_X, val_y, prediction_y):
    plt.plot(val_X, val_y)
    plt.plot(val_X, prediction_y)
    plt.legend(["Actual", "Predictions"])
    plt.show()

def scatter_validation_and_predictions(val_X, val_y, prediction_y):
    plt.scatter(val_X, val_y)
    plt.scatter(val_X, prediction_y)
    plt.legend(["Actual", "Predictions"])
    plt.show()


def train_model():
    values_range = np.linspace(-10, 10, 120)
    X = np.array([x for x in values_range], dtype='float64')
    y = np.array([np.sin(x) + random.uniform(-0.2, 0.2) for x in values_range], dtype='float64')
    train_X, val_X, train_y, val_y = train_test_split(X, y)
    print("Managed to split the data")
    #tree_model = DecisionTreeRegressor(random_state=1, max_depth=7)
    #tree_model.fit(train_X.reshape(-1, 1), train_y.reshape(-1, 1))

    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X.reshape(-1, 1), train_y.reshape(-1, 1))
    accuracy = round(forest_model.score(val_X.reshape(-1, 1), val_y.reshape(-1, 1)) * 100, 3)
    print(f"The model's accuracy is {accuracy}%")
    val_predictions = forest_model.predict(val_X.reshape(-1, 1))
    print(val_y[:10])
    print(val_predictions[:10])
    scatter_validation_and_predictions(val_X, val_y, val_predictions)

def main():
    train_model()


if __name__ == '__main__':
    main()
