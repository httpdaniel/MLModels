import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as SKLinearRegression
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_squared_error as sk_mse,
    root_mean_squared_error as sk_rmse,
    mean_absolute_error as sk_mae,
    r2_score as sk_r2,
)
from metrics.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r_squared,
)


class LinearRegression:
    def __init__(self):
        self.beta_0 = 0
        self.beta_1 = 0

    def fit(self, X, y):
        m = len(y)

        learning_rate = 0.02
        iterations = 2000

        for _ in range(iterations):
            predictions = self.predict(X)

            gradient1 = (1 / m) * sum(predictions - y)
            gradient2 = (1 / m) * sum((predictions - y) * X)

            self.beta_0 -= learning_rate * gradient1
            self.beta_1 -= learning_rate * gradient2

    def predict(self, X):
        return self.beta_0 + self.beta_1 * X


if __name__ == "__main__":
    df = pd.read_csv("../data/Salary_dataset.csv")

    X = np.array(df["YearsExperience"])
    y = np.array(df["Salary"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers"))
    fig.update_layout(
        title="Scatter Plot", xaxis_title="Salary", yaxis_title="YearsExperience"
    )
    fig.show()

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    print("---My Model---")
    print(f"Beta 0: {model.beta_0}")
    print(f"Beta 1: {model.beta_1}")

    print(f"MSE: {mean_squared_error(y, predictions)}")
    print(f"RMSE: {root_mean_squared_error(y, predictions)}")
    print(f"MAE: {mean_absolute_error(y, predictions)}")
    print(f"R Squared: {r_squared(y, predictions)}")

    print("\n")

    sk_model = SKLinearRegression()
    sk_model.fit(X.reshape(-1, 1), y)
    sk_predictions = sk_model.predict(X.reshape(-1, 1))

    print("---SKLearn Model---")
    print(f"Sklearn Beta 0: {sk_model.intercept_}")
    print(f"Sklearn Beta 1: {sk_model.coef_[0]}")

    print(f"MSE: {sk_mse(y, sk_predictions)}")
    print(f"RMSE: {sk_rmse(y, sk_predictions)}")
    print(f"MAE: {sk_mae(y, sk_predictions)}")
    print(f"R Squared: {sk_r2(y, sk_predictions)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers", name="Data Points"))
    fig.add_trace(go.Scatter(x=X, y=predictions, mode="lines", name="MyModel"))
    fig.add_trace(
        go.Scatter(
            x=X, y=sk_predictions, mode="lines", line=dict(dash="dash"), name="SKModel"
        )
    )
    fig.update_layout(
        title="Model Comparison", xaxis_title="Salary", yaxis_title="YearsExperience"
    )
    fig.show()
