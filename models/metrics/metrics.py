import math
import statistics


def mean_squared_error(y_true, y_pred):
    m = len(y_true)
    ssr = sum((y_true - y_pred) ** 2)
    mse = ssr / m

    return mse


def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    return rmse


def mean_absolute_error(y_true, y_pred):
    m = len(y_true)
    sum_of_absolutes = sum(abs(y_true - y_pred))
    mae = sum_of_absolutes / m

    return mae


def r_squared(y_true, y_pred):
    ssr = sum((y_true - y_pred) ** 2)

    y_mean = statistics.mean(y_true)
    sst = sum((y_true - y_mean) ** 2)

    r_squared = 1 - (ssr / sst)

    return r_squared
