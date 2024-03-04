import numpy as np

def is_sorted(arr):
    return np.array_equal(arr, np.sort(arr))

class LinearInterpolation:
    def __init__(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.check_input()

    def check_input(self):
        if len(self.X) != len(self.Y):
            raise ValueError("X and Y must have the same length.")
        if not is_sorted(self.X):
            raise ValueError(f"X must be sorted. Found: {self.X}")
    
    def __call__(self, x) -> np.Any:
        return self.interpolate(x)

    def interpolate(self, x):
        idx = np.searchsorted(self.X, x)
        idx = np.clip(idx, 1, len(self.X) - 1)  # Ensure idx is within bounds

        x0, x1 = self.X[idx - 1], self.X[idx]
        y0, y1 = self.Y[idx - 1], self.Y[idx]

        slope = (y1 - y0) / (x1 - x0)
        interpolated_y = y0 + slope * (x - x0)
        return interpolated_y

    def derivative(self, x):
        idx = np.searchsorted(self.X, x)
        idx = np.clip(idx, 1, len(self.X) - 1)  # Ensure idx is within bounds

        x0, x1 = self.X[idx - 1], self.X[idx]
        y0, y1 = self.Y[idx - 1], self.Y[idx]

        slope = (y1 - y0) / (x1 - x0)
        return slope

# class LinearRegression:
#     def __init__(self, X, Y):
#         self.X = np.asarray(X)
#         self.Y = np.asarray(Y)
#         self.check_input()

#     def check_input(self):
#         if len(self.X) != len(self.Y):
#             raise ValueError("X and Y must have the same length.")

#     def fit(self):
#         # Compute linear regression parameters (slope and intercept)
#         self.slope, self.intercept = np.polyfit(self.X, self.Y, 1)

#     def interpolate(self, x):
#         # Linear interpolation function
#         return self.slope * x + self.intercept

#     def derivative(self, x):
#         # Derivative of the linear regression function (constant slope)
#         return self.slope