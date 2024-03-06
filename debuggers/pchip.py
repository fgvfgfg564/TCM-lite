import numpy as np


class PCHIPInterpolator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.slopes = self.compute_slopes(x, y)

    def compute_slopes(self, x, y):
        h = np.diff(x)
        delta = np.diff(y) / h
        delta_s = np.diff(delta)
        m = np.zeros_like(y)
        m[1:-1] = (delta_s[1:] * h[:-1] + delta_s[:-1] * h[1:]) / (h[:-1] + h[1:])
        m[0] = self._edge_slope(h[0], delta[0], delta_s[0])
        m[-1] = self._edge_slope(h[-1], delta[-1], delta_s[-1])
        return m

    def _edge_slope(self, h, delta, delta_s):
        if np.sign(delta) == np.sign(delta_s):
            return (3 * delta - 2 * delta_s) / h
        else:
            return np.zeros_like(delta)

    def __call__(self, x_new):
        y_new = np.interp(x_new, self.x, self.y)
        return y_new

    def derivative(self, x_new):
        slope_new = np.interp(x_new, self.x, self.slopes)
        return slope_new


# Example usage:
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

interpolator = PCHIPInterpolator(x, y)

x_new = np.linspace(0, 4, 100)
y_new = interpolator(x_new)
slope_new = interpolator.derivative(x_new)

print("Interpolated values:", y_new)
print("Slopes:", slope_new)
