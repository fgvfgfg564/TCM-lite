from src.async_ops import async_map

if __name__ == "__main__":

    def f(x):
        return x * x

    x = list(range(100))
    y = async_map(f, x, 8)
    print(y)
