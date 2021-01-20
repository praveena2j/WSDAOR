import numpy as np
from numpy import isnan, nan, logical_not, logical_or

def pearson(x, y):
    # Assume len(x) == len(y)
    n = len(x)
    print(n)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))
    psum = sum(list(map(lambda x, y: x * y, x, y)))
    num = psum - (sum_x * sum_y / n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0:
        # print(x)
        # print(y)
        ans = len(list(set(x).intersection(y))) / len(x)
        # print(ans)
        return ans
    return num / den
