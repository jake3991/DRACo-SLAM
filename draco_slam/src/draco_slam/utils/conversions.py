import gtsam

def Y(y):
    return gtsam.symbol("y", y)

def Z(z):
    return gtsam.symbol("z", z)