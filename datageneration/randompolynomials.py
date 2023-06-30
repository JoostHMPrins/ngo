import numpy as np


def randompoly1DO3(x):
    c = np.random.uniform(-0.1,0.1,4)
    poly = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
    return poly


def randompoly2DO3(x,y):
    c = np.random.uniform(-0.1,0.1,10)
    poly = c[0] + c[1]*x + c[2]*y + c[3]*x**2 + c[4]*y**2 + c[5]*x*y + c[6]*x**3 + c[7]*y**3 + c[8]*x*y**2 + c[9]*y*x**2
    return poly