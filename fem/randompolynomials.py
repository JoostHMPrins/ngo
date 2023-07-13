import numpy as np


def randompoly1DO3(c):
    
    def polynomial(x):
        poly = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
        return poly
        
    return polynomial


def randompoly2DO3(c):
    
    def polynomial(x,y):
        poly = c[0] + c[1]*x + c[2]*y + c[3]*x**2 + c[4]*y**2 + c[5]*x*y + c[6]*x**3 + c[7]*y**3 + c[8]*x*y**2 + c[9]*y*x**2
        return poly
    
    return polynomial


def randompoly1DO3sqr(c):
    
    def polynomial(x):
        poly = (c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3)**2
        return poly
        
    return polynomial


def randompoly2DO3sqr(c):
    
    def polynomial(x,y):
        poly = 0.1 + (c[0] + c[1]*x + c[2]*y + c[3]*x**2 + c[4]*y**2 + c[5]*x*y + c[6]*x**3 + c[7]*y**3 + c[8]*x*y**2 + c[9]*y*x**2)**2
        return poly
    
    return polynomial
