from scipy.optimize import fmin,fminbound
from sympy import *
import matplotlib.pyplot as plt
import numpy as np

def NewTon(f1,f2,s,maxiter):
    for i in range(maxiter):
        s = s - f1.subs(x,s)/f2.subs(x,s)
        print("经过 {0} 次迭代, 结果被更新为 {1}".format(i+1,s))
    return s

x = Symbol("x")
f=3*(x**4)-4*(x**3)-12*(x**2)
f1=12*(x**3)-12*(x**2)-24*x
f2=36*(x**2)-24*x-24
NewTon(f1,f2,-1.2, maxiter = 5)

def testf(x):
    L = 3*(x**4)-4*(x**3)-12*(x**2)
    return L
min1=fmin(testf,-1.2)
print('经过验证得到的根为：',min1[0])
NewTon(f1,f2,2.5, maxiter = 5)
min2=fmin(testf,2.5)
print('经过验证得到的根为：',min2[0])