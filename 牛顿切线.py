import math
import numpy as np # 数学库
import matplotlib.pyplot as plt  # 图形库

def f(x):
    L = 3*(x**4)-4*(x**3)-12*(x**2)
    return L
#函数的一阶导数
def f1(x):
    L = 12 * (x ** 3) - 12 * (x ** 2) - 24 * x
    return L
#函数的二阶导数
def f2(x):
    L = 36 * (x ** 2) - 24 * x -24
    return L
#a,b为区间的端点，t0为初始点,miu为终止限
#返回值中txing是最终的极值点，fxing是函数的极值
allt=[]
def Newton(a,b,t0,miu):
    i=0
    allt.append(t0)
    t=t0-f1(t0)/f2(t0)
    if abs(t-t0)<miu:
        txing=t0
        fxing=f(t0)
        return txing,fxing
    else:
        allt.append(t)
        Newton(a,b,allt[++i],miu)

re,fre,count=Newton(-2,-0.5,-1.2,0.01)
print('极小值点是：',re)
print('极小值是',fre)
print('迭代次数是',count)


'''
# 画图
x = 0
plt.title("Newton tangent method")#图形标题
x = np.linspace(-2,3)
plt.plot(x,f(x))
plt.show()
'''