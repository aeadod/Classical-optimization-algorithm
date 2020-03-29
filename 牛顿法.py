
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
def dampnm(fun, gfun, hess, x0):
    # 用牛顿法求解无约束问题
    # x0是初始点，fun，gfun和hess分别是目标函数值，梯度，海森矩阵的函数
    maxk = 500
    k = 0
    epsilon = 1e-2
    W = np.zeros((2, 20000))
    while k < maxk:
        W[:, k] = x0
        gk = gfun(x0)
        Gk = hess(x0)
        dk = -1 * np.linalg.solve(Gk, gk)
        if np.linalg.norm(dk) < epsilon:
            break
        x0 += dk
        k += 1
    W = W[:, 0:k + 1]  # 记录迭代点
    return x0, fun(x0), k, W

# 函数表达式fun
fun = lambda x: x[0]**2 +x[1]**2 -x[0]*x[1]- 10*x[0] -4*x[1]+60
# 梯度向量 gfun
gfun = lambda x: np.array([2*x[0]-x[1]-10,2*x[1]-x[0]-4])
# 海森矩阵 hess
hess = lambda x: np.array([[2,-1 ], [-1,2]])

if __name__ == "__main__":
    X1 = np.arange(-2, 10, 0.05)
    X2 = np.arange(-2, 10, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = x1**2+x2**2-x1*x2-10*x1-4*x2+60  # 给定的函数
    plt.contour(x1, x2, f, 40)  # 画出函数的轮廓线
    x0 = np.array([0.0, 0.0])
    out = dampnm(fun, gfun, hess, x0)
    print(out[1])
    W = out[3]
    plt.plot(W[0, :], W[1, :], 'g*-')
    plt.show()
    def testf(x):
        L =x[0]**2+x[1]**2-x[0]*x[1]-10*x[0]-4*x[1]+60
        return L
    min1 = fmin(testf,[0.0,0.0])
    print('经过验证得到的根为：', min1[0])