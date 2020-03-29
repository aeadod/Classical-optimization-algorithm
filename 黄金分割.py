from scipy.optimize import fminbound

def f(x):
    L = 3*(x**4)-4*(x**3)-12*(x**2)
    return L

def golden(f ,*args):
    if not('a' in dir()):
        a=[]
        b=[]
    a.append(args[0])
    b.append(args[1])
    L=0.001  #最终精确度，此处定义为0.001
    n=18  #迭代最大次数  17/18
    lambda1=a[0]+0.382*(b[0]-a[0])
    miu1=a[0]+0.618*(b[0]-a[0])
    #判断是否满足准确度要求
    for k in range(0,n):
        if abs(b[k]-a[k])<=0.001:
            solve=(a[k]+b[k])/2
            break
        f_lambda1=f(lambda1)
        f_miu1=f(miu1)
        #当f_lambda1>f_miu1时，去除区间[a,lambda1]
        if f_lambda1>f_miu1:
            a.append(lambda1)
            b.append(b[k])
            lambda2=miu1
            miu2=a[k+1]+0.618*(b[k+1]-a[k+1])
        # 当f_lambda1<f_miu1时，去除区间[miu1,b]
        else:
            a.append(a[k])
            b.append(miu1)
            miu2=lambda1
            lambda2=a[k+1]+0.382*(b[k+1]-a[k+1])
        lambda1=lambda2
        miu1=miu2
    print('黄金分割法求得结果是：',solve)
    return solve

golden(f,-2,0)
min_global1=fminbound(f,-2,0)
print('验证结果为：',min_global1)
golden(f,0,3)
min_global2=fminbound(f,0,3)
print('验证结果为：',min_global2)

