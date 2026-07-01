import math

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
M = 1560  # 整车质量
v1 = 20  # 被控车辆速度
v2 = 10  # 障碍车速度
Ll = 4.5  # 车身长度
Lw = 1.5  # 车身宽度
Lr = 12  # 道路宽度
Fm = 3000  # 单车轮最大制动力
Xl = 2  # 左车道中心线
Xr = 6  # 右车道中心线
Pm = 0.5  # 道路中心危险势能
Pt = 0.01  # 危险势能切换阈值
# X = 10  # 车辆位置
# Y = 15
X0 = 2  # 障碍物位置
Y0 = 5
X1 = 6  # 障碍物位置
Y1 = 90
Ds = 140
Xg=0
Yg=70
x = np.arange(0, Lr + 0.1, 0.1)
y = np.arange(0, Ds + 0.1, 0.1)
XX, YY = np.meshgrid(x, y)
m, n = XX.shape
ZZ = np.zeros((m, n))

for i in range(m):
    for j in range(n):
        X = XX[i, j]
        Y = YY[i, j]

        # 势场计算
        C=(v1*v1-v2*v2)
        Db = M*C/10/Fm+Ll / 2  # 车辆位置到障碍物的距离

        Dt = Db + 10

        c2 = -np.log(Pt) / Db ** 2

        if abs(Y - Y0) <= Db:
            c1 = -4 * (np.log(Pt) + c2 * (Y - Y0) ** 2) / (
                    (Lw * (np.sin((Y - Y0 - Db) / Db) * np.pi - np.pi / 2)) + 2) ** 2 * 0.4
        else:
            c1 = 0

        Db1 = M*C/10/Fm+Ll / 2  # 车辆位置到障碍物的距离
        Dt1 = Db1 + 10

        c21 = -np.log(Pt) / Db1 ** 2

        if abs(Y - Y1) <= Db1:
            c11 = -4 * (np.log(Pt) + c2 * (Y - Y1) ** 2) / (
                    (Lw * (np.sin((Y - Y1 - Db1) / Db1) * np.pi - np.pi / 2)) + 2) ** 2 * 0.4
        else:
            c11 = 0

        # 道路危险势场
        Ax = np.exp(-(6 - abs(X - 6)) ** 2)  # 道路边界的，像X=0或者x=8时AX最大，0和8就是边界
        Az = np.exp(-(X - 2) ** 2)  # 道路中间边界的，像X等于4最大，四就是道路中间边界
        Az1 = np.exp(-(X - 6) ** 2)

        if abs(Y - Y0) <= Db:
            Ay = 0
        elif abs(Y - Y0) >= Db and abs(Y - Y0) <= Dt:
            Ay = (abs(Y - Y0) - Db) / (Dt - Db)
        else:
            Ay = 1

        if abs(Y - Y1) <= Db:
            Ay1 = 0
        elif abs(Y - Y1) >= Db and abs(Y - Y1) <= Dt:
            Ay1 = (abs(Y - Y1) - Db) / (Dt - Db)
        else:
            Ay1 = 1
        if abs(Y - Yg) <= Db:
            Ayg = 0
        elif abs(Y - Yg) >= Db and abs(Y - Yg) <= Dt:
            Ayg = (abs(Y - Yg) - Db) / (Dt - Db)
        else:
            Ayg = 1
        Pr =  Az * Ay * Ay1  + Az1 * Ay * Ay1
        a = math.sqrt(math.pow(Yg - Y, 2) + math.pow(Xg - X, 2))
        # 引力场
        if a<20:

          Ps = a*a / 5000



        else:
          Ps = (abs(Yg - Y))*(abs(Yg - Y)) / 5000

        if a< 10:
            Ps = (10)*(10) / 5000
        if a==10:
            Ps = (10)*(10) / 5000
        # 斥力场
        Po = abs(np.exp(-c1 * (X - X0) ** 2 - c2 * (Y - Y0) ** 2) - Pt) / (1 - Pt)
        P1 = abs(np.exp(-c11 * (X - X1) ** 2 - c21 * (Y - Y1) ** 2) - Pt) / (1 - Pt)

        # 三维虚拟危险势场
        P = Ps
        ZZ[i, j] = P

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(YY, XX, ZZ)
plt.show()
