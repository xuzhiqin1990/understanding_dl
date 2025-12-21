import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
eps = 1e-3

# 定义二次函数 f(x, y) = ax^2 + by^2
def f(x, y, a=5, b=1):
    return a * x**2 + b * y**2

# 创建网格
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 绘制 3D 曲面图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 添加 x=0 截面（y 方向的曲线）
y_vals = np.linspace(-5, 5, 100)
z_vals_x0 = f(np.zeros_like(y_vals), y_vals)  # x=0 时的函数值
ax.plot(np.zeros_like(y_vals), y_vals, z_vals_x0 + eps, color='r', label='x = 0')

# 添加 y=0 截面（x 方向的曲线）
x_vals = np.linspace(-5, 5, 100)
z_vals_y0 = f(x_vals, np.zeros_like(x_vals))  # y=0 时的函数值
ax.plot(x_vals, np.zeros_like(x_vals), z_vals_y0 + eps, color='b', label='y = 0')


# 曲面绘制
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)

# 去除背景网格和坐标轴
ax.set_axis_off()  # 完全关闭坐标轴和背景格点

# 设置视角（可选）
# ax.view_init(elev=30, azim=45)  # 可以调整这两个角度来改变视角

# 显示图形
plt.savefig('plot_quadratic.pdf', bbox_inches='tight', transparent=True)


# # 添加标题和标签
# # ax.set_title('3D Surface of f(x, y) = ax^2 + by^2 with x=0 and y=0 curves')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x, y)')
# # ax.set_axis_off()  # 完全关闭坐标轴和背景格点

# # # 显示图例
# # ax.legend()


# # 显示图形
# plt.savefig('plot_quadratic.pdf')