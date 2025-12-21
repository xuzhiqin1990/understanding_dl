import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline,Rbf

# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 定义五个数据点
x_points = np.array([0, 1, 2, 3, 4])
y_points = np.array([1, 4, 3, 5, 2])

# 创建一个平滑的x轴用于绘製曲线
x_smooth = np.linspace(x_points.min(), x_points.max(), 400)

# 创建图表
plt.figure(figsize=(12, 8))

# 2. 绘制原始数据点
plt.scatter(x_points, y_points, color='red', s=100, zorder=5, label='数据点 (Data Points)')

# 3. 生成并绘制四条不同的插值曲线

# --- 保留的平滑与基础曲线 ---
# 曲线 1: 线性插值
plt.plot(x_points, y_points, 'o-', alpha=0.7, label='线性插值 (Linear)')

# 曲线 2: 三次样条插值 (作为平滑基准)
cs = CubicSpline(x_points, y_points)
y_cs = cs(x_smooth)
plt.plot(x_smooth, y_cs, label='三次样条插值 (Cubic Spline)')


# --- 新增的复杂与高波动性曲线 ---

# 曲线 3: 径向基函数 (RBF) 插值 (一种复杂的插值方法)
# RBF 的行为可以很复杂，这里使用默认的高斯核函数
rbf = Rbf(x_points, y_points, function='gaussian')
y_rbf = rbf(x_smooth)
plt.plot(x_smooth, y_rbf, label='径向基函数插值 (RBF)')

# 曲线 4: 高频振荡插值 (波动性极大的曲线)
# 我们在平滑的三次样条基础上，添加一个在数据点上为0的正弦波
# np.sin(5 * np.pi * x) 在所有整数点x上都为0，因此曲线必过数据点
y_oscillating = y_cs + np.sin(5 * np.pi * x_smooth) * 0.8
plt.plot(x_smooth, y_oscillating, linestyle='--', 
         label='高频振荡插值 (Highly Oscillating)')


# 4. 添加图例、标题和标签
plt.title('穿过五个数据点的四种不同复杂度曲线')
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5) # 添加x=0的参考线
plt.ylim(0, 8) # 调整Y轴范围以完整显示所有曲线的波动
plt.show()
plt.savefig('interpolation_curves.pdf', dpi=300)  # 保存图像