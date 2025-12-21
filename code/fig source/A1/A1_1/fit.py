import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成示例数据
np.random.seed(42)
x = np.linspace(0, 10, 100)

# 线性数据 y1 = 3x + 5 + 噪声
y1 = 3 * x + 5 + np.random.normal(0, 2, x.size)

# 二次数据 y2 = 2x² + 3x + 5 + 噪声
y2 = 2 * x**2 + 3 * x + 5 + np.random.normal(0, 5, x.size)

# 线性拟合（针对 y1）
linear_coeffs = np.polyfit(x, y1, 1)
linear_fit = np.poly1d(linear_coeffs)
y1_linear = linear_fit(x)

# 二次拟合（针对 y2）
quadratic_coeffs = np.polyfit(x, y2, 2)
quadratic_fit = np.poly1d(quadratic_coeffs)
y2_quadratic = quadratic_fit(x)

# 绘制线性拟合图
plt.figure(figsize=(6, 5))
plt.scatter(x, y1, label='数据（线性）', alpha=0.5)
plt.plot(x, y1_linear, 'r-', label=f'线性拟合')
plt.title('线性拟合')
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.legend()
plt.grid(True)
plt.savefig('linear_fit.pdf', dpi=300)  # 保存图像
# 绘制二次拟合图
plt.figure(figsize=(6, 5))
plt.scatter(x, y2, label='数据（二次）', alpha=0.5)
plt.plot(x, y2_quadratic, 'r-', label=f'二次拟合')
plt.title('二次拟合')
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.legend()
plt.grid(True)
plt.savefig('quadratic_fit.pdf', dpi=300)  # 保存图像