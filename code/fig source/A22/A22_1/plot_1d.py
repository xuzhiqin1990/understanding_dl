import numpy as np
import matplotlib.pyplot as plt

# 创建 x 值的范围
x = np.linspace(-2, 2, 100)

# 计算两个二次函数的值
y1 = x**2        # y = x^2
y2 = 5 * x**2    # y = 5x^2

# 创建图形
plt.figure(figsize=(10, 7))

# 绘制两条曲线
plt.plot(x, y1, 'b', linewidth=2, label=r'y = $x^2$')
plt.plot(x, y2, 'r--', linewidth=2, label=r'y = $5x^2$')

# 设置坐标轴经过原点
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 去除背景网格
plt.grid(False)
ax.set_axis_off()  # 完全关闭坐标轴和背景格点
plt.legend(fontsize=24)

# 保存图形
plt.savefig('plot_1d.pdf', bbox_inches='tight', transparent=True)
plt.close()