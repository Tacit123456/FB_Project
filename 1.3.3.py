import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numba as nb
import csv

# 初始化参数
U_init = 1.0  # 初始流速 (m/s)
a_init = 1.0  # 初始半径 (m)
nu = 1.5e-5  # 空气运动粘度 (m²/s)
EPSILON = 1e-12  # 数值稳定性常数

# 颜色主题
BACKGROUND_COLOR = '#f8f8ff'  # 浅紫色背景
CYLINDER_COLOR = '#9370db'  # 浅紫色圆柱
STREAM_COLOR = '#e6e6fa'  # 浅紫色流线
POTENTIAL_COLOR = '#add8e6'  # 浅蓝色等势线

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


@nb.jit(nopython=True, cache=True)
def complex_velocity(z, U, a):
    """计算复速度场 V(z) = U(1 - a²/z²)"""
    return U * (1 - (a ** 2) / (z ** 2 + EPSILON))  # 改进的数值稳定性处理


@nb.jit(nopython=True, cache=True)
def velocity_components(x, y, U, a):
    """计算速度场分量"""
    z = x + 1j * y
    V = complex_velocity(z, U, a)
    return V.real, -V.imag  # 保持物理正确的速度分量


@nb.jit(nopython=True, cache=True)
def stream_function(x, y, U, a):
    """计算流函数（复势的虚部）"""
    z = x + 1j * y
    return (U * (z + a ** 2 / (z + EPSILON))).imag  # 增加数值稳定性


@nb.jit(nopython=True, cache=True)
def potential_function(x, y, U, a):
    """计算速度势函数（复势的实部）"""
    z = x + 1j * y
    return (U * (z + a ** 2 / (z + EPSILON))).real  # 增加数值稳定性


# 创建绘图窗口
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)

# 设置背景颜色
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

# 创建控制面板
ax_a = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=BACKGROUND_COLOR)
a_slider = Slider(ax_a, '半径 a (m)', 0.5, 2.0, valinit=a_init, valstep=0.1)

ax_U = plt.axes([0.2, 0.1, 0.2, 0.04], facecolor=BACKGROUND_COLOR)
U_text = TextBox(ax_U, '流速 U (m/s): ', initial=str(U_init))

ax_Re = plt.axes([0.6, 0.1, 0.2, 0.04], facecolor=BACKGROUND_COLOR)
Re_text = TextBox(ax_Re, '雷诺数: ', initial='0')

# 初始化流场网格
x = np.linspace(-3, 3, 100)  # 减少网格点数以提高性能
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)


def update(val):
    """更新流场可视化"""
    a = a_slider.val
    try:
        U = float(U_text.text)
    except ValueError:
        U = U_init

    # 计算雷诺数
    Re = 2 * U * a / nu
    Re_text.set_val(f"{Re:.1f}")

    # 清除旧图形
    ax.clear()
    ax.set_facecolor(BACKGROUND_COLOR)

    # 计算速度场
    u, v = velocity_components(X, Y, U, a)

    # 计算流函数和速度势
    psi = stream_function(X, Y, U, a)
    phi = potential_function(X, Y, U, a)

    # 绘制流线
    ax.streamplot(X, Y, u, v, color=STREAM_COLOR, linewidth=0.8,
                  density=1.0, arrowsize=1)

    # 绘制等势线（浅蓝色虚线）
    ax.contour(X, Y, phi, levels=np.linspace(-3, 3, 15),
               colors=POTENTIAL_COLOR, linestyles='--', linewidths=0.8)

    # 绘制圆柱
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(a * np.cos(theta), a * np.sin(theta), color=CYLINDER_COLOR, alpha=0.5)

    # 标注驻点
    ax.plot([a, -a], [0, 0], 'bo', markersize=8)

    # 流线闭合性验证
    theta_val = np.linspace(0, 2 * np.pi, 100)
    x_val = a * np.cos(theta_val)
    y_val = a * np.sin(theta_val)
    psi_val = stream_function(x_val, y_val, U, a)
    std = np.std(psi_val)
    ax.set_title(f"圆柱绕流 (a={a}m, U={U}m/s)\n流函数标准差: {std:.2e}",
                 fontsize=12)

    # 计算并保存压力系数
    calculate_pressure_coefficient(a, U)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    plt.draw()


def calculate_pressure_coefficient(a, U):
    """计算并保存圆柱表面压力系数"""
    theta = np.linspace(0, 2 * np.pi, 360)  # 1度间隔
    z = a * np.exp(1j * theta)
    V = complex_velocity(z, U, a)

    # 压力系数计算
    Cp = 1 - (np.abs(V) / U) ** 2

    # 保存CSV（增加角度修正使数据连续）
    with open('pressure_coefficient.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['角度 (度)', '压力系数'])
        for t, cp in zip(np.rad2deg(theta), Cp):
            writer.writerow([f"{t:.1f}", f"{cp:.4f}"])


# 绑定事件
a_slider.on_changed(update)
U_text.on_submit(update)

# 初始绘制
update(None)
plt.show()