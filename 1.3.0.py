import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numba as nb
import csv

# 初始化参数
U_init = 1.0  # 初始流速 (m/s)
a_init = 1.0  # 初始半径 (m)
nu = 1.5e-5  # 空气运动粘度 (m²/s)


@nb.jit(nopython=True)
def complex_velocity(z, U, a):
    """计算复速度场 V(z) = U(1 - a²/z²)"""
    return U * (1 - (a ** 2) / (z ** 2 + 1e-16j))  # 避免除零错误


@nb.jit(nopython=True)
def velocity_components(x, y, U, a):
    """计算速度场分量"""
    z = x + 1j * y
    V = complex_velocity(z, U, a)
    return V.real, -V.imag  # 注意虚部符号


@nb.jit(nopython=True)
def stream_function(x, y, U, a):
    """计算流函数"""
    z = x + 1j * y
    return (U * (z + a ** 2 / z)).imag


# 创建绘图窗口
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)

# 创建控制面板
ax_a = plt.axes([0.2, 0.15, 0.65, 0.03])
a_slider = Slider(ax_a, 'Radius a (m)', 0.5, 2.0, valinit=a_init, valstep=0.1)

ax_U = plt.axes([0.2, 0.1, 0.2, 0.04])
U_text = TextBox(ax_U, 'Velocity U (m/s): ', initial=str(U_init))

ax_Re = plt.axes([0.6, 0.1, 0.2, 0.04])
Re_text = TextBox(ax_Re, 'Reynolds Number: ', initial='0')

# 初始化流场网格
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)


def update(val):
    """更新流场可视化"""
    a = a_slider.val
    try:
        U = float(U_text.text)
    except:
        U = U_init

    # 计算雷诺数
    Re = 2 * U * a / nu
    Re_text.set_val(f"{Re:.1f}")

    # 清除旧图形
    ax.clear()

    # 计算速度场
    u, v = velocity_components(X, Y, U, a)

    # 计算流函数
    psi = stream_function(X, Y, U, a)

    # 绘制流线
    strm = ax.streamplot(X, Y, u, v, color='white', linewidth=0.8,
                         density=2, arrowsize=1)

    # 绘制等势线
    phi = stream_function(X, Y, U, a)  # 此处应为速度势函数，实际需要修改
    ax.contour(X, Y, phi, levels=20, colors='yellow', linestyles='--', linewidths=0.8)

    # 绘制圆柱
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(a * np.cos(theta), a * np.sin(theta), color='red', alpha=0.3)

    # 标注驻点
    ax.plot([a, -a], [0, 0], 'bo', markersize=8)

    # 流线闭合性验证
    theta_val = np.linspace(0, 2 * np.pi, 10)
    x_val = a * np.cos(theta_val)
    y_val = a * np.sin(theta_val)
    psi_val = stream_function(x_val, y_val, U, a)
    std = np.std(psi_val)
    ax.set_title(f"Cylinder Flow (a={a}m, U={U}m/s)\nStream Function STD: {std:.2e}",
                 fontsize=12)

    # 计算并保存压力系数
    calculate_pressure_coefficient(a, U)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    plt.draw()


def calculate_pressure_coefficient(a, U):
    """计算并保存圆柱表面压力系数"""
    theta = np.deg2rad(np.arange(0, 360, 10))
    z = a * np.exp(1j * theta)
    V = complex_velocity(z, U, a)

    Cp = 1 - (np.abs(V) / U) ** 2  # 压力系数公式

    # 保存CSV
    with open('pressure_coefficient.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Theta (deg)', 'Cp'])
        for t, cp in zip(np.arange(0, 360, 10), Cp):
            writer.writerow([t, f"{cp:.4f}"])


# 绑定事件
a_slider.on_changed(update)
U_text.on_submit(update)

# 初始绘制
update(None)
plt.show()