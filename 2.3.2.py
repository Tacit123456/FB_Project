import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


def calculate_flow_field(U, a, Gamma, grid_resolution=100):
    """
    计算圆柱绕流的流场，包括流函数和速度场。

    参数:
        U (float): 来流速度
        a (float): 圆柱半径
        Gamma (float): 环量
        grid_resolution (int): 流场网格分辨率

    返回:
        tuple: (X, Y, psi, W) 其中 X 和 Y 是网格坐标，psi 是流函数，W 是复速度场
    """
    # 创建网格
    r = 1.5 * a
    x = np.linspace(-r, r, grid_resolution)
    y = np.linspace(-r, r, grid_resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # 预计算 Z 的倒数
    Z_inv = 1 / Z

    # 复势函数
    Phi = U * (Z + a ** 2 * Z_inv) + 1j * Gamma / (2 * np.pi) * np.log(Z)

    # 复速度场
    W = U * (1 - a ** 2 * Z_inv ** 2) + 1j * Gamma / (2 * np.pi * Z)

    # 提取流函数
    psi = np.imag(Phi)

    return X, Y, psi, W


def visualize_flow_field(U, a, Gamma, grid_resolution=100, streamline_color='b'):
    """
    可视化流场，绘制流线和圆柱。

    参数:
        U (float): 来流速度
        a (float): 圆柱半径
        Gamma (float): 环量
        grid_resolution (int): 流场网格分辨率
        streamline_color (str): 流线颜色
    """
    X, Y, psi, _ = calculate_flow_field(U, a, Gamma, grid_resolution)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Flow Field with Circulation Γ = {Gamma:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

    # 绘制流线
    cs = ax.contour(X, Y, psi, levels=20, colors=streamline_color, linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.1f')

    # 绘制圆柱
    cylinder = plt.Circle((0, 0), a, fill=False, edgecolor='k', linewidth=1)
    ax.add_patch(cylinder)

    # 设置坐标范围
    ax.set_xlim(-1.5 * a, 1.5 * a)
    ax.set_ylim(-1.5 * a, 1.5 * a)

    plt.tight_layout()
    plt.show()


def animate_flow_evolution(U, a, Gamma_values, grid_resolution=100, interval=100, cmap='viridis'):
    """
    动态展示流场随环量变化的演化过程。

    参数:
        U (float): 来流速度
        a (float): 圆柱半径
        Gamma_values (array): 环量值数组
        grid_resolution (int): 流场网格分辨率
        interval (int): 动画帧间隔（毫秒）
        cmap (str): 颜色映射

    返回:
        FuncAnimation: 动画对象
    """
    # 预计算所有流场数据
    precomputed_psi = []
    for Gamma in Gamma_values:
        _, _, psi, _ = calculate_flow_field(U, a, Gamma, grid_resolution)
        precomputed_psi.append(psi)

    # 创建动画
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Flow Field Evolution with Circulation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.set_xlim(-1.5 * a, 1.5 * a)
    ax.set_ylim(-1.5 * a, 1.5 * a)

    # 绘制圆柱
    cylinder = plt.Circle((0, 0), a, fill=False, edgecolor='k', linewidth=1)
    ax.add_patch(cylinder)

    # 创建自定义颜色映射
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1, 0, 0), (0.5, 0, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # 初始化流线
    X, Y = np.meshgrid(np.linspace(-1.5 * a, 1.5 * a, grid_resolution),
                       np.linspace(-1.5 * a, 1.5 * a, grid_resolution))
    contour_set = ax.contourf(X, Y, precomputed_psi[0], levels=20, cmap=custom_cmap, alpha=0.5)
    contour_lines = ax.contour(X, Y, precomputed_psi[0], levels=20, colors='k', linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')

    # 动态更新函数
    def update(frame):
        ax.clear()
        Gamma = Gamma_values[frame]

        # 绘制背景
        ax.set_xlim(-1.5 * a, 1.5 * a)
        ax.set_ylim(-1.5 * a, 1.5 * a)
        ax.set_title(f"Flow Field with Circulation Γ = {Gamma:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')

        # 绘制圆柱
        ax.add_patch(plt.Circle((0, 0), a, fill=False, edgecolor='k', linewidth=1))

        # 绘制流线
        contourf = ax.contourf(X, Y, precomputed_psi[frame], levels=20, cmap=custom_cmap, alpha=0.5)
        contour = ax.contour(X, Y, precomputed_psi[frame], levels=20, colors='k', linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

        return ax

    # 生成动画
    ani = FuncAnimation(fig, update, frames=len(Gamma_values), interval=interval, blit=False)
    return ani


# 参数设置
U = 1.0  # 来流速度
a = 1.0  # 圆柱半径
Gamma_values = np.linspace(-4 * np.pi * U * a, 4 * np.pi * U * a, 50)  # 环量值数组

# 生成动画
ani = animate_flow_evolution(U, a, Gamma_values, grid_resolution=100, interval=100)
plt.tight_layout()
plt.show()