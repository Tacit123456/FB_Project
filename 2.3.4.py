import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


def calculate_flow_field(U, a, Gamma, grid_resolution=100):
    """Calculate flow field around a cylinder with vortex"""
    r = 2.0 * a
    x = np.linspace(-r, r, grid_resolution)
    y = np.linspace(-r, r, grid_resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    Z_inv = 1 / Z
    Phi = U * (Z + a**2 * Z_inv) + 1j * Gamma / (2 * np.pi) * np.log(Z)
    W = U * (1 - a**2 * Z_inv**2) + 1j * Gamma / (2 * np.pi * Z)
    psi = np.imag(Phi)

    return X, Y, psi, W


def animate_flow_evolution(U, a, Gamma_values, grid_resolution=100, interval=100):
    """动态展示流场演变（含驻点标注）"""
    # 预计算流场数据
    precomputed_psi = []
    for Gamma in Gamma_values:
        _, _, psi, _ = calculate_flow_field(U, a, Gamma, grid_resolution)
        precomputed_psi.append(psi)

    # 初始化图形
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2 * a, 2 * a)
    ax.set_ylim(-2 * a, 2 * a)
    ax.set_aspect('equal')

    # 自定义颜色映射
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1, 0, 0), (0.5, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

    # 更新函数
    def update(frame):
        ax.clear()
        Gamma = Gamma_values[frame]
        ax.set_title(f"Γ = {Gamma:.2f}", fontsize=12)
        ax.set_xlim(-2 * a, 2 * a)
        ax.set_ylim(-2 * a, 2 * a)
        ax.add_patch(plt.Circle((0, 0), a, fill=False, edgecolor='k', linewidth=1))

        x = np.linspace(-2 * a, 2 * a, grid_resolution)
        y = np.linspace(-2 * a, 2 * a, grid_resolution)
        X, Y = np.meshgrid(x, y)
        contourf = ax.contourf(X, Y, precomputed_psi[frame], levels=20, cmap=cmap, alpha=0.5)
        contour = ax.contour(X, Y, precomputed_psi[frame], levels=20, colors='k', linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

        # 计算并标注驻点
        if abs(Gamma) <= 4 * np.pi * U * a:
            theta_s = np.arcsin(-Gamma / (4 * np.pi * U * a))
            x_s = a * np.cos(theta_s)
            y_s = a * np.sin(theta_s)
            ax.plot(x_s, y_s, 'ro', markersize=6, label='Stagnation Point')
        else:
            ax.text(0, 1.5 * a, 'Flow Separation Detected', color='red', ha='center', va='bottom', fontweight='bold')

        return ax

    # 优化动画生成
    ani = FuncAnimation(fig, update, frames=len(Gamma_values), interval=interval, blit=False)
    plt.close()
    return ani


# 参数设置与运行动画
U = 1.0
a = 1.0
Gamma_values = np.linspace(-5 * np.pi * U * a, 5 * np.pi * U * a, 100)  # 扩展环量范围
ani = animate_flow_evolution(U, a, Gamma_values, interval=150)
ani.save('flow_evolution.gif', writer='pillow', fps=15)