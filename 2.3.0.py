import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def generate_streamlines(U, a, Gamma, theta_res=100):
    theta = np.linspace(0, 2 * np.pi, theta_res)
    r = 1.5 * a  # 观测范围
    x = np.linspace(-r, r, 100)
    y = np.linspace(-r, r, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # 复势函数（含涡项）
    Phi = U * (Z + a ** 2 / Z) + 1j * Gamma / (2 * np.pi) * np.log(Z)
    W = U * (1 - a ** 2 / Z ** 2) + 1j * Gamma / (2 * np.pi * Z)  # 复速度场

    # 计算流函数
    psi = np.imag(Phi)
    return X, Y, psi


# 参数设置
U = 1.0  # 来流速度
a = 1.0  # 圆柱半径
Gamma_values = np.linspace(-4 * np.pi * U * a, 4 * np.pi * U * a, 50)

# 初始化画布
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Flow Field Evolution with Circulation")
ax.set_xlabel("x")
ax.set_ylabel("y")


# 动态更新函数
def update(frame):
    ax.clear()
    Gamma = Gamma_values[frame]
    X, Y, psi = generate_streamlines(U, a, Gamma)
    cs = ax.contour(X, Y, psi, levels=20, colors='b', linewidths=0.5)
    ax.plot(a * np.cos(np.linspace(0, 2 * np.pi, 100)),
            a * np.sin(np.linspace(0, 2 * np.pi, 100)), 'k-')  # 绘制圆柱
    ax.set_title(f"Γ = {Gamma:.2f}")
    return cs


# 生成动画
ani = FuncAnimation(fig, update, frames=len(Gamma_values), interval=100)
plt.show()