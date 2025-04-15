import numpy as np
from scipy.integrate import simpson


def calc_lift(U, Gamma, a, rho, N=1000):
    theta = np.linspace(0, 2 * np.pi, N)
    u_theta = Gamma / (2 * np.pi * a) + 2 * U * np.sin(theta)  # 切向速度
    integrand = 0.5 * rho * u_theta ** 2 * a * np.sin(theta)  # 添加系数0.5

    # 辛普森法积分
    L_num = simpson(integrand, theta)

    L_theory = rho * U * Gamma  # 理论升力
    error = abs(L_num / L_theory - 1) * 100  # 百分比误差
    return L_num, L_theory, error


# 示例调用（安全环量）
U = 10.0  # 来流速度 (m/s)
a = 0.5  # 圆柱半径 (m)
n = 1.5  # 安全系数
Gamma_safe = (4 * np.pi * U * a) / n  # 安全环量

L_num, L_theory, error = calc_lift(U, Gamma_safe, a, rho=1.225)
print(f"数值升力: {L_num:.2f} N, 理论升力: {L_theory:.2f} N, 误差: {error:.2f}%")