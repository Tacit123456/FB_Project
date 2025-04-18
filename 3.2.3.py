import numpy as np
from scipy.integrate import simpson


def calculate_tangential_velocity(U, a, Gamma, theta):
    """计算圆柱表面切向速度"""
    return Gamma / (2 * np.pi * a) + 2 * U * np.sin(theta)


def calculate_pressure(U, a, Gamma, rho, theta):
    """计算表面压力分布（基于伯努利方程）"""
    u_theta = calculate_tangential_velocity(U, a, Gamma, theta)
    return 0.5 * rho * (U ** 2 - u_theta ** 2)


def calculate_lift(U, a, Gamma, rho, N=1000):
    """计算数值升力与理论升力"""
    theta = np.linspace(0, 2 * np.pi, N)
    pressure = calculate_pressure(U, a, Gamma, rho, theta)

    # 升力计算（积分压力的竖直分量）
    L_num = -2 * np.pi * a * simpson(pressure * np.sin(theta), theta)

    # 理论升力（二维，单位长度）
    L_theory = rho * U * Gamma

    error = abs((L_num - L_theory) / L_theory) * 100

    return L_num, L_theory, error


def parameter_sensitivity_analysis():
    """参数敏感性分析"""
    U, a, rho = 10.0, 0.5, 1.225
    Gamma = 4 * np.pi * U * a / 1.5  # 安全环量

    N_values = [50, 100, 500, 1000, 2000]
    results = []
    for N in N_values:
        L_num, L_theory, error = calculate_lift(U, a, Gamma, rho, N)
        results.append((N, L_num, error))

    print("积分点数 | 数值升力 (N) | 相对误差 (%)")
    print("-------------------------------------")
    for n, l, e in results:
        print(f"{n:6d} | {l:10.2f} | {e:8.2f}%")


if __name__ == "__main__":
    U, a, rho = 10.0, 0.5, 1.225
    Gamma_safe = 4 * np.pi * U * a / 1.5

    L_num, L_theory, error = calculate_lift(U, a, Gamma_safe, rho)
    print(f"[结果] 数值升力: {L_num:.2f} N, 理论升力: {L_theory:.2f} N, 误差: {error:.2f}%")

    parameter_sensitivity_analysis()