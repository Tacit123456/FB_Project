import numpy as np
from scipy.integrate import simpson


def calculate_tangential_velocity(U, a, Gamma, theta):
    """
    计算圆柱表面的切向速度。

    参数:
    U (float): 来流速度 (m/s)
    a (float): 圆柱半径 (m)
    Gamma (float): 环量
    theta (array): 角度数组 (弧度)

    返回:
    array: 切向速度
    """
    return Gamma / (2 * np.pi * a) + 2 * U * np.sin(theta)


def calculate_integrand(U, a, Gamma, rho, theta):
    """
    计算升力积分的被积函数。

    参数:
    U (float): 来流速度 (m/s)
    a (float): 圆柱半径 (m)
    Gamma (float): 环量
    rho (float): 流体密度 (kg/m³)
    theta (array): 角度数组 (弧度)

    返回:
    array: 被积函数值
    """
    u_theta = calculate_tangential_velocity(U, a, Gamma, theta)
    return 0.5 * rho * u_theta ** 2 * a * np.sin(theta)


def calculate_lift(U, a, Gamma, rho, N=1000):
    """
    计算圆柱绕流的升力，包括数值计算和理论计算。

    参数:
    U (float): 来流速度 (m/s)
    a (float): 圆柱半径 (m)
    Gamma (float): 环量
    rho (float): 流体密度 (kg/m³)
    N (int): 角度网格点数

    返回:
    tuple: (L_num, L_theory, error) 其中 L_num 是数值升力，L_theory 是理论升力，error 是百分比误差
    """
    theta = np.linspace(0, 2 * np.pi, N)
    integrand = calculate_integrand(U, a, Gamma, rho, theta)

    # 辛普森法积分
    L_num = simpson(integrand, theta)

    # 理论升力
    L_theory = rho * U * Gamma

    # 计算误差
    error = abs(L_num / L_theory - 1) * 100  # 百分比误差

    return L_num, L_theory, error


def calculate_safe_circulation(U, a, n):
    """
    计算安全环量。

    参数:
    U (float): 来流速度 (m/s)
    a (float): 圆柱半径 (m)
    n (float): 安全系数

    返回:
    float: 安全环量
    """
    return (4 * np.pi * U * a) / n


def main():
    # 参数设置
    U = 10.0  # 来流速度 (m/s)
    a = 0.5   # 圆柱半径 (m)
    n = 1.5   # 安全系数
    rho = 1.225  # 流体密度 (kg/m³)

    # 计算安全环量
    Gamma_safe = calculate_safe_circulation(U, a, n)

    # 计算升力
    L_num, L_theory, error = calculate_lift(U, a, Gamma_safe, rho)

    # 输出结果
    print(f"数值升力: {L_num:.2f} N")
    print(f"理论升力: {L_theory:.2f} N")
    print(f"误差: {error:.2f}%")


if __name__ == "__main__":
    main()