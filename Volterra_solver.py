import numpy as np
import matplotlib.pyplot as plt

# ============================
# НАСТРОЙКА ШРИФТА
# ============================

plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================
# ПАРАМЕТРЫ
# ============================

l = 1.0
b = 1.0
q = 1_000_000

T = 1.0
Nt = 200
dt = T / Nt
t_grid = np.linspace(0, T, Nt)

N_modes = 30

# ============================
# СОБСТВЕННЫЕ ФУНКЦИИ
# ============================

def lambda_n(n):
    return (b / 2) * (np.pi * n / l) ** 2

def X_n(n, x):
    return np.sin(np.pi * n * x / l)

def C_n(n):
    return l / (np.pi * n)

# ============================
# ДИСКРЕТИЗАЦИЯ x
# ============================

x_dense = np.linspace(0, l, 500)

def M_n(n):
    return np.trapezoid(x_dense**q * X_n(n, x_dense), x_dense)

H_const = np.trapezoid(x_dense**q * (1 - x_dense / l), x_dense)

def H():
    return H_const

# ============================
# НАЧАЛЬНОЕ УСЛОВИЕ
# ============================

def phi(x, mode=1):
    if mode == 1:
        return np.sin(np.pi * x * 0.5)
    elif mode == 2:
        return np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    elif mode == 3:
        return np.sin(9 * np.pi * x)
    elif mode == 4:
        return np.abs(x - 0.5)
    else:
        return np.sin(np.pi * x)

def compute_a0(n):
    return np.trapezoid(phi(x_dense) * X_n(n, x_dense), x_dense)

a0_cache = np.array([compute_a0(n) for n in range(N_modes)])

# ============================
# ПРАВАЯ ЧАСТЬ
# ============================

def F(x, t):
    return np.exp(-t) * np.sin(np.pi * x) * (-1 + np.pi**2 / 8)

# ============================
# ЯДРО ВОЛЬТЕРРА
# ============================

def compute_K(tau):
    val = 0.0

    for n in range(1, N_modes):
        val += (
            lambda_n(n)
            * C_n(n)
            * M_n(n)
            * np.exp(-lambda_n(n) * tau)
        )

    return val

# ============================
# ИНТЕГРАЛЬНЫЕ ДАННЫЕ
# ============================

def N_q(t):
    u = np.exp(-t) * np.sin(np.pi * x_dense)
    return np.trapezoid(x_dense**q * u, x_dense)

# ============================
# МАТРИЦА ВОЛЬТЕРРА
# ============================

def build_volterra_matrix():

    A = np.zeros((Nt, Nt))

    for k in range(Nt):

        for j in range(k + 1):

            tau = t_grid[k] - t_grid[j]

            A[k, j] = compute_K(tau) * dt

        A[k, k] += 1.0

    return A

# ============================
# ПРАВАЯ ЧАСТЬ
# ============================

def build_rhs(Nq_vals):

    f = np.zeros(Nt)

    for k in range(Nt):

        t = t_grid[k]

        F1 = Nq_vals[k]

        for n in range(1, N_modes):

            lam = lambda_n(n)
            Mn = M_n(n)

            Bn = a0_cache[n] * np.exp(-lam * t)

            F1 -= Bn * Mn

        f[k] = F1

    return f

# ============================
# МАТРИЦА ВТОРОЙ ПРОИЗВОДНОЙ
# ============================

def second_derivative_matrix(N):

    L = np.zeros((N, N))

    for i in range(1, N - 1):

        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0

    return L / dt**2

# ============================
# РЕГУЛЯРИЗАЦИЯ ТИХОНОВА
# ============================

def solve_tikhonov_sobolev(A, f, alpha):

    L = second_derivative_matrix(Nt)

    ATA = A.T @ A
    ATf = A.T @ f

    regularizer = alpha * (L.T @ L)

    g = np.linalg.solve(ATA + regularizer, ATf)

    return g

# ============================
# РЕШЕНИЕ УРАВНЕНИЯ
# ============================

def solve_volterra(noise_level=0.0, alpha=0.0):

    Nq_vals = np.array([N_q(t) for t in t_grid])

    noise = noise_level * np.random.randn(Nt)

    Nq_noisy = Nq_vals + noise

    A = build_volterra_matrix()

    f = build_rhs(Nq_noisy)

    g = solve_tikhonov_sobolev(A, f, alpha)

    return g, Nq_vals, Nq_noisy

# ============================
# МЕТРИКА НЕУСТОЙЧИВОСТИ
# ============================

def compute_instability(g_true, g_noisy, N_true, N_noisy):

    dg = np.linalg.norm(g_true - g_noisy)

    dN = np.linalg.norm(N_true - N_noisy)

    return dg / dN

# ============================
# ВЫЧИСЛЕНИЯ
# ============================

g_clean, N_true, _ = solve_volterra(
    noise_level=0.0,
    alpha=0.0
)

g_noisy, _, N_noisy = solve_volterra(
    noise_level=0.001,
    alpha=0.0
)

g_reg, _, _ = solve_volterra(
    noise_level=0.001,
    alpha=0.000000005
)

# ============================
# ГРАФИКИ
# ============================

plt.figure(figsize=(10, 5))

plt.plot(
    t_grid,
    g_clean,
    color='black',
    linestyle='-',
    linewidth=2,
    label='Точное решение'
)

plt.plot(
    t_grid,
    g_noisy,
    color='black',
    linestyle='--',
    linewidth=1.5,
    label='Решение с шумом'
)

# plt.plot(
#     t_grid,
#     g_reg,
#     color='black',
#     linestyle='-',
#     linewidth=2,
#     label='Регуляризованное решение'
# )

plt.xlabel('t',fontsize=14)
plt.ylabel('g(t)',fontsize=14)

plt.title('Восстановление функции g(t)',fontsize=16)

plt.legend()

plt.grid(True)

plt.show()

# ============================
# ГРАФИК ИНТЕГРАЛЬНЫХ ДАННЫХ
# ============================

plt.figure(figsize=(10, 5))

plt.plot(
    t_grid,
    N_true,
    color='black',
    linestyle='-',
    linewidth=2,
    label='Точные данные'
)

plt.plot(
    t_grid,
    N_noisy,
    color='black',
    linestyle='--',
    linewidth=1.5,
    label='Данные с шумом'
)

plt.xlabel('t')
plt.ylabel('N_q(t)')

plt.title('Интегральные данные')

plt.legend()

plt.grid(True)

plt.show()

# ============================
# ОЦЕНКА НЕУСТОЙЧИВОСТИ
# ============================

instability_noisy = compute_instability(
    g_clean,
    g_noisy,
    N_true,
    N_noisy
)

instability_reg = compute_instability(
    g_clean,
    g_reg,
    N_true,
    N_noisy
)

print("Неустойчивость без регуляризации:",
      instability_noisy)

print("Неустойчивость с регуляризацией:",
      instability_reg)