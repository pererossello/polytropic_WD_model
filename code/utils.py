import numpy as np
import numba

def RK4_second_order(t, x, v, f, h):
    try:
        k1_x = h * v
        k1_v = h * f(t, x, v)

        k2_x = h * (v + k1_v / 2)
        k2_v = h * f(t + h / 2, x + k1_x / 2, v + k1_v / 2)

        k3_x = h * (v + k2_v / 2)
        k3_v = h * f(t + h / 2, x + k2_x / 2, v + k2_v / 2)

        k4_x = h * (v + k3_v)
        k4_v = h * f(t + h, x + k3_x, v + k3_v)

        x_ = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        v_ = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    except ZeroDivisionError:
        return np.nan, np.nan
    return x_, v_


def ode_2_paths(
    f, t_range, initial_conditions, N=100, ortho=False, path_idx=None, n_orth_paths=20
):
    if isinstance(initial_conditions, (list, tuple)):
        if all(isinstance(item, (int, float)) for item in initial_conditions):
            initial_conditions = [initial_conditions]
        elif all(isinstance(item, (list, tuple)) for item in initial_conditions):
            initial_conditions = initial_conditions
        else:
            raise ValueError("Invalid initial conditions")

    t = np.linspace(t_range[0], t_range[1], N)
    dt = t[1] - t[0]
    xs, vs = np.zeros((N, len(initial_conditions))), np.zeros(
        (N, len(initial_conditions))
    )

    for k, initial_condition in enumerate(initial_conditions):
        t_0, x_0, v_0 = initial_condition[:3]
        idx = np.argmin(np.abs(t - t_0))
        xs[idx, k], vs[idx, k] = x_0, v_0

        # Compute x(t) forward
        for i in range(idx + 1, N):
            xs[i, k], vs[i, k] = RK4_second_order(
                t[i - 1], xs[i - 1, k], vs[i - 1, k], f, dt
            )
            if xs[i, k] < 0.001:
                xs[i, k] = np.nan
                vs[i, k] = np.nan

    return t, xs, vs


def solve_polytropes(enes, dt=1e-3, xi_range=(0,5)):

    def f_n(xi, theta, dthetadxi, n):
        try:
            edo = - theta ** n - 2 / xi * dthetadxi
            return edo
        except ZeroDivisionError:
            return np.nan

    M = len(enes)
    initial_conditions = [(0.01, 1, 0 )]
    N = int((xi_range[1] - xi_range[0]) / dt)
    th_arr, dthdxi_arr = np.zeros((N, len(enes))), np.zeros((N, len(enes)))

    for i, n in enumerate(enes):
        def f(xi, theta, dthetadxi):
            return f_n(xi, theta, dthetadxi, n)

        xis, thetas, dthetadxis = ode_2_paths(f, xi_range, initial_conditions, N=N)

        th_arr[:, i] = thetas[:,0]
        dthdxi_arr[:, i] = dthetadxis[:,0]

    return xis, th_arr, dthdxi_arr


def get_coefs(enes, xis, th_arr, dthdxi_arr):
    
    coefs = {f'{n:0.1f}': {} for n in list(enes)}

    for i, n in enumerate(enes):
        th = th_arr[:, i]
        th_min = np.argmin(th)
        xi_1 = xis[th_min]
        dthdxi_1 = dthdxi_arr[th_min-1, i]

        coefs[f'{n:0.1f}']['xi_1'] = xi_1
        coefs[f'{n:0.1f}']['dthdxi_1'] = dthdxi_1
        


        R_n = xi_1
        G_n = - dthdxi_1
        D_n = R_n / (3 * G_n)
        M_n = G_n * R_n ** 2 
        B2_n = (1/(n+1)) * ((R_n**(2) * G_n) ** (-2/3))



        coefs[f'{n:0.1f}']['R_n'] = R_n
        coefs[f'{n:0.1f}']['G_n'] = G_n
        coefs[f'{n:0.1f}']['D_n'] = D_n
        coefs[f'{n:0.1f}']['M_n'] = M_n
        coefs[f'{n:0.1f}']['B_n'] = B2_n


    n_coef = {coef: [coefs[f'{n:0.1f}'][coef] for n in enes] for coef in ['R_n', 'G_n','D_n', 'M_n', 'B_n']}

    return coefs, n_coef


def format_scientific(num, prec=2):
    # Format the number in scientific notation with 2 decimal places for 'a'
    formatted =f"{num:.{prec}e}"

    # Split the string into the coefficient and exponent parts
    coefficient, exponent = formatted.split('e')
    # Remove the plus sign from the exponent if it exists
    exponent = exponent.replace('+', '')

    # Return the formatted string
    return rf"${coefficient} \times 10^{int(exponent)}$"