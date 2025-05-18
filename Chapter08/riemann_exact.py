# This code is based on Zhengyu(Daniel) Huang's Pythonized version
# (see https://cfdnotes2017.blogspot.com/2017/02/test.html)
# of Eleuterio F. Toro's exact Riemann solver, originally written in FORTRAN
# and published in the book "Riemann Solvers and Numerical Methods for Fluid Dynamics".

import numpy as np

def pressure_function(P, Q, gamma):
    """
    Computes the pressure function and its derivative for a given pressure P,
    state vector Q, and specific heat ratio gamma. This function handles both
    shock and rarefaction cases.
    Parameters:
      P (float): The pressure at the current state.
      Q (tuple): A tuple containing the density (rho_k), velocity (v_k), and 
             pressure (P_k) of the state.
      gamma (float): The specific heat ratio (adiabatic index).
    Returns:
      tuple: A tuple containing:
        - f (float): The value of the pressure function.
        - df (float): The derivative of the pressure function with respect to P.
    """
    
    rho_k, _, P_k = Q
    if (P > P_k):  # shock
        A_k, B_k = 2 / ((gamma+1) * rho_k), (gamma-1)/(gamma+1) * P_k
        f = (P-P_k) * np.sqrt(A_k / (P + B_k))
        df = np.sqrt(A_k/(B_k+P)) * (1-(P-P_k) / (2*(B_k+P)))

    else:  # rarefaction
        cs_k = np.sqrt(gamma*P_k/rho_k)
        f = 2*cs_k / (gamma - 1) * ((P/P_k) ** ((gamma-1)/(2*gamma))-1)
        df = 1 / (rho_k*cs_k) * (P/P_k) ** (-(gamma+1)/(2*gamma))

    return f, df

def solve_contact_discontinuity(Q_l, Q_r, gamma):
    """
    Solves the contact discontinuity in the Riemann problem using the Newton-Raphson method.
    Parameters:
      Q_l (tuple): State vector on the left side of the discontinuity (density, velocity, pressure).
      Q_r (tuple): State vector on the right side of the discontinuity (density, velocity, pressure).
      gamma (float): Ratio of specific heats (adiabatic index).
    Returns:
      tuple: A tuple containing:
        - P (float): Pressure at the contact discontinuity.
        - v (float): Velocity at the contact discontinuity.
    Notes:
      - The function uses an iterative Newton-Raphson method to solve for the pressure at the contact discontinuity.
      - Convergence is determined by a relative tolerance (`tol`) on the pressure.
      - If the method does not converge within the maximum number of iterations (`iter_max`), a warning is printed.
      - Negative pressures are avoided by clamping the pressure to a minimum value (`tol`).
    Raises:
      RuntimeWarning: If the Newton-Raphson method fails to converge within the maximum number of iterations.
    """
    # Unpack the left and right state vectors
    _, v_l, P_l = Q_l
    _, v_r, P_r = Q_r

    d_v = v_r - v_l

    # Initial guess for the pressure at the contact discontinuity
    iter_max  = 100
    tol       = 1e-8
    converged = False

    P_old = 0.5 * (P_l + P_r)
    # Newton-Raphson iteration
    for _ in range(iter_max):
        f_l, df_l = pressure_function(P_old, Q_l, gamma)
        f_r, df_r = pressure_function(P_old, Q_r, gamma)

        P = P_old - (f_l + f_r + d_v) / (df_l + df_r)

        P = max(P, tol) # avoid negative pressure
        
        if (2 * abs(P - P_old) / (P + P_old) < tol):
            converged = True
            break
        P_old = P

    if not converged:
        print('No convergence in Newton-Raphson iterations')

    # Calculate the velocity at the contact discontinuity
    v = 0.5 * (v_l + v_r + f_r - f_l)

    return P, v

def sample_solution(P_m, v_m, Q_l, Q_r, gamma, s):
    """
    Compute the solution at a given sampling point `s` for the Riemann problem 
    with given left and right states.
    The function determines whether the sampling point lies in the left or 
    right region, and whether it is in a shock wave, rarefaction wave, or 
    contact discontinuity. It then computes the corresponding density, velocity, 
    and pressure at the sampling point.
    Parameters:
    -----------
    P_m : float
      Pressure in the middle state (post-contact discontinuity).
    v_m : float
      Velocity in the middle state (post-contact discontinuity).
    Q_l : tuple of floats
      Left state variables (rho_l, v_l, P_l), where:
      - rho_l: Density in the left state.
      - v_l: Velocity in the left state.
      - P_l: Pressure in the left state.
    Q_r : tuple of floats
      Right state variables (rho_r, v_r, P_r), where:
      - rho_r: Density in the right state.
      - v_r: Velocity in the right state.
      - P_r: Pressure in the right state.
    gamma : float
      Ratio of specific heats (adiabatic index).
    s : float
      Sampling point in the spatial domain.
    Returns:
    --------
    tuple of floats
      The state variables (rho, v, p) at the sampling point `s`, where:
      - rho: Density at the sampling point.
      - v: Velocity at the sampling point.
      - p: Pressure at the sampling point.
    """
    
    rho_l, v_l, P_l = Q_l
    cs_l = np.sqrt(gamma * P_l / rho_l)

    rho_r, v_r, P_r = Q_r
    cs_r = np.sqrt(gamma * P_r / rho_r)

    if s < v_m:
        # sampling point lies to the left of the contact discontinuity
        if P_m < P_l:
            # left rarefaction
            s_l = v_l - cs_l
            a_ml = cs_l * (P_m / P_l) ** ((gamma - 1) / (2 * gamma))
            s_ml = v_m - a_ml
            if s < s_l:  # left state
                return Q_l
            elif s < s_ml:  # left rarefaction wave
                rho = rho_l * (2/(gamma+1) + (gamma-1) / ((gamma+1)*cs_l) * (v_l-s)) ** (2/(gamma-1))
                v = 2 / (gamma + 1) * (cs_l + (gamma-1) / 2.0 * v_l + s)
                p = P_l * (2/(gamma+1) + (gamma-1) / ((gamma+1) * cs_l) * (v_l-s)) ** (2*gamma/(gamma-1))
                return rho, v, p
            else:  # left contact discontinuity
                return rho_l * (P_m/P_l) ** (1/gamma), v_m, P_m
        else:
            # left shock
            s_shock = v_l - cs_l * np.sqrt((gamma+1) * P_m / (2*gamma*P_l) + (gamma-1) / (2*gamma))
            if s < s_shock:
                return Q_l
            else:
                rho_m = rho_l * (P_m/P_l + (gamma-1)/(gamma+1)) / ((gamma-1) * P_m / ((gamma+1)*P_l) + 1)
                return rho_m, v_m, P_m
    else:
        # sampling point lies to the right of the contact discontinuity
        if P_m < P_r:
            # right rarefaction
            s_r = v_r + cs_r
            a_mr = cs_r * (P_m/P_r) ** ((gamma-1) / (2*gamma))
            s_mr = v_m + a_mr
            if s > s_r:  # left state
                return Q_r
            elif s > s_mr:  # left rarefaction wave
                rho = rho_r * (2/(gamma+1) - (gamma-1) / ((gamma+1)*cs_r) * (v_r-s)) ** (2/(gamma-1))
                v = 2/(gamma+1) * (-cs_r + (gamma-1) / 2.0 * v_r+s)
                p = P_r * (2/(gamma+1) - (gamma-1)/((gamma+1) * cs_r) * (v_r-s)) ** (2*gamma/(gamma-1))
                return rho, v, p

            else:  # left contact discontinuity
                rho_m = rho_r * (P_m / P_r) ** (1/gamma)
                return rho_m, v_m, P_m
        else:
            # right shock

            s_shock = v_r + cs_r * np.sqrt((gamma+1) * P_m / (2*gamma*P_r) + (gamma-1)/(2 * gamma))

            if s > s_shock: # after shock
                return Q_r
            else:
                #preshock, contact discontinuity
                rho_m = rho_r * (P_m/P_r + (gamma-1)/(gamma+1)) / ((gamma-1) * P_m / ((gamma+1)*P_r) + 1)
                return rho_m, v_m, P_m


def solve_riemann_exact(*, left_state, right_state, t, gamma, npts ):
    def solve_riemann_exact(*, left_state, right_state, t, gamma, npts):
      """Solves the Riemann problem for a given set of initial conditions using the exact solution.
         Solution domain is [0,1] with a contact discontinuity at x=0.5.
      Parameters:
      -----------
      left_state : tuple
        Left primitive variables (rho_L, u_L, P_L), where:
        - rho_L: density on the left side
        - u_L: velocity on the left side
        - P_L: pressure on the left side
      right_state : tuple
        Right primitive variables (rho_R, u_R, P_R), where:
        - rho_R: density on the right side
        - u_R: velocity on the right side
        - P_R: pressure on the right side
      t : float
        Time at which the solution is evaluated.
      gamma : float
        Polytropic exponent (ratio of specific heats).
      npts : int
        Number of points at which to sample the solution.

      Returns:
      --------
      dict
        A dictionary containing the solution values:
        - 'x': numpy array of spatial positions where the solution is sampled.
        - 'rho': numpy array of density values at the sampled positions.
        - 'v': numpy array of velocity values at the sampled positions.
        - 'P': numpy array of pressure values at the sampled positions.

    """
    x         = np.linspace(0.0, 1.0, num=npts)
    Q_sampled = np.zeros((npts,3))

    P_m, v_m = solve_contact_discontinuity( left_state, right_state, gamma)

    for i in range(npts):
        Q_sampled[i, :] = sample_solution(P_m, v_m, left_state, right_state, gamma, (x[i]-0.5) / t)

    # set return values in dictionary
    solution = {}
    solution['x'] = x
    solution['rho'] = Q_sampled[:, 0]
    solution['v'] = Q_sampled[:, 1]
    solution['P'] = Q_sampled[:, 2]

    return solution
