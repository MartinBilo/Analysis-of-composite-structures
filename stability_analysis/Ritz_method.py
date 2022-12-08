# Import modules
from   datetime          import datetime
from   matplotlib        import rcParams
rcParams['font.size'] = 8
import matplotlib.pyplot as plt
from   numpy             import arange, array, concatenate, isclose, linspace, ones, pi, sort, tan, tile, zeros
from   numpy.linalg      import solve, eigvals
from   tabulate          import tabulate
from   textwrap          import fill
from   typing            import Dict, List


# The function 'Ritz_method'
def Ritz_method(assumeddeflection : Dict[str, List[float] ], geometry : Dict[str, float], settings : Dict[str, float or str],
                stiffness : Dict[str, List[float] ] ) -> None:
    # Ritz method (section 5.8 from [1])

    # [1] Whitney, J. M. (1987). Structural analysis of laminated anisotropic plates. Technomic Publishing Company
    # [2] Tuttle, M., Singhatanadgid, P., and Hinds, G. (1999). Buckling of composite panels subjected to biaxial loading. Experimental Mechanics, 39(3), pp. 191-201
    # [3] Kassapoglou, C. (2010). Design and analysis of composite structures: With applications to aerospace structures. John Wiley & Sons.

    #
    XkXm     = assumeddeflection['XkXm']
    YlYn     = assumeddeflection['YlYn']
    XkXm_m   = assumeddeflection['XkXm_m']
    YlYn_m   = assumeddeflection['YlYn_m']
    XkddXm   = assumeddeflection['XkddXm']
    YlddYn   = assumeddeflection['YlddYn']
    dXkdXm   = assumeddeflection['dXkdXm']
    dYldYn   = assumeddeflection['dYldYn']
    ddXkddXm = assumeddeflection['ddXkddXm']
    ddYlddYn = assumeddeflection['ddYlddYn']
    XkdXm    = assumeddeflection['XkdXm']
    YldYn    = assumeddeflection['YldYn']
    ddXkdXm  = assumeddeflection['ddXkdXm']
    ddYldYn  = assumeddeflection['ddYldYn']

    #
    a = geometry['a']
    #
    b = geometry['b']

    #
    number_of_data_points = settings['number_of_data_points']
    # The format of the files as which the generated illustrations are saved
    fileformat = settings['fileformat']
    #
    margin = settings['margin']
    #
    mn_max = settings['mn_max']
    #
    mn_min = settings['mn_min']
    # The resolution of the generated illustrations in dots-per-inch
    resolution = settings['resolution']
    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    #
    Colors = [ ( 31 / 255., 119 / 255., 180 / 255.), (114 / 255., 158 / 255., 206 / 255.), (174 / 255., 199 / 255., 232 / 255.),
               (255 / 255., 127 / 255.,  14 / 255.), (255 / 255., 158 / 255.,  74 / 255.), (255 / 255., 187 / 255., 120 / 255.),
               ( 44 / 255., 160 / 255.,  44 / 255.), (103 / 255., 191 / 255.,  92 / 255.), (152 / 255., 223 / 255., 138 / 255.),
               (214 / 255.,  39 / 255.,  40 / 255.), (237 / 255., 102 / 255.,  93 / 255.), (255 / 255., 152 / 255., 150 / 255.),
               (148 / 255., 103 / 255., 189 / 255.), (173 / 255., 139 / 255., 201 / 255.), (197 / 255., 176 / 255., 213 / 255.),
               (140 / 255.,  86 / 255.,  75 / 255.), (168 / 255., 120 / 255., 110 / 255.), (196 / 255., 156 / 255., 148 / 255.),
               (227 / 255., 119 / 255., 194 / 255.), (237 / 255., 151 / 255., 202 / 255.), (247 / 255., 182 / 255., 210 / 255.),
               (127 / 255., 127 / 255., 127 / 255.), (162 / 255., 162 / 255., 162 / 255.), (199 / 255., 199 / 255., 199 / 255.),
               (188 / 255., 189 / 255.,  34 / 255.), (205 / 255., 204 / 255.,  93 / 255.), (219 / 255., 219 / 255., 141 / 255.),
               ( 23 / 255., 190 / 255., 207 / 255.), (109 / 255., 204 / 255., 218 / 255.), (158 / 255., 218 / 255., 229 / 255.) ]

    #
    boundary_conditions = [f'Free (x = 0,a); simply supported (y = 0,b)',
                           f'Simply supported',
                           f'Simply supported (x = 0,a); clamped (y = 0); simply supported (y = b)',
                           f'Simply supported (x = 0,a); clamped (y = 0,b)',
                           f'Clamped (x = 0); simply supported (x = a); free (y = 0,b)',
                           f'Clamped (x = 0); simply supported (x = a; y = 0,b)',
                           f'Clamped (x = 0; y = 0); simply supported (x = a; y = b)',
                           f'Clamped (x = 0); simply supported (x = a); clamped (y = 0,b)',
                           f'Clamped (x = 0,a); free (y = 0,b)',
                           f'Clamped (x = 0,a); simply supported (y = 0,b)',
                           f'Clamped (x = 0,a); clamped (y = 0); simply supported (y = b)',
                           f'Clamped']

    #
    index_boundary_conditions = array( [ [0, 1], [1, 1], [1, 2], [1, 3],
                                         [2, 0], [2, 1], [2, 2], [2, 3],
                                         [3, 0], [3, 1], [3, 2], [3, 3] ] )
    # 0, 2; 0, 3; 1, 0; 2, 2???; 2, 3???; 3,0???; 3, 2???; 0, 4; 1, 4; 2,4; 3,4; 4,4

    #
    situations = len(index_boundary_conditions)

    # An array of shape-(mn_max, situations) containing the error of each normal buckling load in the 1-direction with respect to the previous
    # iteration
    error_force_x_Ritz        = zeros( (mn_max, situations) )
    error_force_x_Ritz[:3, :] = 100

    # An array of shape-(mn_max, situations) containing the error of each normal buckling load in the 2-direction with respect to the previous
    # iteration
    error_force_y_Ritz        = zeros( (mn_max, situations) )
    error_force_y_Ritz[:3, :] = 100

    # An array of shape-(mn_max, situations) containing the error of each shear buckling load in the 12-plane with respect to the previous
    # iteration
    error_force_xy_Ritz           = zeros( (mn_max, 2, situations) )
    error_force_xy_Ritz[:3, :, :] = 100

    # An array of shape-(mn_max, situations) containing the error of each normal buckling load in the 1-direction with respect to the previous
    # iteration
    error_moment_x_Ritz           = zeros( (mn_max, 2, situations) )
    error_moment_x_Ritz[:3, :, :] = 100

    # An array of shape-(mn_max, situations) containing the error of each normal buckling load in the 2-direction with respect to the previous
    # iteration
    error_moment_y_Ritz           = zeros( (mn_max, 2, situations) )
    error_moment_y_Ritz[:3, :, :] = 100

    # An array of shape-(mn_max, situations) containing the normal buckling load in the 1-direction for each iteration
    N_x_Ritz = zeros( (mn_max, situations) )

    # An array of shape-(mn_max, situations) containing the normal buckling load in the 2-direction for each iteration
    N_y_Ritz = zeros( (mn_max, situations) )

    # An array of shape-(mn_max, 2, situations) containing the shear buckling load in the 12-plane for each iteration
    N_xy_Ritz = zeros( (mn_max, 2, situations) )

    # An array of shape-(mn_max, situations) containing the normal buckling load in the 1-direction for each iteration
    M_x_Ritz = zeros( (mn_max, 2, situations) )

    # An array of shape-(mn_max, situations) containing the normal buckling load in the 2-direction for each iteration
    M_y_Ritz = zeros( (mn_max, 2, situations) )

    #
    mn_force_x      = ones(situations, dtype = int)
    mn_force_y      = ones(situations, dtype = int)
    mn_force_xy     = ones(situations, dtype = int)
    mn_force_xy_neg = ones(situations, dtype = int)
    mn_force_xy_pos = ones(situations, dtype = int)
    mn_moment_x     = ones(situations, dtype = int)
    mn_moment_x_neg = ones(situations, dtype = int)
    mn_moment_x_pos = ones(situations, dtype = int)
    mn_moment_y     = ones(situations, dtype = int)
    mn_moment_y_neg = ones(situations, dtype = int)
    mn_moment_y_pos = ones(situations, dtype = int)

    # For each of the situations under consideration
    for i in range(situations):
        # The homogeneous algebraic equation (expression 5.110 from [1]) dictating the shear, buckling load for various boundary conditions
        # can be rewritten as:
        #
        # A + Nᵪ · Bᵪ + Nᵧ · Bᵧ + Nᵪᵧ · Bᵪᵧ = 0
        #
        # where Nᵪ, Nᵧ, and Nᵪᵧ are the in-plane, normal and shear, buckling loads respectively, and A, Bᵪ, Bᵧ, and Bᵪᵧ are (m x n) by (m x n)
        # matrices which contain the load independent and dependent components of the homogeneous algebraic equation respectively.
        # D[0, 2] = 0; D[1, 2] = 0
        # An array of shape-(m by n, m by n) containing the shear load independent components of the homogeneous algebraic equation
        # (expression 6.43 from [1])
        A =     D[0, 0] *  ddXkddXm[:, :, index_boundary_conditions[i, 0] ]   *     YlYn[:, :, index_boundary_conditions[i, 1] ]    \
          +     D[0, 1] * (  XkddXm[:, :, index_boundary_conditions[i, 0] ].T *   YlddYn[:, :, index_boundary_conditions[i, 1] ]    \
          +                  XkddXm[:, :, index_boundary_conditions[i, 0] ]   *   YlddYn[:, :, index_boundary_conditions[i, 1] ].T) \
          +     D[1, 1] *      XkXm[:, :, index_boundary_conditions[i, 0] ]   * ddYlddYn[:, :, index_boundary_conditions[i, 1] ]    \
          + 4 * D[2, 2] *    dXkdXm[:, :, index_boundary_conditions[i, 0] ]   *   dYldYn[:, :, index_boundary_conditions[i, 1] ]    \
          + 2 * D[0, 2] * ( ddXkdXm[:, :, index_boundary_conditions[i, 0] ]   *    YldYn[:, :, index_boundary_conditions[i, 1] ]    \
          +                 ddXkdXm[:, :, index_boundary_conditions[i, 0] ].T *    YldYn[:, :, index_boundary_conditions[i, 1] ].T) \
          + 2 * D[1, 2] * (   XkdXm[:, :, index_boundary_conditions[i, 0] ].T *  ddYldYn[:, :, index_boundary_conditions[i, 1] ].T  \
          +                   XkdXm[:, :, index_boundary_conditions[i, 0] ]   *  ddYldYn[:, :, index_boundary_conditions[i, 1] ] )

        # An array of shape-(m by n, m by n) containing the shear load dependent components of the homogeneous algebraic equation (equation
        # 6.43 from [1])
        B_force_x = dXkdXm[:, :, index_boundary_conditions[i, 0] ] * YlYn[:, :, index_boundary_conditions[i, 1] ]

        B_force_y = XkXm[:, :, index_boundary_conditions[i, 0] ] * dYldYn[:, :, index_boundary_conditions[i, 1] ]

        # Symmetric
        B_force_xy = XkdXm[:, :, index_boundary_conditions[i, 0] ].T * YldYn[:, :, index_boundary_conditions[i, 1] ].T \
                   + XkdXm[:, :, index_boundary_conditions[i, 0] ]   * YldYn[:, :, index_boundary_conditions[i, 1] ]

        B_moment_x = dXkdXm[:, :, index_boundary_conditions[i, 0] ] * YlYn_m[:, :, index_boundary_conditions[i, 1] ]

        B_moment_y = XkXm_m[:, :, index_boundary_conditions[i, 0] ] * dYldYn[:, :, index_boundary_conditions[i, 1] ]

        #
        A_force_x = solve(A, B_force_x)

        #
        A_force_y = solve(A, B_force_y)

        #
        A_force_xy = solve(A, B_force_xy)

        #
        A_moment_x = solve(A, B_moment_x)

        #
        A_moment_y = solve(A, B_moment_y)

        # A counter indicating the number of terms in the 1 (m) and 2-direction (n) taken into account during the successive evaluation of the
        # buckling load
        mn    = 1

        #
        error_Ritz = 100

        # margin = 0 to circumvent stalling of the convergence of the buckling load; mismatch between eigenvalues at boundaries and curve due
        # to numerical errors: https://en.wikipedia.org/wiki/Lanczos_algorithm#Numerical_stability ; see [1]

        # M_x and M_y are theoretically identical for + and -, numerically this does not hold unfortunately

        #
        while (abs(error_Ritz) > margin and mn < mn_max) or (mn < mn_min):

            #
            mn += 1

            #
            index = (tile(linspace(0, mn - 1, mn), (mn, 1) ) + tile(linspace(0, mn_max * (mn - 1), mn), (mn, 1) ).T).flatten().astype(int)

            #
            if (abs(error_force_x_Ritz[mn_force_x[i] - 1, i] ) > margin) or (mn_force_x[i] < mn_min):
                #
                mn_force_x[i] += 1

                # equation 6.43 from [1]
                n = eigvals(A_force_x[index.reshape(-1,1), index] )

                N_x_Ritz[mn_force_x[i] - 1, i] = max(- 1 / n[ (n.real > 0) ].real)

                #
                if mn_force_x[i] > 2:
                    # Substitute the error of the buckling load with respect to the previous iteration
                    error_force_x_Ritz[mn_force_x[i] - 1, i] = abs( (N_x_Ritz[mn_force_x[i] - 2, i] - N_x_Ritz[mn_force_x[i] - 1, i] ) \
                        / N_x_Ritz[mn_force_x[i] - 2, i] ) * 100

            #
            if (abs(error_force_y_Ritz[mn_force_y[i] - 1, i] ) > margin) or (mn_force_y[i] < mn_min):
                #
                mn_force_y[i] += 1

                # equation 6.43 from [1]
                n = eigvals(A_force_y[index.reshape(-1,1), index] )
                N = - 1 / n[ (n.real > 0) ].real
                N_y_Ritz[mn_force_y[i] - 1, i] = sort(N)[-1]

                #
                if mn_force_y[i] > 2:
                    # Substitute the error of the buckling load with respect to the previous iteration
                    error_force_y_Ritz[mn_force_y[i] - 1, i] = abs( (N_y_Ritz[mn_force_y[i] - 2, i] - N_y_Ritz[mn_force_y[i] - 1, i] ) \
                        / N_y_Ritz[mn_force_y[i] - 2, i] ) * 100

            #
            if (max(abs(error_force_xy_Ritz[mn_force_xy[i] - 1, :, i] ) ) > margin) or (mn_force_xy[i] < mn_min):
                #
                mn_force_xy[i] += 1

                # equation 6.43 from [1]
                n = eigvals(A_force_xy[index.reshape(-1,1), index] )
                N = 1 / n[ (n.real != 0) ].real

                # Selection due to numerical, round-off errors
                N_neg = - N[N > 0]
                N_pos = - N[N < 0]
                if abs(error_force_xy_Ritz[mn_force_xy[i] - 2, 0, i] ) > margin:
                    mn_force_xy_neg[i] = mn_force_xy[i]
                    N_xy_Ritz[mn_force_xy[i] - 1, 0, i] = max(N_neg)
                if abs(error_force_xy_Ritz[mn_force_xy[i] - 2, 1, i] ) > margin:
                    mn_force_xy_pos[i] = mn_force_xy[i]
                    N_xy_Ritz[mn_force_xy[i] - 1, 1, i] = min(N_pos)

                #
                if mn_force_xy[i] > 2:
                    # Substitute the error of the buckling load with respect to the previous iteration
                    if abs(error_force_xy_Ritz[mn_force_xy_neg[i] - 2, 0, i] ) > margin:
                        error_force_xy_Ritz[mn_force_xy_neg[i] - 1, 0, i] = abs( (N_xy_Ritz[mn_force_xy_neg[i] - 2, 0, i] - N_xy_Ritz[mn_force_xy_neg[i] - 1, 0, i] ) \
                            / N_xy_Ritz[mn_force_xy_neg[i] - 2, 0, i] ) * 100
                    if abs(error_force_xy_Ritz[mn_force_xy_pos[i] - 2, 1, i] ) > margin:
                        error_force_xy_Ritz[mn_force_xy_pos[i] - 1, 1, i] = abs( (N_xy_Ritz[mn_force_xy_pos[i] - 2, 1, i] - N_xy_Ritz[mn_force_xy_pos[i] - 1, 1, i] ) \
                            / N_xy_Ritz[mn_force_xy_pos[i] - 2, 1, i] ) * 100

            #
            if (max(abs(error_moment_x_Ritz[mn_moment_x[i] - 1, :, i] ) ) > margin) or (mn_moment_x[i] < mn_min):
                #
                mn_moment_x[i] += 1

                # equation 6.43 from [1]
                n = eigvals(A_moment_x[index.reshape(-1,1), index] )
                N = 1 / n[ (n.real != 0) ].real

                # Selection due to numerical, round-off errors
                N_neg = - N[N > 0]
                N_pos = - N[N < 0]
                if abs(error_moment_x_Ritz[mn_moment_x[i] - 2, 0, i] ) > margin:
                    mn_moment_x_neg[i] = mn_moment_x[i]
                    M_x_Ritz[mn_moment_x[i] - 1, 0, i] = max(N_neg)
                if abs(error_moment_x_Ritz[mn_moment_x[i] - 2, 1, i] ) > margin:
                    mn_moment_x_pos[i] = mn_moment_x[i]
                    M_x_Ritz[mn_moment_x[i] - 1, 1, i] = min(N_pos)

                #
                if mn_moment_x[i] > 2:
                    # Substitute the error of the buckling load with respect to the previous iteration
                    if abs(error_moment_x_Ritz[mn_moment_x_neg[i] - 2, 0, i] ) > margin:
                        error_moment_x_Ritz[mn_moment_x_neg[i] - 1, 0, i] = abs( (M_x_Ritz[mn_moment_x_neg[i] - 2, 0, i] - M_x_Ritz[mn_moment_x_neg[i] - 1, 0, i] ) \
                            / M_x_Ritz[mn_moment_x_neg[i] - 2, 0, i] ) * 100
                    if abs(error_moment_x_Ritz[mn_moment_x_pos[i] - 2, 1, i] ) > margin:
                        error_moment_x_Ritz[mn_moment_x_pos[i] - 1, 1, i] = abs( (M_x_Ritz[mn_moment_x_pos[i] - 2, 1, i] - M_x_Ritz[mn_moment_x_pos[i] - 1, 1, i] ) \
                            / M_x_Ritz[mn_moment_x_pos[i] - 2, 1, i] ) * 100

            if (max(abs(error_moment_y_Ritz[mn_moment_x[i] - 1, :, i] ) ) > margin) or (mn_moment_y[i] < mn_min):
                #
                mn_moment_y[i] += 1

                # equation 6.43 from [1]
                n = eigvals(A_moment_y[index.reshape(-1,1), index] )
                N = 1 / n[ (n.real != 0) ].real

                # Selection due to numerical, round-off errors
                N_neg = - N[N > 0]
                N_pos = - N[N < 0]
                if abs(error_moment_y_Ritz[mn_moment_y[i] - 2, 0, i] ) > margin:
                    mn_moment_y_neg[i] = mn_moment_y[i]
                    M_y_Ritz[mn_moment_y[i] - 1, 0, i] = max(N_neg)
                if abs(error_moment_y_Ritz[mn_moment_y[i] - 2, 1, i] ) > margin:
                    mn_moment_y_pos[i] = mn_moment_y[i]
                    M_y_Ritz[mn_moment_y[i] - 1, 1, i] = min(N_pos)

                #
                if mn_moment_y[i] > 2:
                    # Substitute the error of the buckling load with respect to the previous iteration
                    if abs(error_moment_y_Ritz[mn_moment_y_neg[i] - 2, 0, i] ) > margin:
                        error_moment_y_Ritz[mn_moment_y_neg[i] - 1, 0, i] = abs( (M_y_Ritz[mn_moment_y_neg[i] - 2, 0, i] - M_y_Ritz[mn_moment_y_neg[i] - 1, 0, i] ) \
                            / M_y_Ritz[mn_moment_y_neg[i] - 2, 0, i] ) * 100
                    if abs(error_moment_y_Ritz[mn_moment_y_pos[i] - 2, 1, i] ) > margin:
                        error_moment_y_Ritz[mn_moment_y_pos[i] - 1, 1, i] = abs( (M_y_Ritz[mn_moment_y_pos[i] - 2, 1, i] - M_y_Ritz[mn_moment_y_pos[i] - 1, 1, i] ) \
                            / M_y_Ritz[mn_moment_y_pos[i] - 2, 1, i] ) * 100

        #
        mn_reference = max(mn_force_x[i], mn_force_y[i] )

        #
        index_reference = (tile(linspace(0, mn_reference - 1, mn_reference), (mn_reference, 1) ) \
                        + tile(linspace(0, mn_max * (mn_reference - 1), mn_reference), (mn_reference, 1) ).T).flatten().astype(int)

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NᵪNᵧ-plane in which the interaction
        # curve is restricted (90° <= θ <= 180°)
        theta = pi / 180 * linspace(90, 180, num = number_of_data_points // 4)

        #
        n_x_Ritz = zeros(number_of_data_points // 4)

        #
        n_y_Ritz = zeros(number_of_data_points // 4)

        #
        n_x = eigvals(A_force_x[index_reference.reshape(-1,1), index_reference] )
        n_x = - 1 / n_x[ (n_x.real != 0) ].real
        n_x_Ritz[0] = sort(n_x[n_x.real < 0] )[-1]
        n_y_Ritz[0] = 0

        #
        n_x_Ritz[number_of_data_points // 4 - 1] = 0
        n_y = eigvals(A_force_y[index_reference.reshape(-1,1), index_reference] )
        n_y = - 1 / n_y[ (n_y.real != 0) ].real
        n_y_Ritz[number_of_data_points // 4 - 1] = sort(n_y[n_y.real < 0] )[-1]

        #
        for j in range(1, number_of_data_points // 4 - 1):

            #
            C_x = A_force_x[index_reference.reshape(-1,1), index_reference] - A_force_y[index_reference.reshape(-1,1), index_reference] / tan(theta[j] )

            # equation 6.43 from [1]
            n = eigvals(C_x)
            N = - 1 / n[ (n.real > 0) ].real

            # Selection due to numerical, round-off errors
            n_x_Ritz[j] = sort(N)[-1]

            # Nᵪ = - Nᵧ * tan theta
            n_y_Ritz[j] = - n_x_Ritz[j] / tan(theta[j] )

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        #
        plt.plot(n_x_Ritz, n_y_Ritz, color = Colors[0], linewidth = 1, label = f'm  = n  = {mn_reference}')

        #
        plt.plot(N_x_Ritz[mn_force_x[i] - 1, i], 0, color = Colors[0], linestyle = 'None', marker = '.', label = f'm\u1D6A = n\u1D6A = {mn_force_x[i]}')

        #
        plt.plot(0, N_y_Ritz[mn_force_y[i] - 1, i], color = Colors[0], linestyle = 'None', marker = '^',  markersize = 4, label = f'm\u1D67 = n\u1D67 = {mn_force_y[i]}')

        # Format the xlabel
        plt.xlabel(f'N\u1D6A [N/m]')

        #
        plt.xlim(right = 0)

        #
        plt.gca().invert_xaxis()

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'N\u1D67 [N/m]')

        #
        plt.ylim(top = 0)

        #
        plt.gca().invert_yaxis()

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Ritz_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Ritz_method/NxNy/{i:02d}_{boundary_conditions[i].replace(" ", "_")}.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()



        #
        mn_reference = max(mn_force_x[i], mn_moment_x[i], mn_moment_y[i] )

        #
        index_reference = (tile(linspace(0, mn_reference - 1, mn_reference), (mn_reference, 1) ) \
                        + tile(linspace(0, mn_max * (mn_reference - 1), mn_reference), (mn_reference, 1) ).T).flatten().astype(int)

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NᵪNᵧ-plane in which the interaction
        # curve is restricted (90° <= θ <= 180°)
        theta = pi / 180 * linspace(0, 180, num = number_of_data_points // 2)

        #
        n_x_Ritz = zeros( [number_of_data_points // 2, 2] )

        #
        m_x_Ritz = zeros(number_of_data_points // 2)

        #
        m_y_Ritz = zeros(number_of_data_points // 2)

        #
        m_x = eigvals(A_moment_x[index_reference.reshape(-1,1), index_reference] )
        m_x = - 1 / m_x[ (m_x.real != 0) ].real
        m_x_Ritz[0]  = sort(m_x[m_x > 0] )[0]
        m_x_Ritz[-1] = sort(m_x[m_x < 0] )[-1]

        #
        m_y = eigvals(A_moment_y[index_reference.reshape(-1,1), index_reference] )
        m_y = - 1 / m_y[ (m_y.real != 0) ].real
        m_y_Ritz[0]  = sort(m_y[m_y > 0] )[0]
        m_y_Ritz[-1] = sort(m_y[m_y < 0] )[-1]

        #
        for j in range(1, number_of_data_points // 2 - 1):

            #
            C_x = - tan(theta[j] ) * A_force_x[index_reference.reshape(-1,1), index_reference] + A_moment_x[index_reference.reshape(-1,1), index_reference]

            #
            C_y = - tan(theta[j] ) * A_force_x[index_reference.reshape(-1,1), index_reference] + A_moment_y[index_reference.reshape(-1,1), index_reference]

            # equation 6.43 from [1]
            n = eigvals(C_x)
            N = - 1 / n[ (n.real != 0) ].real
            if theta[j] < pi / 2:
                m_x_Ritz[j] = sort(N[N > 0] )[0]
            else:
                m_x_Ritz[j] = sort(N[N < 0] )[-1]

            # equation 6.43 from [1]
            n = eigvals(C_y)
            N = - 1 / n[ (n.real != 0) ].real
            if theta[j] < pi / 2:
                m_y_Ritz[j] = sort(N[N > 0] )[0]
            else:
                m_y_Ritz[j] = sort(N[N < 0] )[-1]

            # Nᵪ = - Nᵧ * tan theta
            n_x_Ritz[j, 0] = - m_x_Ritz[j] * tan(theta[j] )

            n_x_Ritz[j, 1] = - m_y_Ritz[j] * tan(theta[j] )

        #
        n_x = eigvals(A_force_x[index_reference.reshape(-1,1), index_reference] )
        n_x = - 1 / n_x[ (n_x.real != 0) ].real
        n_x = sort(n_x[n_x < 0] )[-1]

        m_x_Ritz = concatenate( (m_x_Ritz[theta < pi / 2], zeros(1), m_x_Ritz[theta > pi / 2] ) )
        m_y_Ritz = concatenate( (m_y_Ritz[theta < pi / 2], zeros(1), m_y_Ritz[theta > pi / 2] ) )
        n_x_Ritz = concatenate( (n_x_Ritz[theta < pi / 2, :], array( [ [n_x, n_x] ] ), n_x_Ritz[theta > pi / 2, :]), axis = 0)

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        #
        plt.plot(n_x_Ritz[:, 0], m_x_Ritz, color = Colors[0], linewidth = 1, label = f'M\u1D6A (m  = n   = {mn_reference})')

        #
        plt.plot(0, M_x_Ritz[mn_moment_x_neg[i] - 1, 0, i], color = Colors[0], linestyle = 'None', marker = '.', label = f'M\u1D6A (m\u1D6A = n\u1D6A = {mn_moment_x_neg[i]})')

        #
        plt.plot(0, M_x_Ritz[mn_moment_x_pos[i] - 1, 1, i], color = Colors[0], linestyle = 'None', marker = '^',  markersize = 4, label = f'M\u1D6A (m\u1D6A = n\u1D6A = {mn_moment_x_pos[i]})')

        #
        plt.plot(n_x_Ritz[:, 1], m_y_Ritz, color = Colors[3], linewidth = 1, label = f'M\u1D67 (m  = n   = {mn_reference})')

        #
        plt.plot(0, M_y_Ritz[mn_moment_y_neg[i] - 1, 0, i], color = Colors[3], linestyle = 'None', marker = '.', label = f'M\u1D67 (m\u1D67 = n\u1D67 = {mn_moment_y_neg[i]})')

        #
        plt.plot(0, M_y_Ritz[mn_moment_y_pos[i] - 1, 1, i], color = Colors[3], linestyle = 'None', marker = '^',  markersize = 4, label = f'M\u1D67 (m\u1D67 = n\u1D67 = {mn_moment_y_pos[i]})')

        #
        plt.plot(N_x_Ritz[mn_force_x[i] - 1, i], 0, color = Colors[6], linestyle = 'None', marker = '.', label = f'N\u1D6A (m\u1D6A = n\u1D6A = {mn_force_x[i]})')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'N\u1D6A [N/m]')

        #
        plt.xlim(right = 0)

        #
        plt.gca().invert_xaxis()

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'M [Nm]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Ritz_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Ritz_method/NxM/{i:02d}_{boundary_conditions[i].replace(" ", "_")}.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()


        #
        mn_reference = max(mn_force_y[i], mn_moment_x[i], mn_moment_y[i] )

        #
        index_reference = (tile(linspace(0, mn_reference - 1, mn_reference), (mn_reference, 1) ) \
                        + tile(linspace(0, mn_max * (mn_reference - 1), mn_reference), (mn_reference, 1) ).T).flatten().astype(int)

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NᵪNᵧ-plane in which the interaction
        # curve is restricted (90° <= θ <= 180°)
        theta = pi / 180 * linspace(0, 180, num = number_of_data_points // 2)

        #
        n_y_Ritz = zeros( [number_of_data_points // 2, 2] )

        #
        m_x_Ritz = zeros(number_of_data_points // 2)

        #
        m_y_Ritz = zeros(number_of_data_points // 2)

        #
        m_x = eigvals(A_moment_x[index_reference.reshape(-1,1), index_reference] )
        m_x = - 1 / m_x[ (m_x.real != 0) ].real
        m_x_Ritz[0]  = sort(m_x[m_x > 0] )[0]
        m_x_Ritz[-1] = sort(m_x[m_x < 0] )[-1]

        #
        m_y = eigvals(A_moment_y[index_reference.reshape(-1,1), index_reference] )
        m_y = - 1 / m_y[ (m_y.real != 0) ].real
        m_y_Ritz[0]  = sort(m_y[m_y > 0] )[0]
        m_y_Ritz[-1] = sort(m_y[m_y < 0] )[-1]

        #
        for j in range(1, number_of_data_points // 2 - 1):

            #
            C_x = - tan(theta[j] ) * A_force_y[index_reference.reshape(-1,1), index_reference] + A_moment_x[index_reference.reshape(-1,1), index_reference]

            #
            C_y = - tan(theta[j] ) * A_force_y[index_reference.reshape(-1,1), index_reference] + A_moment_y[index_reference.reshape(-1,1), index_reference]

            # equation 6.43 from [1]
            n = eigvals(C_x)
            N = - 1 / n[ (n.real != 0) ].real
            if theta[j] < pi / 2:
                m_x_Ritz[j] = sort(N[N > 0] )[0]
            else:
                m_x_Ritz[j] = sort(N[N < 0] )[-1]

            # equation 6.43 from [1]
            n = eigvals(C_y)
            N = - 1 / n[ (n.real != 0) ].real
            if theta[j] < pi / 2:
                m_y_Ritz[j] = sort(N[N > 0] )[0]
            else:
                m_y_Ritz[j] = sort(N[N < 0] )[-1]

            # Nᵪ = - Nᵧ * tan theta
            n_y_Ritz[j, 0] = - m_x_Ritz[j] * tan(theta[j] )

            n_y_Ritz[j, 1] = - m_y_Ritz[j] * tan(theta[j] )

        #
        n_y = eigvals(A_force_y[index_reference.reshape(-1,1), index_reference] )
        n_y = - 1 / n_y[ (n_y.real != 0) ].real
        n_y = sort(n_y[n_y < 0] )[-1]

        m_x_Ritz = concatenate( (m_x_Ritz[theta < pi / 2], zeros(1), m_x_Ritz[theta > pi / 2] ) )
        m_y_Ritz = concatenate( (m_y_Ritz[theta < pi / 2], zeros(1), m_y_Ritz[theta > pi / 2] ) )
        n_y_Ritz = concatenate( (n_y_Ritz[theta < pi / 2, :], array( [ [n_y, n_y] ] ), n_y_Ritz[theta > pi / 2, :]), axis = 0)

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        #
        plt.plot(n_y_Ritz[:, 0], m_x_Ritz, color = Colors[0], linewidth = 1, label = f'M\u1D6A (m  = n   = {mn_reference})')

        #
        plt.plot(0, M_x_Ritz[mn_moment_x_neg[i] - 1, 0, i], color = Colors[0], linestyle = 'None', marker = '.', label = f'M\u1D6A (m\u1D6A = n\u1D6A = {mn_moment_x_neg[i]})')

        #
        plt.plot(0, M_x_Ritz[mn_moment_x_pos[i] - 1, 1, i], color = Colors[0], linestyle = 'None', marker = '^',  markersize = 4, label = f'M\u1D6A (m\u1D6A = n\u1D6A = {mn_moment_x_pos[i]})')

        #
        plt.plot(n_y_Ritz[:, 1], m_y_Ritz, color = Colors[3], linewidth = 1, label = f'M\u1D67 (m  = n   = {mn_reference})')

        #
        plt.plot(0, M_y_Ritz[mn_moment_y_neg[i] - 1, 0, i], color = Colors[3], linestyle = 'None', marker = '.', label = f'M\u1D67 (m\u1D67 = n\u1D67 = {mn_moment_y_neg[i]})')

        #
        plt.plot(0, M_y_Ritz[mn_moment_y_pos[i] - 1, 1, i], color = Colors[3], linestyle = 'None', marker = '^',  markersize = 4, label = f'M\u1D67 (m\u1D67 = n\u1D67 = {mn_moment_y_pos[i]})')

        #
        plt.plot(N_y_Ritz[mn_force_y[i] - 1, i], 0, color = Colors[6], linestyle = 'None', marker = '.', label = f'N\u1D67 (m\u1D67 = n\u1D67 = {mn_force_y[i]})')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'N\u1D67 [N/m]')

        #
        plt.xlim(right = 0)

        #
        plt.gca().invert_xaxis()

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'M [Nm]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Ritz_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Ritz_method/NyM/{i:02d}_{boundary_conditions[i].replace(" ", "_")}.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()


        #
        mn_reference = max(mn_force_x[i], mn_force_y[i], mn_force_xy_neg[i], mn_force_xy_pos[i] )

        #
        index_reference = (tile(linspace(0, mn_reference - 1, mn_reference), (mn_reference, 1) ) \
                        + tile(linspace(0, mn_max * (mn_reference - 1), mn_reference), (mn_reference, 1) ).T).flatten().astype(int)

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NNᵪᵧ-plane in which the interaction
        # curve is restricted (0° <= θ <= 180°)
        theta = pi / 180 * linspace(0, 180, num = number_of_data_points // 2)

        #
        n_x_Ritz = zeros(number_of_data_points // 2)

        #
        n_y_Ritz = zeros(number_of_data_points // 2)

        #
        n_xy_Ritz = zeros( [number_of_data_points // 2, 2] )

        #
        n_x_Ritz[0] = 0
        n_y_Ritz[0] = 0

        #
        n_xy = eigvals(A_force_xy[index_reference.reshape(-1,1), index_reference] )
        n_xy = - 1 / n_xy[ (n_xy.real != 0) ].real
        n_xy_Ritz[0, :] = sort(n_xy[n_xy.real > 0] )[0]

        #
        for j in range(1, number_of_data_points // 2):
            #
            C_x = - tan(theta[j] ) * A_force_x[index_reference.reshape(-1,1), index_reference] + A_force_xy[index_reference.reshape(-1,1), index_reference]

            #
            C_y = - tan(theta[j] ) * A_force_y[index_reference.reshape(-1,1), index_reference] + A_force_xy[index_reference.reshape(-1,1), index_reference]

            # equation 6.43 from [1]
            n   = eigvals(C_x)
            n_x = - 1 / n[ (n.real != 0) ].real

            # equation 6.43 from [1]
            n   = eigvals(C_y)
            n_y = - 1 / n[ (n.real != 0) ].real

            # Selection due to numerical, round-off errors
            if theta[j] < pi / 2:
                n_xy_Ritz[j, 0] = sort(n_x[n_x.real > 0] )[0]
                n_xy_Ritz[j, 1] = sort(n_y[n_y.real > 0] )[0]
                n_x_Ritz[j] = - n_xy_Ritz[j, 0] * tan(theta[j] )
                n_y_Ritz[j] = - n_xy_Ritz[j, 1] * tan(theta[j] )
            elif isclose(theta[j], pi / 2):
                n_xy_Ritz[j, 0] = 0
                n_xy_Ritz[j, 1] = 0
                n_x_Ritz[j]  = N_x_Ritz[mn_force_x[i], i]
                n_y_Ritz[j]  = N_y_Ritz[mn_force_y[i], i]
            elif theta[j] > pi / 2:
                n_xy_Ritz[j, 0] = sort(n_x[n_x.real < 0] )[-1]
                n_xy_Ritz[j, 1] = sort(n_y[n_y.real < 0] )[-1]
                n_x_Ritz[j] = - n_xy_Ritz[j, 0] * tan(theta[j] )
                n_y_Ritz[j] = - n_xy_Ritz[j, 1] * tan(theta[j] )

        #
        n_x = eigvals(A_force_x[index_reference.reshape(-1,1), index_reference] )
        n_x = - 1 / n_x[ (n_x.real != 0) ].real
        n_x = sort(n_x[n_x < 0] )[-1]

        #
        n_y = eigvals(A_force_y[index_reference.reshape(-1,1), index_reference] )
        n_y = - 1 / n_y[ (n_y.real != 0) ].real
        n_y = sort(n_y[n_y < 0] )[-1]

        n_x_Ritz  = concatenate( (n_x_Ritz[theta < pi / 2], array( [n_x] ), n_x_Ritz[theta > pi / 2] ) )
        n_y_Ritz  = concatenate( (n_y_Ritz[theta < pi / 2], array( [n_y] ), n_y_Ritz[theta > pi / 2] ) )
        n_xy_Ritz = concatenate( (n_xy_Ritz[theta < pi / 2, :], array( [ [0, 0] ] ), n_xy_Ritz[theta > pi / 2, :]), axis = 0)

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        #
        plt.plot(n_x_Ritz, n_xy_Ritz[:, 0], color = Colors[0], linewidth = 1, label = f'N\u1D6A  (m   = n   = {mn_reference})')

        #
        plt.plot(N_x_Ritz[mn_force_x[i] - 1, i], 0, color = Colors[0], linestyle = 'None', marker = '.', label = f'N\u1D6A  (m\u1D6A  = n\u1D6A  = {mn_force_x[i]})')

        #
        plt.plot(n_y_Ritz, n_xy_Ritz[:, 1], color = Colors[3], linewidth = 1, label = f'N\u1D67  (m   = n   = {mn_reference})')

        #
        plt.plot(N_y_Ritz[mn_force_y[i] - 1, i], 0, color = Colors[3], linestyle = 'None', marker = '.', label = f'N\u1D67  (m\u1D6A  = n\u1D6A  = {mn_force_y[i]})')

        #
        plt.plot(0, N_xy_Ritz[mn_force_xy_neg[i] - 1, 0, i], color = Colors[6], linestyle = 'None', marker = '.',
            label = f'N\u1D6A\u1D67 (m\u1D6A\u1D67 = n\u1D6A\u1D67 = {mn_force_xy_neg[i]})')

        #
        plt.plot(0, N_xy_Ritz[mn_force_xy_pos[i] - 1, -1, i], color = Colors[6], linestyle = 'None', marker = '^',  markersize = 4,
            label = f'N\u1D6A\u1D67 (m\u1D6A\u1D67 = n\u1D6A\u1D67 = {mn_force_xy_pos[i]})')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'N [N/m]')

        #
        plt.xlim(right = 0)

        #
        plt.gca().invert_xaxis()

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'N\u1D6A\u1D67 [N/m]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Ritz_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Ritz_method/NNxy/{i:02d}_{boundary_conditions[i].replace(" ", "_")}.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()


        #
        mn_reference = max(mn_moment_x[i], mn_moment_y[i], mn_force_xy_neg[i], mn_force_xy_pos[i] )

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NNᵪᵧ-plane in which the interaction
        # curve is restricted (0° <= θ <= 180°)
        theta = pi / 180 * linspace(0, 360 - (360 / number_of_data_points), num = number_of_data_points)

        #
        m_x_Ritz = zeros(number_of_data_points)

        #
        m_y_Ritz = zeros(number_of_data_points)

        #
        n_xy_Ritz = zeros( [number_of_data_points, 2] )

        #
        m_x_Ritz[0] = 0
        m_y_Ritz[0] = 0

        #
        n_xy = eigvals(A_force_xy[index_reference.reshape(-1,1), index_reference] )
        n_xy = - 1 / n_xy[ (n_xy.real != 0) & (isclose(n_xy.imag, 0) ) ].real
        n_xy_Ritz[0, :] = sort(n_xy[n_xy.real > 0] )[0]

        #
        for j in range(1, number_of_data_points):
            #
            C_x = - tan(theta[j] ) * A_moment_x[index_reference.reshape(-1,1), index_reference] + A_force_xy[index_reference.reshape(-1,1), index_reference]

            #
            C_y = - tan(theta[j] ) * A_moment_y[index_reference.reshape(-1,1), index_reference] + A_force_xy[index_reference.reshape(-1,1), index_reference]

            # equation 6.43 from [1]
            m   = eigvals(C_x)
            m_x = - 1 / m[ (m.real != 0) & (isclose(m.imag, 0) ) ].real

            # equation 6.43 from [1]
            m   = eigvals(C_y)
            m_y = - 1 / m[ (m.real != 0) & (isclose(m.imag, 0) ) ].real

            # Selection due to numerical, round-off errors
            if theta[j] < pi / 2:
                n_xy_Ritz[j, 0] = min(m_x[m_x.real > 0] )
                n_xy_Ritz[j, 1] = min(m_y[m_y.real > 0] )
                m_x_Ritz[j] = - n_xy_Ritz[j, 0] * tan(theta[j] )
                m_y_Ritz[j] = - n_xy_Ritz[j, 1] * tan(theta[j] )
            elif isclose(theta[j], pi / 2):
                n_xy_Ritz[j, 0] = 0
                n_xy_Ritz[j, 1] = 0

                # equation 6.43 from [1]
                m           = eigvals(A_moment_x[index_reference.reshape(-1,1), index_reference])
                m_x_Ritz[j] = max(- 1 / m[ (m.real > 0) & (isclose(m.imag, 0) ) ].real)

                # equation 6.43 from [1]
                m           = eigvals(A_moment_y[index_reference.reshape(-1,1), index_reference])
                m_y_Ritz[j] = max(- 1 / m[ (m.real > 0) & (isclose(m.imag, 0) ) ].real)
            elif theta[j] < 3 * pi / 2:
                n_xy_Ritz[j, 0] = max(m_x[m_x.real < 0] )
                n_xy_Ritz[j, 1] = max(m_y[m_y.real < 0] )
                m_x_Ritz[j] = - n_xy_Ritz[j, 0] * tan(theta[j] )
                m_y_Ritz[j] = - n_xy_Ritz[j, 1] * tan(theta[j] )
            elif isclose(theta[j], 3 * pi / 2):
                n_xy_Ritz[j, 0] = 0
                n_xy_Ritz[j, 1] = 0

                # equation 6.43 from [1]
                m           = eigvals(A_moment_x[index_reference.reshape(-1,1), index_reference])
                m_x_Ritz[j] = min(- 1 / m[ (m.real < 0) & (isclose(m.imag, 0) ) ].real)

                # equation 6.43 from [1]
                m           = eigvals(A_moment_y[index_reference.reshape(-1,1), index_reference])
                m_y_Ritz[j] = min(- 1 / m[ (m.real < 0) & (isclose(m.imag, 0) ) ].real)
            else:
                n_xy_Ritz[j, 0] = min(m_x[m_x.real > 0] )
                n_xy_Ritz[j, 1] = min(m_y[m_y.real > 0] )
                m_x_Ritz[j] = - n_xy_Ritz[j, 0] * tan(theta[j] )
                m_y_Ritz[j] = - n_xy_Ritz[j, 1] * tan(theta[j] )

        #
        m_x_Ritz  = concatenate( (m_x_Ritz, array( [m_x_Ritz[0] ] ) ) )
        m_y_Ritz  = concatenate( (m_y_Ritz, array( [m_y_Ritz[0] ] ) ) )
        n_xy_Ritz = concatenate( (n_xy_Ritz, array( [n_xy_Ritz[0, :] ] ) ), axis = 0)

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        #
        plt.plot(m_x_Ritz, n_xy_Ritz[:, 0], color = Colors[0], linewidth = 1, label = f'M\u1D6A  (m   = n   = {mn_reference})')

        #
        plt.plot(M_x_Ritz[mn_moment_x_neg[i] - 1, 0, i], 0, color = Colors[0], linestyle = 'None', marker = '.', label = f'M\u1D6A  (m\u1D6A  = n\u1D6A  = {mn_moment_x_neg[i]})')

        #
        plt.plot(M_x_Ritz[mn_moment_x_pos[i] - 1, 1, i], 0, color = Colors[0], linestyle = 'None', marker = '^',  markersize = 4, label = f'M\u1D6A  (m\u1D6A  = n\u1D6A  = {mn_moment_x_pos[i]})')

        #
        plt.plot(m_y_Ritz, n_xy_Ritz[:, 1], color = Colors[3], linewidth = 1, label = f'M\u1D67  (m   = n   = {mn_reference})')

        #
        plt.plot(M_y_Ritz[mn_moment_y_neg[i] - 1, 0, i], 0, color = Colors[3], linestyle = 'None', marker = '.', label = f'M\u1D67  (m\u1D6A  = n\u1D6A  = {mn_moment_y_neg[i]})')

        #
        plt.plot(M_y_Ritz[mn_moment_y_pos[i] - 1, 1, i], 0, color = Colors[3], linestyle = 'None',  marker = '^',  markersize = 4, label = f'M\u1D67  (m\u1D6A  = n\u1D6A  = {mn_moment_y_pos[i]})')

        #
        plt.plot(0, N_xy_Ritz[mn_force_xy_neg[i] - 1, 0, i], color = Colors[6], linestyle = 'None', marker = '.',
            label = f'N\u1D6A\u1D67 (m\u1D6A\u1D67 = n\u1D6A\u1D67 = {mn_force_xy_neg[i]})')

        #
        plt.plot(0, N_xy_Ritz[mn_force_xy_pos[i] - 1, -1, i], color = Colors[6], linestyle = 'None', marker = '^',  markersize = 4,
            label = f'N\u1D6A\u1D67 (m\u1D6A\u1D67 = n\u1D6A\u1D67 = {mn_force_xy_pos[i]})')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Add a vertical line through the origin
        plt.axvline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'M [Nm]')

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'N\u1D6A\u1D67 [N/m]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Ritz_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Ritz_method/MNxy/{i:02d}_{boundary_conditions[i].replace(" ", "_")}.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()


    # Create and open the text file
    with open(f'{simulation}/data/stability_analysis/Ritz_method/Ritz_method.txt', 'w', encoding = 'utf8') as textfile:
        # Add a description of the contents
        textfile.write(fill(f'Ritz method') )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the source and a timestamp
        textfile.write(fill(f'Source: stability_analysis/Ritz_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
        textfile.write(tabulate(array( [boundary_conditions, N_x_Ritz[mn_force_x - 1, arange(N_x_Ritz.shape[1] ) ], \
                    error_force_x_Ritz[mn_force_x - 1, arange(error_force_x_Ritz.shape[1] ) ], mn_force_x] ).T.tolist(),
                    headers = ['Boundary conditions', 'N\u1D6A [N/m]', '\u03F5\u1D6A [%]', 'm\u1D6A = n\u1D6A [-]'], \
                    colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.0f', '.5e', '.2e', '.0f') ) )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
        textfile.write(tabulate(array( [boundary_conditions, N_y_Ritz[mn_force_y - 1, arange(N_y_Ritz.shape[1] ) ], \
                    error_force_y_Ritz[mn_force_y - 1, arange(error_force_y_Ritz.shape[1] ) ], mn_force_y] ).T.tolist(),
                    headers = ['Boundary conditions', 'N\u1D67 [N/m]', '\u03F5\u1D67 [%]', 'm\u1D67 = n\u1D67 [-]'], \
                    colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.0f', '.5e', '.2e', '.0f') ) )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
        textfile.write(tabulate(array( [boundary_conditions, N_xy_Ritz[mn_force_xy_neg - 1, 0, arange(N_xy_Ritz.shape[2] ) ], \
                    N_xy_Ritz[mn_force_xy_pos - 1, 1, arange(N_xy_Ritz.shape[2] ) ], error_force_xy_Ritz[mn_force_xy_neg - 1, 0, arange(error_force_xy_Ritz.shape[2] ) ], \
                    error_force_xy_Ritz[mn_force_xy_pos - 1, 1,  arange(error_force_xy_Ritz.shape[2] ) ], mn_force_xy_neg, mn_force_xy_pos] ).T.tolist(),
                    headers = ['Boundary conditions', 'N\u1D6A\u1D67', '[N/m]', '\u03F5\u1D6A\u1D67', '[%]', 'm\u1D67 = ', 'n\u1D67 [-]'], \
                    colalign = ('left', 'right', 'left', 'right', 'left', 'right', 'left'), floatfmt = ('.0f', '.5e', '.5e', '.2e', '.2e', \
                        '.0f', '.0f') ) )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
        textfile.write(tabulate(array( [boundary_conditions, M_x_Ritz[mn_moment_x_neg - 1, 0, arange(M_x_Ritz.shape[2] ) ], M_x_Ritz[mn_moment_x_pos - 1, 1, arange(M_x_Ritz.shape[2] ) ], \
                    error_moment_x_Ritz[mn_moment_x_neg - 1, 0, arange(error_moment_x_Ritz.shape[2] ) ], error_moment_x_Ritz[mn_moment_x_pos - 1, 1, arange(error_moment_x_Ritz.shape[2] ) ], \
                    mn_moment_x_neg, mn_moment_x_pos] ).T.tolist(),
                    headers = ['Boundary conditions', 'M\u1D6A', '[Nm]', '\u03F5\u1D6A', '[%]', 'm\u1D6A = ', 'n\u1D6A [-]'], \
                    colalign = ('left', 'right', 'left', 'right', 'left', 'right', 'left'), \
                    floatfmt = ('.0f', '.5e', '.5e', '.2e', '.2e', '.0f', '.0f') ) )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
        textfile.write(tabulate(array( [boundary_conditions, M_y_Ritz[mn_moment_y_neg - 1, 0, arange(M_y_Ritz.shape[2] ) ], M_y_Ritz[mn_moment_y_pos - 1, 1, arange(M_y_Ritz.shape[2] ) ], \
                    error_moment_y_Ritz[mn_moment_y_neg - 1, 0, arange(error_moment_y_Ritz.shape[2] ) ], error_moment_y_Ritz[mn_moment_y_pos - 1, 1, arange(error_moment_y_Ritz.shape[2] ) ], \
                    mn_moment_y_neg, mn_moment_y_pos] ).T.tolist(),
                    headers = ['Boundary conditions', 'M\u1D67', '[Nm]', '\u03F5\u1D67', '[%]', 'm\u1D67 = ', 'n\u1D67 [-]'], \
                    colalign = ('left', 'right', 'left', 'right', 'left', 'right', 'left'), \
                    floatfmt = ('.0f', '.5e', '.5e', '.2e', '.2e', '.0f', '.0f') ) )

    # return(N_Ritz)