# Import modules
from   datetime          import datetime
from   matplotlib        import rcParams
rcParams['font.size'] = 8
import matplotlib.pyplot as plt
from   numpy             import any, argmax, amin, amax, stack, array, concatenate, einsum, inf, isclose, linspace, meshgrid, moveaxis, \
                                newaxis, pi, repeat, sin, tan, tile, zeros
from   numpy.ma          import masked_where
from   numpy.linalg      import solve, eig, eigvals
from   tabulate          import tabulate
from   textwrap          import fill
from   typing            import Dict, List


# The function 'convergence_plot'
def convergence_plot(x_Galerkin: List[float], mn_x: float, y_Galerkin: List[float], mn_y: float, xy_Galerkin: List[float],
                     mn_xy_neg: float, mn_xy_pos: float, fileformat: str, label: str, name: str, resolution: float,
                     simulation: float, margin: bool = False) -> None:
    # Use the Tableau 30 colors (consisting of Tableau 10, Tableau 10 Medium, and Tableau 10 Light from [3]) for the figure
    Colors = [ (31 / 255., 119 / 255., 180 / 255.), (114 / 255., 158 / 255., 206 / 255.), (174 / 255., 199 / 255., 232 / 255.),
              (255 / 255., 127 / 255.,  14 / 255.), (255 / 255., 158 / 255.,  74 / 255.), (255 / 255., 187 / 255., 120 / 255.),
               (44 / 255., 160 / 255.,  44 / 255.), (103 / 255., 191 / 255.,  92 / 255.), (152 / 255., 223 / 255., 138 / 255.),
              (214 / 255.,  39 / 255.,  40 / 255.), (237 / 255., 102 / 255.,  93 / 255.), (255 / 255., 152 / 255., 150 / 255.),
              (148 / 255., 103 / 255., 189 / 255.), (173 / 255., 139 / 255., 201 / 255.), (197 / 255., 176 / 255., 213 / 255.),
              (140 / 255.,  86 / 255.,  75 / 255.), (168 / 255., 120 / 255., 110 / 255.), (196 / 255., 156 / 255., 148 / 255.),
              (227 / 255., 119 / 255., 194 / 255.), (237 / 255., 151 / 255., 202 / 255.), (247 / 255., 182 / 255., 210 / 255.),
              (127 / 255., 127 / 255., 127 / 255.), (162 / 255., 162 / 255., 162 / 255.), (199 / 255., 199 / 255., 199 / 255.),
              (188 / 255., 189 / 255.,  34 / 255.), (205 / 255., 204 / 255.,  93 / 255.), (219 / 255., 219 / 255., 141 / 255.),
               (23 / 255., 190 / 255., 207 / 255.), (109 / 255., 204 / 255., 218 / 255.), (158 / 255., 218 / 255., 229 / 255.)]

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    # Add the parameter under consideration for the in-plane, compressive, normal force in the 1-direction
    plt.plot(linspace(2, mn_x, num=mn_x - 1), x_Galerkin[1:mn_x], color=Colors[0], linewidth=1, label=' N\u2081')

    # Add the parameter under consideration for the in-plane, compressive, normal force in the 2-direction
    plt.plot(linspace(2, mn_y, num=mn_y - 1), y_Galerkin[1:mn_y], color=Colors[3], linewidth=1, label=' N\u2082')

    # Add the parameter under consideration for the negative, in-plane, shear force in the 12-plane
    plt.plot(linspace(2, mn_xy_neg, num=mn_xy_neg - 1), xy_Galerkin[1:mn_xy_neg, 0], color=Colors[6], linewidth=1,
             label='-N\u2081\u2082')

    # Add the parameter under consideration for the positive, in-plane, shear force in the 12-plane
    plt.plot(linspace(2, mn_xy_pos, num=mn_xy_pos - 1), xy_Galerkin[1:mn_xy_pos, 1], color=Colors[9], linewidth=1,
             label=' N\u2081\u2082')

    # Format the xlabel
    plt.xlabel(f'm = n [-]')

    # Set the upper and lower limit of the number of terms along the x-axis
    plt.xlim(2, max(mn_x, mn_y, mn_xy_neg, mn_xy_pos))

    # Format the ylabel
    plt.ylabel(label)

    # If the y-axis of the plot is logarithmic
    if margin:
        # Format the yticks
        plt.yscale('symlog', linthreshy=margin / 10)
    else:
        # Add a horizontal line through the origin
        plt.axhline(color='black', linewidth=0.5, linestyle=(0, (5, 10)))

    # Add a legend
    plt.legend(loc='center left', bbox_to_anchor=(
        1, 0.5), fancybox=True, edgecolor='inherit')

    # Add text with the moniker of the simulation, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
             fontsize=6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Galerkin_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
             fontsize=6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Save the figure
    plt.savefig(f'{simulation}/illustrations/stability_analysis/Galerkin_method/{name}_convergence.{fileformat}',
                dpi=resolution, bbox_inches='tight')

    # Close the figure
    plt.close()


# The function 'deformation_plot'
def deformation_plot(a: float, b: float, G: List[float], m: List[float], n: List[float], mn: float, mn_max: float,
                     name: str, fileformat: str, resolution: float, simulation: str, x: List[float], y: List[float]):
    #
    index = (tile(linspace(0, mn - 1, mn), (mn, 1)) + tile(linspace(0, mn_max * (mn - 1), mn), (mn, 1)).T).flatten().astype(int)

    #
    index_even = index[(m[0, index] + n[0, index]) % 2 == 0]

    #
    index_odd = index[(m[0, index] + n[0, index]) % 2 != 0]

    # Evaluate the eigenvalue problem corresponding to the 'm + n = even' set of equations (equations 5.103 and 6.38 from [1], expression A4
    # from [2] and equations 6.16, 6.17 and 6.18 from [3])
    N_even, H_even = eig(G[index_even.reshape(-1, 1), index_even])

    # Evaluate the eigenvalue problem corresponding to the 'm + n = odd' set of equations (equations 5.103 and 6.38 from [1], expression A4
    # from [2] and equations 6.16, 6.17 and 6.18 from [3])
    N_odd,  H_odd = eig(G[index_odd.reshape(-1, 1), index_odd])

    # An array of shape-(m x n) containing the eigenvalues of the homogeneous equations
    N = concatenate((N_even, N_odd))

    # Mask the eigenvalues which are infinite, complex, or less than zero
    N = masked_where((N == inf) | (N.imag != 0) | (N.real < 0), N.real)

    # Determine the index of the smallest, negative, buckling load
    index_n = argmax(- 1 / N)

    # Determine the amount of eigenvalues corresponding to the 'm + n = even' set of equations
    length = len(N_even)

    # If the smallest, negative, buckling load corresponds to the 'm + n = even' set of equations
    if index_n < length:
        # An array of shape-() containing the eigenvector corresponding to the smallest, negative, buckling load
        H = H_even[:, index_n].real
        # An array containing the
        m_index = (m[0, index] )[ (m[0, index] + n[0, index] ) % 2 == 0]
        #
        n_index = (n[0, index] )[ (m[0, index] + n[0, index] ) % 2 == 0]
    # If the smallest, negative, buckling load corresponds to the 'm + n = odd' set of equations
    else:
        # An array of shape-() containing the eigenvector corresponding to the smallest, negative, buckling load
        H = H_odd[:, index_n - length].real
        #
        m_index = (m[0, index] )[ (m[0, index] + n[0, index] ) % 2 != 0]
        #
        n_index = (n[0, index] )[ (m[0, index] + n[0, index] ) % 2 != 0]

    # An array of shape-(number_of_elements_x, number_of_elemens_y) containing the out-of-plane displacement corresponding to the smallest,
    # negative, buckling load
    w = einsum('k,ijk->ij', H, sin(einsum('k,ij->ijk', m_index, pi * x) ) * sin(einsum('k,ij->ijk', n_index, pi * y) ) )

    # Ensure the largest out-of-plane displacement is negative to clarify the resulting illustration
    w = ( (max(w.flatten(), key = abs) > 0) * - 1 \
      +   (max(w.flatten(), key = abs) < 0) *   1) * w

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    # Add the out-of-plane displacement corresponding to the smallest, negative, buckling load
    plt.imshow(w, cmap = 'PuBu_r', extent = [0, a, 0, b], interpolation = 'bilinear', aspect = 'equal', origin = 'lower')

    # Format the xlabel
    plt.xlabel(f'x [m]')

    # Format the ylabel
    plt.ylabel(f'y [m]')

    # Add text with the moniker of the simulation, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
             fontsize=6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Galerkin_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
             fontsize=6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Save the figure
    plt.savefig(f'{simulation}/illustrations/stability_analysis/Galerkin_method/Deformation_{name}.{fileformat}',
                dpi = resolution, bbox_inches = 'tight')

    # Close the figure
    plt.close()


#
def shear_buckling_load(G, index_reference_even, index_reference_odd, positive: bool = False):
    # The governing equation resulting from the Galerkin method is given by (equations 5.103 and 6.38 from [1], expression A4 from [2] and
    # equations 6.16, 6.17 and 6.18 from [3])
    #
    # E + F₁ · N₁ + F₂ · N₂ + F₁₂ · N₁₂ = 0
    #
    # which corresponds to a classical eigenvalue problem in which the eigenvalue corresponds to the buckling load and the eigenmode to the
    # buckling mode. For a singular load this expression can be rewritten to yield
    #      _               _              _           _
    #     |          1      |            |      1      |
    # det | E \ F + --- · I |= 0 --> det | G + --- · I |= 0 --> det (G - λ I) = 0
    #     |_         N     _|            |_     N     _|
    #
    # where I is the identity matrix, λ = - 1 / N the eigenvalue of matrix G and G the solution to the linear matrix equation E · G = F.

    #
    n_2_even = eigvals(
        G[:, index_reference_even.reshape(-1, 1), index_reference_even])

    #
    n_2_odd = eigvals(
        G[:, index_reference_odd.reshape(-1, 1), index_reference_odd])

    #
    n = concatenate((n_2_even, n_2_odd), axis=1)

    #
    if positive:
        #
        n = masked_where( (n == inf) | (n.imag != 0) | (n.real > 0), n.real)

        #
        N_Galerkin = amin(- 1 / n, axis=1)

    #
    else:
        #
        n = masked_where( (n == inf) | (n.imag != 0) | (n.real < 0), n.real)

        #
        N_Galerkin = amax(- 1 / n, axis=1)

    #
    return(N_Galerkin)


#
def envelope_plot(G_1: List[float], G_2: List[float], m: List[float], n: List[float], mn_max: float, mn_reference: float,
                  axes: List[str], labels: List[str], simulation: str, theta: List[float]) -> None:
    # Use the Tableau 30 colors (consisting of Tableau 10, Tableau 10 Medium, and Tableau 10 Light from [3]) for the figure
    Colors = [(31 / 255., 119 / 255., 180 / 255.), (114 / 255., 158 / 255., 206 / 255.), (174 / 255., 199 / 255., 232 / 255.),
              (255 / 255., 127 / 255.,  14 / 255.), (255 / 255., 158 /
                                                     255.,  74 / 255.), (255 / 255., 187 / 255., 120 / 255.),
              (44 / 255., 160 / 255.,  44 / 255.), (103 / 255., 191 /
                                                    255.,  92 / 255.), (152 / 255., 223 / 255., 138 / 255.),
              (214 / 255.,  39 / 255.,  40 / 255.), (237 / 255., 102 /
                                                     255.,  93 / 255.), (255 / 255., 152 / 255., 150 / 255.),
              (148 / 255., 103 / 255., 189 / 255.), (173 / 255., 139 /
                                                     255., 201 / 255.), (197 / 255., 176 / 255., 213 / 255.),
              (140 / 255.,  86 / 255.,  75 / 255.), (168 / 255., 120 /
                                                     255., 110 / 255.), (196 / 255., 156 / 255., 148 / 255.),
              (227 / 255., 119 / 255., 194 / 255.), (237 / 255., 151 /
                                                     255., 202 / 255.), (247 / 255., 182 / 255., 210 / 255.),
              (127 / 255., 127 / 255., 127 / 255.), (162 / 255., 162 /
                                                     255., 162 / 255.), (199 / 255., 199 / 255., 199 / 255.),
              (188 / 255., 189 / 255.,  34 / 255.), (205 / 255., 204 /
                                                     255.,  93 / 255.), (219 / 255., 219 / 255., 141 / 255.),
              (23 / 255., 190 / 255., 207 / 255.), (109 / 255., 204 / 255., 218 / 255.), (158 / 255., 218 / 255., 229 / 255.)]

    #
    index_reference = (tile(linspace(0, mn_reference - 1,            mn_reference), (mn_reference, 1) )
                     + tile(linspace(0, mn_max * (mn_reference - 1), mn_reference), (mn_reference, 1) ).T).flatten().astype(int)

    #
    index_reference_even = index_reference[ (m[0, index_reference] + n[0, index_reference] ) % 2 == 0]

    #
    index_reference_odd = index_reference[ (m[0, index_reference] + n[0, index_reference] ) % 2 != 0]

    #
    length = len(theta)

    #
    number_of_load_cases = G_1.size // mn_max**4

    #
    n_1_Galerkin = zeros( [length, number_of_load_cases] )

    #
    n_2_Galerkin = zeros( [length, number_of_load_cases] )

    #
    if number_of_load_cases > 1:
        g_1 = moveaxis(G_1, 2, 0)
        g_2 = moveaxis(repeat(G_2[:, :, newaxis], number_of_load_cases, axis=2), 2, 0)
    else:
        g_1 = moveaxis(G_1[:, :, newaxis], 2, 0)
        g_2 = moveaxis(G_2[:, :, newaxis], 2, 0)

    #
    for j in range(1, length):
        # The plane defined by the two loads under consideration (N₁ and N₂), which contains the interaction curve dictating buckling, is
        # depicted in figure 1 below. In addition, the angle θ between both loads is specified.
        #
        #                         N₂
        #                      θ = 0, 2π
        #                         ^
        #                         |
        #                     \ θ |
        #                      \<-|
        #                       \ |
        #                        \|
        #   θ = π / 2 ------------+------------> θ = 3π / 2, N₁
        #                         |
        #                         |
        #                         |
        #                         |
        #                         |
        #                       θ = π
        #
        # Figure XX.1: Illustration of the N₁N₂-plane and the angle
        #                θ between the two loads.
        #
        # From this figure can be derived that the angle θ is related to both loads via
        #
        #            N₁
        # tan θ = - ----
        #            N₂
        #
        # resulting in
        #
        # N₁ = - N₂ ⋅ tan θ .
        #
        # Substitution of this expression in the governing equation resulting from the Galerkin method for two in-plane loads  (equations 5.103
        # and 6.38 from [1], expression A4 from [2] and equations 6.16, 6.17 and 6.18 from [3]) yields
        #
        # E + F₁ · N₁ + F₂ · N₂ = 0 --> E + N₂ · (F₂ - tan θ · F₁) = 0
        #
        # which corresponds to a classical eigenvalue problem in which the eigenvalue corresponds to the buckling load (N₂) and the eigenmode
        # to the buckling mode. This expression can be rewritten to yield
        #      _                                _              _                          _
        #     |                          1       |            |                     1      |
        # det | E \ (F₂ - tan θ · F₁) + ---- · I |= 0 --> det | G₂ - tan θ · G₁  + --- · I |= 0 --> det (G - λ I) = 0
        #     |_                         N₂     _|            |_                    N₂    _|
        #
        # where I is the identity matrix, λ = - 1 / N₂ the eigenvalue of matrix G, and G₁ and G₂ the solution to the linear matrix equation
        # E · G = F as defined in function ''.

        #
        n_12_even = eigvals(- tan(theta[j]) * g_1[:, index_reference_even.reshape(-1, 1), index_reference_even]
                                            + g_2[:, index_reference_even.reshape(-1, 1), index_reference_even] )

        #
        n_12_odd = eigvals(- tan(theta[j]) * g_1[:, index_reference_odd.reshape(-1, 1), index_reference_odd]
                                           + g_2[:, index_reference_odd.reshape(-1, 1), index_reference_odd] )

        #
        n_12 = concatenate( (n_12_even, n_12_odd), axis=1)

        #
        if (theta[j] > pi / 2) & (theta[j] < 2 * pi / 2):
            #
            n_12 = masked_where((n_12 == inf) | (n_12.imag != 0) | (n_12.real < 0), n_12.real)

            #
            n_2_Galerkin[j, :] = amax(- 1 / n_12, axis=1)

        #
        else:
            #
            n_12 = masked_where((n_12 == inf) | (n_12.imag != 0) | (n_12.real > 0), n_12.real)

            #
            n_2_Galerkin[j, :] = amin(- 1 / n_12, axis=1)

        #
        n_1_Galerkin[j, :] = - n_2_Galerkin[j, :] * tan(theta[j] )

    #
    if any(isclose(theta, 0)):
        #
        n_2_Galerkin[isclose(theta, 0), :] = shear_buckling_load(g_2, index_reference_even, index_reference_odd, positive=True)

    #
    if any(isclose(theta, pi / 2)):
        #
        n_1_Galerkin[isclose(theta, pi / 2), :] = shear_buckling_load(g_1, index_reference_even, index_reference_odd)

    #
    if any(isclose(theta, pi)):
        #
        n_2_Galerkin[isclose(theta, pi), :] = shear_buckling_load(g_2, index_reference_even, index_reference_odd)

    #
    if any(isclose(theta, 3 * pi / 2)):
        #
        n_1_Galerkin[isclose(theta, 3 * pi / 2), :] = shear_buckling_load(g_1, index_reference_even, index_reference_odd, positive=True)

    #
    if any(isclose(theta, 2 * pi)):
        #
        n_2_Galerkin[isclose(theta, 2 * pi), :] = shear_buckling_load(g_2, index_reference_even, index_reference_odd, positive=True)

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    #
    for i in range(number_of_load_cases):
        #
        plt.plot(n_1_Galerkin[:, i], n_2_Galerkin[:, i], color=Colors[3 * i], linewidth=1, label=labels[i])

    # Format the xlabel
    plt.xlabel(axes[0])

    #
    if theta[-1] > pi:
        # Add a vertical line through the origin
        plt.axvline(color='black', linewidth=0.5, linestyle=(0, (5, 10)))
    else:
        #
        plt.xlim(right=0)

        # Invert the x-axis
        plt.gca().invert_xaxis()

    # Format the xticks
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # Format the ylabel
    plt.ylabel(axes[1])

    #
    if theta[0] != 0:
        #
        plt.ylim(top=0)

        # Invert the y-axis
        plt.gca().invert_yaxis()
    else:
        # Add a horizontal line through the origin
        plt.axhline(color='black', linewidth=0.5, linestyle=(0, (5, 10)))

    # Format the yticks
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Add text with the moniker of the simulation, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
             fontsize=6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Galerkin_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
             fontsize=6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)


# The function 'shear_buckling_convergence'
def shear_buckling_convergence(G: List[float], index_even: List[float], index_odd: List[float], mn: float,
                               N_Galerkin: List[float], error_Galerkin: List[float], positive: bool = False) -> None:
    #
    mn += 1

    # Evaluate the eigvalsenvalue problem corresponding to the m + n even set of equations (equation 5.103 from [1] and equation 6.17 from [2])
    N_even = eigvals(G[index_even.reshape(-1, 1), index_even])

    # Evaluate the eigvalsenvalue problem corresponding to the m + n odd set of equations (equation 5.103 from [1] and equation 6.17 from [2])
    N_odd = eigvals(G[index_odd.reshape(-1, 1), index_odd])

    #
    N = concatenate((N_even, N_odd))

    #
    if positive:
        #
        N = masked_where((N == inf) | (N.imag != 0) | (N.real > 0), N.real)

        #
        N_Galerkin[mn - 1] = amin(- 1 / N, axis=0)

    #
    else:
        #
        N = masked_where((N == inf) | (N.imag != 0) | (N.real < 0), N.real)

        #
        N_Galerkin[mn - 1] = amax(- 1 / N, axis=0)

    #
    if mn > 2:
        # Substitute the error of the buckling load with respect to the previous iteration
        error_Galerkin[mn - 1] = abs((N_Galerkin[mn - 2] - N_Galerkin[mn - 1] ) / N_Galerkin[mn - 2] ) * 100

    #
    return(mn, N_Galerkin, error_Galerkin)


# The function 'Galerkin_method'
def Galerkin_method(geometry: Dict[str, float], settings: Dict[str, float or str], stiffness: Dict[str, List[float]],
                    data: bool = False, illustrations: bool = False) -> List[float]:
    # Galerkin method for the in-plane, shear, buckling load assuming simply supported boundary conditions of a rectangular, midplane
    # symmetric laminate (section 6.5 from [1])

    # [1] Whitney, J. M. (1987). Structural analysis of laminated anisotropic plates. Technomic Publishing Company
    # [2] Tuttle, M., Singhatanadgid, P., and Hinds, G. (1999). Buckling of composite panels subjected to biaxial loading. Experimental Mechanics, 39(3), pp. 191-201
    # [3] Kassapoglou, C. (2010). Design and analysis of composite structures: With applications to aerospace structures. John Wiley & Sons.
    # [4] Gerrard, C. (2020). accessed 5 August 2020.
    # <https://public.tableau.com/profile/chris.gerrard#!/vizhome/TableauColors/ColorPaletteswithRGBValues>

    # The plate dimension (width) in the x-direction
    a = geometry['a']
    # The plate dimension (length) in the y-direction
    b = geometry['b']

    # The number of data points used for the generated illustrations
    number_of_data_points = settings['number_of_data_points']
    # The format of the files as which the generated illustrations are saved
    fileformat = settings['fileformat']
    # The upper bound for error measures
    margin = settings['margin']
    # The maximum number of terms to be taken into account
    mn_max = settings['mn_max']
    # The minimum number of terms to be taken into account
    mn_min = settings['mn_min']
    # The number of elements in the 1-direction
    number_of_elements_x = settings['number_of_elements_x']
    # The number of elements in the 2-direction
    number_of_elements_y = settings['number_of_elements_y']
    # The resolution of the generated illustrations in dots-per-inch
    resolution = settings['resolution']
    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    # The aspect ratio of the plate
    r = a / b

    # An array of shape-(mn_max², mn_max²) containing the number of half-waves in the 1-direction for each of the mn_max² homogeneous equations
    # (equations 5.103 and 6.38 from [1], expression A4 from [2] and equations 6.16, 6.17 and 6.18 from [3])
    m = tile(repeat(linspace(1, mn_max, mn_max), mn_max), (mn_max**2, 1))

    # An array of shape-(mn_max², mn_max²) containing the number of half-waves in the 2-direction for each of the mn_max² homogeneous equations
    # (equations 5.103 and 6.38 from [1], expression A4 from [2] and equations 6.16, 6.17 and 6.18 from [3])
    n = tile(tile(linspace(1, mn_max, mn_max), mn_max), (mn_max**2, 1))

    # An array of shape-(mn_max², mn_max²) containing the first element of the coefficient of the governing equation (equations 5.103 and 6.38
    # from [1], expression A4 from [2] and equations 6.16, 6.17 and 6.18 from [3])
    M = ((m == m.T) & (n == n.T)) * 1

    # An array of shape-(mn_max², mn_max²) containing the second element of the coefficient of the governing equation (equations 5.103 and 6.38
    # from [1], expression A4 from [2] and equations 6.16, 6.17 and 6.18 from [3])
    t = (m**2 - m.T**2) * (n**2 - n.T**2)

    # An array of shape-(mn_max², mn_max²) containing the coefficient of the governing equation (equations 5.103 and 6.38 from [1], expression
    # A4 from [2] and equations 6.16, 6.17 and 6.18 from [3])
    T = zeros((mn_max**2, mn_max**2))
    T[t != 0] = m[t != 0].T * n[t != 0].T / t[t != 0]

    # An array of shape-(mn_max², mn_max²) containing the components of the governing equation which are independent of the in-plane, forces
    # (equations 5.103 and 6.38 from [1], expression A4 from [2] and equations 6.16, 6.17 and 6.18 from [3])
    E = (pi**4 * (D[0, 0] * m**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * m**2 * n**2 * r**2 + D[1, 1] * n**4 * r**4)) * M \
        - 32 * m * n * r * pi**2 * \
        (D[0, 2] * (m**2 + m.T**2) + D[1, 2] * (n**2 + n.T**2)) * T

    # An array of shape-(mn_max², mn_max²) containing the components of the governing equation which are dependent on the in-plane, normal and
    # shear forces (equations 5.103 and 6.38 from [1], expression A4 from [2] and equations 6.16, 6.17 and 6.18 from [3])
    F_1 = (m**2 * a**2 / pi**2) * M
    F_2 = (n**2 * b**2 / pi**2) * M
    F_12 = 32 * m * n * r**2 * a**2 * T

    # A counter for each in-plane, buckling load indicating the number of terms in the 1 (m) and 2-direction (n) taken into account during the
    # successive evaluation of each buckling load (m = n)
    mn_x = 1
    mn_y = 1
    mn_xy_neg = 1
    mn_xy_pos = 1

    # An array of shape-(mn_max) containing the error of each in-plane, normal buckling load in the 1-direction with respect to the previous
    # iteration
    error_1_Galerkin = zeros(mn_max)
    error_1_Galerkin[:3] = 100

    # An array of shape-(mn_max) containing the error of each in-plane, normal buckling load in the 2-direction with respect to the previous
    # iteration
    error_2_Galerkin = zeros(mn_max)
    error_2_Galerkin[:3] = 100

    # An array of shape-(mn_max) containing the error of each in-plane, shear buckling load in the 12-plane with respect to the previous
    # iteration
    error_12_Galerkin = zeros((mn_max, 2))
    error_12_Galerkin[:3, :] = 100

    # An array of shape-(mn_max) containing the in-plane, normal buckling load in the 1-direction for each iteration
    N_1_Galerkin = zeros(mn_max)

    # An array of shape-(mn_max) containing the in-plane, normal buckling load in the 2-direction for each iteration
    N_2_Galerkin = zeros(mn_max)

    # An array of shape-(mn_max) containing the in-plane, shear buckling load in the 12-plane for each iteration
    N_12_Galerkin = zeros((mn_max, 2))

    # An overall counter indicating the number of terms in the 1 (m) and 2-direction (n) taken into account during the successive evaluation
    # of each buckling load (m = n)
    mn = 1

    #
    error_Galerkin = 100

    # The governing equation resulting from the Galerkin method is given by (equations 5.103 and 6.38 from [1], expression A4 from [2] and
    # equations 6.16, 6.17 and 6.18 from [3])
    #
    # E + F₁ · N₁ + F₂ · N₂ + F₁₂ · N₁₂ = 0
    #
    # which corresponds to a classical eigenvalue problem in which the eigenvalue corresponds to the buckling load and the eigenmode to the
    # buckling mode. For a singular load this expression can be rewritten to yield
    #      _               _              _           _
    #     |          1      |            |      1      |
    # det | E \ F + --- · I |= 0 --> det | G + --- · I |= 0 --> det (G - λ I) = 0
    #     |_         N     _|            |_     N     _|
    #
    # where I is the identity matrix, λ = - 1 / N the eigenvalue of matrix G and G the solution to the linear matrix equation E · G = F.

    # An array of shape-(mn_max², mn_max²) containing the solution to the linear matrix equation for the in-plane, normal force in the
    # 1-direction
    G_1 = solve(E, F_1)

    # An array of shape-(mn_max², mn_max²) containing the solution to the linear matrix equation for the in-plane, normal force in the
    # 2-direction
    G_2 = solve(E, F_2)

    # An array of shape-(mn_max², mn_max²) containing the solution to the linear matrix equation for the in-plane, shear force in the
    # 12-plane
    G_12 = solve(E, F_12)

    #
    while (error_Galerkin > margin and mn < mn_max) or (mn < mn_min):

        #
        mn += 1

        #
        index = (tile(linspace(0, mn - 1, mn), (mn, 1)) + tile(linspace(0,
                                                                        mn_max * (mn - 1), mn), (mn, 1)).T).flatten().astype(int)

        #
        index_even = index[(m[0, index] + n[0, index]) % 2 == 0]

        #
        index_odd = index[(m[0, index] + n[0, index]) % 2 != 0]

        #
        if (abs(error_1_Galerkin[mn_x - 1]) > margin) or (mn_x < mn_min):
            #
            mn_x, N_1_Galerkin, error_1_Galerkin = shear_buckling_convergence(
                G_1, index_even, index_odd, mn_x, N_1_Galerkin, error_1_Galerkin)

        #
        if (abs(error_2_Galerkin[mn_y - 1]) > margin) or (mn_y < mn_min):
            #
            mn_y, N_2_Galerkin, error_2_Galerkin = shear_buckling_convergence(
                G_2, index_even, index_odd, mn_y, N_2_Galerkin, error_2_Galerkin)

        #
        if (abs(error_12_Galerkin[mn_xy_neg - 1, 0]) > margin) or (mn_xy_neg < mn_min):
            #
            mn_xy_neg, N_12_Galerkin[:, 0], error_12_Galerkin[:, 0] = shear_buckling_convergence(
                G_12, index_even, index_odd, mn_xy_neg, N_12_Galerkin[:, 0], error_12_Galerkin[:, 0])

        #
        if (abs(error_12_Galerkin[mn_xy_pos - 1, 1]) > margin) or (mn_xy_pos < mn_min):
            #
            mn_xy_pos, N_12_Galerkin[:, 1], error_12_Galerkin[:, 1] = shear_buckling_convergence(
                G_12, index_even, index_odd, mn_xy_pos, N_12_Galerkin[:, 1], error_12_Galerkin[:, 1], positive=True)

        #
        error_Galerkin = max(error_1_Galerkin[mn - 1], error_2_Galerkin[mn - 1],
                             error_12_Galerkin[mn - 1, 0], error_12_Galerkin[mn - 1, 1])

    #
    N_Galerkin = array([N_1_Galerkin[mn_x - 1], N_2_Galerkin[mn_y - 1],
                        N_12_Galerkin[mn_xy_neg - 1, 0], N_12_Galerkin[mn_xy_pos - 1, 1]])

    if data:
        # Create and open the text file
        with open(f'{simulation}/data/stability_analysis/Galerkin_method.txt', 'w', encoding='utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'Stability analysis of a rectangular, midplane symmetric, laminate plate for uniaxial, in-plane loads'
                                f' according to the Galerkin method'))

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: stability_analysis/Galerkin_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).'))

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
            textfile.write(tabulate(array(
                [linspace(2, mn, num=mn - 1),
                 [f'{buckling_load:.5e}' if buckling_load !=
                     0 else f' ' for buckling_load in N_1_Galerkin[1:mn]],
                 [f'{error:.2e}' if error !=
                     0 else f' ' for error in error_1_Galerkin[1:mn]],
                 [f'{buckling_load:.5e}' if buckling_load !=
                     0 else f' ' for buckling_load in N_2_Galerkin[1:mn]],
                 [f'{error:.2e}' if error !=
                     0 else f' ' for error in error_2_Galerkin[1:mn]],
                 [f'{buckling_load:.5e}' if buckling_load !=
                     0 else f' ' for buckling_load in N_12_Galerkin[1:mn, 0]],
                 [f'{buckling_load:.5e}' if buckling_load !=
                     0 else f' ' for buckling_load in N_12_Galerkin[1:mn, 1]],
                 [f'{error:.2e}' if error !=
                     0 else f' ' for error in error_12_Galerkin[1:mn, 0]],
                 [f'{error:.2e}' if error != 0 else f' ' for error in error_12_Galerkin[1:mn, 1]]]).T.tolist(),
                headers=[f'm = n [-]', f'N\u1D6A [N/m]',           f'\u03F5\u1D6A [%]',
                         f'N\u1D67 [N/m]',           f'\u03F5\u1D67 [%]',
                         f'N\u1D6A\u1D67', f'[N/m]', f'\u03F5\u1D6A\u1D67', f'[%]'],
                colalign=('center', 'center', 'center', 'center',
                          'center', 'right', 'left', 'right', 'left'),
                floatfmt=('.0f', '.5e', '.2e', '.5e', '.2e', '.5e', '.5e', '.2e', '.2e')))

    if illustrations:
        #
        Colors = [(31 / 255., 119 / 255., 180 / 255.), (114 / 255., 158 / 255., 206 / 255.), (174 / 255., 199 / 255., 232 / 255.),
                  (255 / 255., 127 / 255.,  14 / 255.), (255 / 255., 158 /
                                                         255.,  74 / 255.), (255 / 255., 187 / 255., 120 / 255.),
                  (44 / 255., 160 / 255.,  44 / 255.), (103 / 255., 191 /
                                                        255.,  92 / 255.), (152 / 255., 223 / 255., 138 / 255.),
                  (214 / 255.,  39 / 255.,  40 / 255.), (237 / 255., 102 /
                                                         255.,  93 / 255.), (255 / 255., 152 / 255., 150 / 255.),
                  (148 / 255., 103 / 255., 189 / 255.), (173 / 255., 139 /
                                                         255., 201 / 255.), (197 / 255., 176 / 255., 213 / 255.),
                  (140 / 255.,  86 / 255.,  75 / 255.), (168 / 255., 120 /
                                                         255., 110 / 255.), (196 / 255., 156 / 255., 148 / 255.),
                  (227 / 255., 119 / 255., 194 / 255.), (237 / 255., 151 /
                                                         255., 202 / 255.), (247 / 255., 182 / 255., 210 / 255.),
                  (127 / 255., 127 / 255., 127 / 255.), (162 / 255., 162 /
                                                         255., 162 / 255.), (199 / 255., 199 / 255., 199 / 255.),
                  (188 / 255., 189 / 255.,  34 / 255.), (205 / 255., 204 /
                                                         255.,  93 / 255.), (219 / 255., 219 / 255., 141 / 255.),
                  (23 / 255., 190 / 255., 207 / 255.), (109 / 255., 204 / 255., 218 / 255.), (158 / 255., 218 / 255., 229 / 255.)]

        #
        convergence_plot(N_1_Galerkin, mn_x, N_2_Galerkin, mn_y, N_12_Galerkin, mn_xy_neg, mn_xy_pos,
                         fileformat, f'N [N/m]', f'Load', resolution, simulation)

        #
        convergence_plot(error_1_Galerkin, mn_x, error_2_Galerkin, mn_y, error_12_Galerkin, mn_xy_neg, mn_xy_pos,
                         fileformat, f'\u03F5 [%]', f'Error', resolution, simulation, margin=margin)

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NᵪNᵧ-plane in which the interaction
        # curve is restricted (90° <= θ <= 180°)
        theta = pi / 180 * linspace(90, 180, num=number_of_data_points // 4)

        #
        mn_reference = max(mn_x, mn_y)

        #
        axes = array([f'N\u1D6A [N/m]', f'N\u1D67 [N/m]'])

        #
        labels = array([f'm   = n   = {mn_reference}'])

        #
        envelope_plot(G_1, G_2, m, n, mn_max, mn_reference,
                      axes, labels, simulation, theta)

        #
        plt.plot(N_1_Galerkin[mn_x - 1], 0, color=Colors[0], linestyle='None',
                 marker='.', label=f'm\u1D6A  = n\u1D6A  = {mn_x}')

        #
        plt.plot(0, N_2_Galerkin[mn_y - 1], color=Colors[0], linestyle='None',
                 marker='.', label=f'm\u1D67  = n\u1D67  = {mn_y}')

        # Add a legend
        plt.legend(loc='center left', bbox_to_anchor=(
            1, 0.5), fancybox=True, edgecolor='inherit')

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Galerkin_method/NxNy.{fileformat}',
                    dpi=resolution, bbox_inches='tight')

        # Close the figure
        plt.close()

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NNᵪᵧ-plane in which the interaction
        # curve is restricted (0° <= θ <= 180°)
        theta = pi / 180 * linspace(0, 180, num=number_of_data_points // 2)

        #
        mn_reference = max(mn_x, mn_y, mn_xy_neg, mn_xy_pos)

        #
        axes = array([f'N [N/m]', f'N\u1D6A\u1D67 [N/m]'])

        #
        labels = array(
            [f'N\u1D6A  (m   = n   = {mn_reference})', f'N\u1D67  (m   = n   = {mn_reference})'])

        #
        envelope_plot(stack((G_1, G_2), axis=2), G_12, m, n, mn_max,
                      mn_reference, axes, labels, simulation, theta)

        #
        plt.plot(N_1_Galerkin[mn_x - 1], 0, color=Colors[0], linestyle='None',
                 marker='.', label=f'N\u1D6A  (m\u1D6A  = n\u1D6A  = {mn_x})')

        #
        plt.plot(N_2_Galerkin[mn_y - 1], 0, color=Colors[3], linestyle='None',
                 marker='.', label=f'N\u1D67  (m\u1D67  = n\u1D67  = {mn_y})')

        #
        plt.plot(0, N_12_Galerkin[mn_xy_neg - 1, 0], color=Colors[6], linestyle='None', marker='.',
                 label=f'N\u1D6A\u1D67 (m\u1D6A\u1D67 = n\u1D6A\u1D67 = {mn_xy_neg})')

        #
        plt.plot(0, N_12_Galerkin[mn_xy_pos - 1, -1], color=Colors[6], linestyle='None', marker='^',  markersize=4,
                 label=f'N\u1D6A\u1D67 (m\u1D6A\u1D67 = n\u1D6A\u1D67 = {mn_xy_pos})')

        #
        handles, labels = plt.gca().get_legend_handles_labels()

        #
        handles = [handles[0], handles[2], handles[1],
                   handles[3], handles[4], handles[5]]

        #
        labels = [labels[0], labels[2], labels[1],
                  labels[3], labels[4], labels[5]]

        # Add a legend
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(
            1, 0.5), fancybox=True, edgecolor='inherit')

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Galerkin_method/NNxy.{fileformat}',
                    dpi=resolution, bbox_inches='tight')

        # Close the figure
        plt.close()

        #
        x, y = meshgrid(linspace(0, 1, number_of_elements_x),
                        linspace(0, 1, number_of_elements_y))

        deformation_plot(a, b, G_1,  m, n, mn_x,      mn_max, f'Nx',
                         fileformat, resolution, simulation, x, y)
        deformation_plot(a, b, G_2,  m, n, mn_y,      mn_max, f'Ny',
                         fileformat, resolution, simulation, x, y)
        deformation_plot(a, b, G_12, m, n, mn_xy_neg, mn_max,
                         f'Nxy_negative', fileformat, resolution, simulation, x, y)
        deformation_plot(a, b, G_12, m, n, mn_xy_pos, mn_max,
                         f'Nxy_positive', fileformat, resolution, simulation, x, y)

    #
    return(N_Galerkin)
