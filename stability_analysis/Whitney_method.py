# Import modules
from   datetime          import datetime
from   matplotlib        import rcParams
rcParams['font.size'] = 8
import matplotlib.pyplot as plt
from   numpy             import amax, amin, argmax, array, ceil, errstate, finfo, floor, hstack, isnan, linspace, ones, pi, tan, sqrt, vstack, \
                                zeros
from   numpy.ma          import masked_equal, masked_greater_equal, masked_less, masked_less_equal
from   tabulate          import tabulate
from   textwrap          import fill
from   typing            import Dict, List, Tuple


# The function 'number_of_half_waves'
def number_of_half_waves(number_of_data_points : float, k : List[float], r : float, D : List[float] ) -> Tuple[List[float], List[float] ]:

    # Initialize an array of shape-(9, number_of_data_points) containing the number of half-waves in the 1-direction minimizing the biaxial,
    # in-plane, normal, buckling load
    m = ones( (9, number_of_data_points) )

    # Initialize an array of shape-(9, number_of_data_points) containing the number of half-waves in the 2-direction minimizing the biaxial,
    # in-plane, normal, buckling load
    n = ones( (9, number_of_data_points) )

    # The buckling load (N₀ = - Nᵪ = - Nᵧ / k) for a rectangular, specially orthotropic laminate under biaxial, normal, loading is given by
    # (equation 5.72 from [1] and expression 6.5 from [2])
    #
    #       π²(D₁₁m⁴ + 2(D₁₂ + 2D₆₆)m²n²r² + D₂₂n⁴r⁴)
    # N₀ = ------------------------------------------- .
    #                     a²(m²+kn²r²)
    #
    # For these circumstances, it holds that the number of half-waves in one of the two directions is 1 (pages 122 and 123 from [2]). The
    # number of half-waves in the 1-direction which minimizes the buckling load can therefore be derived by equating the derivative
    # of the buckling with respect to this number of half-waves to zero
    #
    # δN₀             2π²m(D₁₁(m⁴ + 2km²r²) + r⁴(- D₂₂ + 2D₁₂k + 4D₆₆k) )
    # --- (n = 1) = ------------------------------------------------------ = 0,
    # δm                                a²(m² + kr²)²
    #
    # from which it follows that
    #                                        _________________________________________________             _________________________
    #               ________                /               /  D₁₁k² - 2D₁₂k - 4D₆₆k + D₂₂ | |            /               /  Dₖ   | |
    # m = 0 V m ≠ \/  - kr²  V m = ±  r    / -  k  ±       / ------------------------------    = ±  r    / -  k  ±       /  -----
    #                                    \/              \/               D₁₁                          \/              \/    D₁₁
    #
    # where Dₖ = D₁₁k² - 2D₁₂k - 4D₆₆k + D₂₂. The same procedure for the number of half-waves in the 2-direction yields
    #
    # δN₀             2π²nr²(D₂₂(kn⁴r⁴ + 2n²r²) - D₁₁k + 2D₁₂ + 4D₆₆)
    # --- (m = 1) = --------------------------------------------------- = 0,
    # δn                              a²(1 + kn²r²)²
    #
    # which results in
    #                 ______                 _________________________________________________             _________________________
    #                / - 1            1     /   1     1     /  D₁₁k² - 2D₁₂k - 4D₆₆k + D₂₂ | |      1     /   1     1     /  Dₖ   | |
    # n = 0 V n ≠   / ------ V n = ± ---   / - --- ± ---   / ------------------------------    = ± ---   / - --- ± ---   /  -----   .
    #             \/    kr²           r  \/     k     k  \/                D₂₂                      r  \/     k     k  \/    D₂₂

    # An array of shape-(k) containing the common stiffness component of the previously derived equations
    D_k = D[0, 0] * k**2 - 2 * (D[0, 1] - 2 * D[2, 2] ) * k + D[1, 1]

    # Substitute the number of half-waves in the 1-direction which minimizes the biaxial, normal, in-plane, buckling load according to the
    # previously derived equations in the corresponding array while ignoring errors due to invalid, imaginary values which result in nan
    with errstate(invalid = 'ignore'):
        m[1:3, :] = [r * sqrt(- k - sqrt(D_k / D[0, 0] ) ), \
                     r * sqrt(- k + sqrt(D_k / D[0, 0] ) ) ]

    # Regulate the theoretical number of half-waves in the 1-direction which minizes the biaxial, normal, in-plane, buckling load, since the
    # number of half-waves can only be a positive integer (m ∈ ℕ₁) (equation 5.68 from [1])
    m[isnan(m) ] = 1
    m[3:5, :]    = floor(m[1:3, :] )
    m[1:3, :]    = ceil(m[1:3, :] )
    m[m == 0]    = 1

    # Substitute the number of half-waves in the 2-direction which minimize the biaxial, normal, in-plane, buckling load according to the
    # previously derived equations in the corresponding array while ignoring errors due to division by a biaxial loading ratio of zero and
    # invalid, imaginary values which both result in nan
    with errstate(divide='ignore', invalid = 'ignore'):
        n[5:7, :] = [1 / r * sqrt(- 1 / k - 1 / k * sqrt(D_k / D[1, 1] ) ), \
                     1 / r * sqrt(- 1 / k + 1 / k * sqrt(D_k / D[1, 1] ) ) ]

    # Regulate the theoretical number of half-waves in the 2-direction which minize the biaxial, normal, in-plane, buckling load, since the
    # number of half-waves can only be a positive integer (m ∈ ℕ₁) (equation 5.68 from [1])
    n[isnan(n) ] = 1
    n[7:, :]     = floor(n[5:7, :] )
    n[5:7, :]    = ceil(n[5:7, :] )
    n[n == 0]    = 1

    # End the function and return the combinations of half-waves in the 1 and 2-direction which minimize the biaxial, normal, in-plane,
    # buckling load
    return(m, n)


# The function 'Whitney_method'
def Whitney_method(geometry : Dict[str, float], settings : Dict[str, float], stiffness : Dict[str, List[float] ], data : bool = False,
                   illustrations : bool = False) -> Tuple[List[float], List[float], List[float] ]:
    # Whitney method for the in-plane, biaxial, normal, buckling load assuming simply supported boundary conditions (section 5.5 from [1]
    # # and section 6.1 from [2] )
    #     [3]: Gerrard, Chris. 2020. accessed 5 August 2020.
    # <https://public.tableau.com/profile/chris.gerrard#!/vizhome/TableauColors/ColorPaletteswithRGBValues>


    # The plate dimension in the x-direction
    a = geometry['a']
    # The plate dimension in the y-direction
    b = geometry['b']

    # The number of data points used for the generated illustrations
    number_of_data_points = settings['number_of_data_points']
    # The format of the files as which the generated illustrations are saved
    fileformat = settings['fileformat']
    # The resolution of the generated illustrations in dots-per-inch
    resolution = settings['resolution']
    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    # Initialize an array of shape-(2) containing the buckling load in the 1 and 2-direction respectively
    N_Whitney = zeros(2)

    # Initialize an array of shape-(2) containing the number of half-waves in the 1-direction corresponding to the minimum buckling load
    m_Whitney = ones(2)

    # Initialize an array of shape-(2) containing the number of half-waves in the 2-direction corresponding to the minimum buckling load
    n_Whitney = ones(2)

    # The aspect ratio of the plate in the 1-direction
    r = a / b

    # An expression for the number of half-waves in the 1 and 2-direction (m and n respectively) which minizes the uniaxial, in-plane, normal,
    # buckling load (N₁ and N₂ respectively) can be derived by equating the derivative of the buckling load for uniaxial, in-plane, compression
    # (equation 5.70 from [1] and expression 6.4 from [2]) with respect to the number of half-waves in one direction while observing that the
    # number of half-waves in the other direction is 1 (pages 122 and 123 from [2]). For the number of half-waves in the 1-direction this yields
    #                                                        _______
    # δN₁            2π²(D₁₁m⁴ -D₂₂r⁴)                    4 /  D₂₂
    # --- (n = 1) = ------------------- = 0 --> m =   r    /  -----   &  m ≠ 0
    # δm                   a²m³                          \/    D₁₁
    #
    # whereas this results in
    #                                                        _______
    # δN₂            2π²(D₂₂n⁴r⁴ - D₁₁)               1   4 /  D₁₁
    # --- (m = 1) = -------------------- = 0 --> n = ---   /  -----   &  n ≠ 0
    # δn                   a²n³r²                     r  \/    D₂₂
    #
    # for the 2-direction. This number of half-waves, a float, is designated by κ (kappa). It can be noticed that these equations are each
    # others inverse.

    # The number of half-waves which minimizes the uniaxial, in-plane, normal, buckling load in the 1-direction (N₁)
    kappa = r * (D[1, 1] / D[0, 0] )**(1 / 4)

    # Regulate the theoretical number of half-waves in the 1-direction which minizes the uniaxial, in-plane, normal, buckling load since the
    # number of half-waves can only be a positive integer (m ∈ ℕ₁) (equation 5.68 from [1])
    m = masked_equal(array( [floor(kappa), ceil(kappa) ] ), 0)

    # The uniaxial, in-plane, normal, buckling load in the 1-direction corresponding to the number of half-waves minimizing this force
    # (equation 5.74 from [1] and expression 6.7 from [2])
    n_uniaxial = - pi**2 / (m**2 * a**2) * (D[0, 0] * m**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * m**2 * r**2 + D[1, 1] * r**4)

    # The uniaxial, in-plane, normal, buckling load in the 1-direction (N₁)
    N_Whitney[0] = n_uniaxial.max()

    # The number of half-waves in the 1-direction corresponding to the uniaxial, in-plane, normal, buckling load in the 1-direction
    m_Whitney[0] = m[argmax(n_uniaxial) ]

    # As previously derived, the theoretical number of half-waves in the 2-direction which minizes the uniaxial, in-plane, normal buckling load
    # has to be regulated since the number of half-waves can only be a positive integer (n ∈ ℕ₁) (equation 5.68 from [1])
    n = masked_equal(array( [floor(1 / kappa), ceil(1 / kappa) ] ), 0)

    # The buckling load of the laminate in the 2-direction corresponding to the number of half-waves minimizing this force (equation 5.70
    # from [1] and expression 6.4 from [2])
    n_uniaxial = - pi**2 / (n**2 * a**2 * r**2) * (D[0, 0] + 2 * (D[0, 1] + 2 * D[2, 2] ) * n**2 * r**2 + D[1, 1] * n**4 * r**4)

    # The uniaxial, in-plane, normal, buckling load in the 2-direction (N₂)
    N_Whitney[1] = n_uniaxial.max()

    # The number of half-waves in the 2-direction corresponding to the uniaxial, in-plane, normal, buckling load in the 2-direction
    n_Whitney[1] = n[argmax(n_uniaxial) ]

    # If a text file containing the uniaxial, in-plane, normal, buckling load in the 1 and 2-direction, and the corresponding number of
    # half-waves in each direction should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/stability_analysis/Whitney_method.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'The uniaxial, in-plane, normal, buckling load of a simply supported, rectangular, specially orthotropic '
                                f'laminate according to Whitney\'s method') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: stability_analysis/Whitney_method.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the uniaxial, in-plane, normal, buckling load in the 1 and 2-direction, and the corresponding number of half-waves in each
            # direction
            textfile.write(tabulate(
                            array( [ [f'N\u1D6A [N/m]', f'N\u1D67 [N/m]'],        N_Whitney, m_Whitney, n_Whitney] ).T.tolist(),
                        headers  = [                     f'Applied force', f'Buckling load',  f'm [-]',  f'n [-]'],
                        colalign = (                               'left',         'center',  'center',  'center'),
                        floatfmt = (                                '.5e',            '.5e',     '.0f',     '.0f') ) )

    # If an illustration showing ... should be generated
    if illustrations:
        # Use the Tableau 30 colors (consisting of Tableau 10, Tableau 10 Medium, and Tableau 10 Light from [3]) for the figure
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

        # An array of shape-(number_of_data_points) containing the variation of the biaxial loading ratio (k) (equation 5.72 from [1] and
        # expression 6.5 from [2])
        k = linspace(- 5, 5, num = number_of_data_points)

        # Two arrays of shape-(9, number_of_data_points) containing the combinations of half-waves in the 1 and 2-direction respectively which
        # minimize the biaxial, in-plane, normal, buckling load
        m, n = number_of_half_waves(number_of_data_points, k, r, D)

        # An array of shape-(9, number_of_data_points) containing the corresponding the in-plane, normal, buckling load in the 1-direction (N₁)
        # (equation 5.72 from [1] and expression 6.5 from [2])
        N_biaxial = - pi**2 / (a**2 * (m**2 + k * n**2 * r**2) ) \
                * (D[0, 0] * m**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * m**2 * n**2 * r**2 + D[1, 1] * n**4 * r**4)

        # An array of shape-(2, number_of_data_points) containing the negative and positive, in-plane, normal, buckling load in the 1-direction
        # (N₁) respectively as a function of the biaxial loading ratio (k)
        N_1 = vstack( (hstack( (amax(masked_greater_equal(N_biaxial[:, k < 0], 0), axis = 0), amax(N_biaxial[:, k >= 0], axis = 0) ) ),
                       hstack( (amin(masked_less_equal(   N_biaxial[:, k < 0], 0), axis = 0), amax(N_biaxial[:, k >= 0], axis = 0) ) ) ) )

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        # Add the variation of the negative and positive, in-plane, normal, buckling load in the 1-direction (N₁) as a function of the biaxial
        # loading ratio (k)
        plt.plot(k, N_1[0, :], color = Colors[0], linewidth = 1)

        # Determine the bottom limit of the y-axis
        bottom, __ = plt.ylim()

        # Set the bottom and top limit of the y-axis to the previously determined ±bottom limit
        plt.ylim(bottom = bottom, top = - bottom)

        # Fill the area between the negative and positive, in-plane, normal, buckling load in the 1-direction (N₁) for non-negative real numbers
        # of the biaxial loading ratio (k)
        plt.gca().fill_between(k[k <= 0], N_1[0, k <= 0], N_1[1, k <= 0], color = Colors[0] )

        # Fill the area between the negative, in-plane, normal, buckling load in the 1-direction (N₁) and infinity for non-positive real numbers
        # of the biaxial loading ratio (k)
        plt.gca().fill_between(k[k >= 0], N_1[0, k >= 0], - bottom * ones(number_of_data_points // 2), color = Colors[0] )

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Add a vertical line through the origin
        plt.axvline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'k [-]')

        # Set the limits of the x-axis
        plt.xlim(min(k), max(k) )

        # Format the ylabel
        plt.ylabel(f'N\u2081 = N\u2082 / k [N/m]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Whitney_method/k_vs_N1.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()

        # The plane defined by the two in-plane, normal loads under consideration (N₁ and N₂), which contains the interaction curve dictating
        # buckling, is depicted in figure 1 below. In addition, the angle θ between both loads is specified.
        #
        #                         N₁
        #                      θ = 0, 2π
        #                         ^
        #                         |
        #                     \ θ |
        #                      \<-|
        #                       \ |
        #                        \|
        #   θ = π / 2 ------------+------------> θ = 3π / 2, N₂
        #                         |
        #                         |
        #                         |
        #                         |
        #                         |
        #                       θ = π
        #
        # Figure WM.1: Illustration of the N₁N₂-plane and the angle
        #                θ between the two loads.
        #
        # From this figure can be derived that the angle θ is related to both loads via
        #
        #            N₁
        # tan θ = - ---- .
        #            N₂
        #
        # The biaxial loading ratio (k) is defined as (equation 5.71 from [1] and page 121 from [2])
        #
        #      Nᵧ
        # k = ---- .
        #      Nᵪ
        #
        # This ratio as a function of the angle θ is thus given by
        #
        #          1
        # k = - ------- .
        #        tan θ

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the N₁N₂-plane in which the
        # interaction curve is bounded (90° <= θ <= 180°)
        theta = pi / 180 * linspace(90, 180, num = number_of_data_points)

        # An array of shape-(number_of_data_points) containing the biaxial loading ratio (k) taking into account the singularity at
        # θ = 90° = π radians
        k = (theta != pi) * - 1 / (tan(theta) + (theta != pi) * finfo(float).eps) \
          + (theta == pi) * finfo(float).eps

        # Two arrays of shape-(9, number_of_data_points) containing the combinations of half-waves in the 1 and 2-direction respectively which
        # minimize the biaxial, in-plane, normal, buckling load
        m, n = number_of_half_waves(number_of_data_points, k, r, D)

        # An array of shape-(9, number_of_data_points) containing the corresponding the in-plane, normal, buckling load (N₀) (equation 5.72
        # from [1] and expression 6.5 from [2])
        N_0 = pi**2 / (a**2 * (m**2 + k * n**2 * r**2) ) \
            * (D[0, 0] * m**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * m**2 * n**2 * r**2 + D[1, 1] * n**4 * r**4)

        # For the section of the N₁N₂-plane under consideration (90° <= θ <= 180°), it holds that Nᵪ = - N₀ < 0 (equation 5.71 from [1] and page
        # 121 from [2]) and thus N₀ > 0. An array of shape-(number_of_data_points) containing the buckling load in the 1-direction (N₁) for each
        # of the biaxial loading ratios under consideration
        N_1 = - amin(masked_less(N_0, 0), axis = 0)

        # An array of shape-(number_of_data_points) containing the buckling load in the 2-direction (N₂) for each of the biaxial loading ratios
        # under consideration (equation 5.71 from [1] and page 121 from [2])
        N_2 = k * N_1

        # Correct the values of the buckling load at the boundary of the section under consideration (θ = 180°) due to the presence of
        # singularities
        N_1[theta == pi] = 0
        N_2[theta == pi] = N_Whitney[1]

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        # Add the variation of the in-plane, normal, buckling load in the 1 and 2-direction (N₁ and N₂ respectively)
        plt.plot(N_1, N_2, color = Colors[0], linewidth = 1)

        # Format the xlabel
        plt.xlabel(f'N\u2081 [N/m]')

        # Set the upper limit of the x-axis
        plt.xlim(right = 0)

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'N\u2082 [N/m]')

        # Set the upper limit of the y-axis
        plt.ylim(top = 0)

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Whitney_method/N1_vs_N2.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()

    # End the function and return the arrays of shape-(2) containing the uniaxial, in-plane, normal, buckling load in the 1 and 2-direction and
    # the corresponding number of half-waves in each direction for a simply supported, rectangular, specially orthotropic laminate according to
    # Whitney's method
    return(N_Whitney, m_Whitney, n_Whitney)
