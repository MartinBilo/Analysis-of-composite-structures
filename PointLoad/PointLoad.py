def PointLoad(assumeddeflection, force, geometry, laminate, mesh, settings, stiffness, data=False, illustrations=False):
    """[summary]

    Args:
        assumeddeflection (dictionary): [description]
        force (dictionary): [description]
        geometry (dictionary): [description]
        laminate (dictionary): [description]
        mesh (dictionary): [description]
        settings (dictionary): [description]
        stiffness (dictionary): [description]
        data (bool, optional): [description]. Defaults to False.
        illustrations (bool, optional): [description]. Defaults to False.
    """

    # Import packages into library
    from datetime import datetime
    from matplotlib import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.ticker import MaxNLocator
    from math              import inf, pi
    from numpy             import real, allclose, arange, array, copy, cos, cosh, einsum, linspace, reshape, sin, sinh, sqrt, tile, unique, zeros
    from numpy.linalg      import inv, solve
    from os                import remove
    from os.path           import exists
    from statistics        import median
    from tabulate          import tabulate
    from textwrap          import fill

    ##
    #
    xm       = assumeddeflection['xm']
    yn       = assumeddeflection['yn']
    XkXm     = assumeddeflection['XkXm']
    YlYn     = assumeddeflection['YlYn']
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
    XkdXm    = assumeddeflection['XkdXm']
    YldYn    = assumeddeflection['YldYn']
    ddXkdXm  = assumeddeflection['ddXkdXm']
    ddYldYn  = assumeddeflection['ddYldYn']

    ##
    #
    xi = force['xi']
    #
    eta = force['eta']
    #
    F_z = force['F_z']

    ## Import geometry properties
    #
    a = geometry['a']
    #
    b = geometry['b']

    ##
    #
    theta = laminate['theta']

    ##
    #
    nodes_x = mesh['nodes_x']
    #
    nodes_y = mesh['nodes_y']

    ##
    #
    fileformat           = settings['fileformat']
    number_of_elements_x = settings['number_of_elements_x']
    number_of_elements_y = settings['number_of_elements_y']
    m                    = settings['m']
    margin               = settings['margin']
    mn_max               = settings['mn_max']
    n                    = settings['n']
    simulation           = settings['simulation']

    ## Import stiffness properties
    #
    A = stiffness['A']
    #
    B = stiffness['B']
    #
    D = stiffness['D']

    ##
    #
    BoundaryConditions_Ritz   = [f'Simply supported',
                                 f'Simply supported (x = 0,a; y = 0); clamped (y = b)',
                                 f'Simply supported (y = 0,b; x = 0); clamped (x = a)',
                                 f'Simply supported (x = 0,a); clamped (y = 0,b)',
                                 f'Simply supported (x = 0,a); clamped (y = 0,b) [Polynomial]',
                                 f'Simply supported (y = 0,b); clamped (x = 0,a)',
                                 f'Simply supported (y = 0,b); clamped (x = 0,a) [Polynomial]',
                                 f'Simply supported (x = a; y = b); clamped (x = 0; y = 0)',
                                 f'Simply supported (x = a); clamped (x = 0); clamped (y = 0,b)',
                                 f'Simply supported (x = a); clamped (x = 0); clamped (y = 0,b) [Polynomial]',
                                 f'Simply supported (y = b); clamped (y = 0); clamped (x = 0,a)',
                                 f'Simply supported (y = b); clamped (y = 0); clamped (x = 0,a) [Polynomial]',
                                 f'Clamped',
                                 f'Clamped (x = 0,a); clamped (y = 0,b) [Polynomial]',
                                 f'Clamped (x = 0,a) [Polynomial]; clamped (y = 0,b)',
                                 f'Clamped [Polynomial]',
                                 f'Simply supported (x = 0,a); free (y = 0,b)',
                                 f'Simply supported (y = 0,b); free (x = 0,a)',
                                 f'Simply supported (x = a); clamped (x = 0); free (y = 0,b)',
                                 f'Simply supported (y = b); clamped (y = 0); free (x = 0,a)',
                                 f'Clamped (x = 0,a); free (y = 0,b)',
                                 f'Clamped (x = 0,a) [Polynomial]; free (y = 0,b)',
                                 f'Clamped (y = 0,b); free (x = 0,a)',
                                 f'Clamped (y = 0,b) [Polynomial]; free (x = 0,a)' ]

    #
    Situations_Ritz = len(BoundaryConditions_Ritz)

    #
    BC_Ritz = array( [ [1, 1],
                       [1, 2],
                       [2, 1],
                       [1, 3],
                       [1, 4],
                       [3, 1],
                       [4, 1],
                       [2, 2],
                       [2, 3],
                       [2, 4],
                       [3, 2],
                       [4, 2],
                       [3, 3],
                       [3, 4],
                       [4, 3],
                       [4, 4],
                       [1, 0],
                       [0, 1],
                       [2, 0],
                       [0, 2],
                       [3, 0],
                       [4, 0],
                       [0, 3],
                       [0, 4] ] )

    #
    error_Ritz      = zeros( [mn_max, Situations_Ritz] )
    error_Ritz[0,:] = inf

    #
    delta_Ritz = zeros( [mn_max, Situations_Ritz] )

    #
    mn_Ritz = zeros(Situations_Ritz, dtype = int)

    #
    x_Ritz = zeros(Situations_Ritz)
    y_Ritz = zeros(Situations_Ritz)

    ## Reduced bending stiffness approximation
    # Equation 2.64 from [1]
    A_red = inv(A)
    B_red = - A_red @ B
    D_red = D + B @ B_red

    #
    M = n[0:mn_max, 0:mn_max].T

    #
    Xm = zeros( [len(m), 5] )

    lambda_m = (M.flatten() == 1) * 4.712 \
             + (M.flatten() == 2) * 7.854 \
             + (M.flatten() > 2)  * (2 * M.flatten() + 1) * pi / 2
    gamma_m = (cosh(lambda_m) - cos(lambda_m) ) / (sin(lambda_m) + sinh(lambda_m) )
    Xm[M.flatten() == 1, 0] = 1
    Xm[M.flatten() == 2, 0] = real(sqrt(3) ) * (1 - 2 * xi / a)
    Xm[M.flatten() > 2, 0]  =                            cosh(lambda_m[M.flatten() > 2] / a * xi) \
                            +                             cos(lambda_m[M.flatten() > 2] / a * xi) \
                            - gamma_m[M.flatten() > 2] * sinh(lambda_m[M.flatten() > 2] / a * xi) \
                            - gamma_m[M.flatten() > 2] *  sin(lambda_m[M.flatten() > 2] / a * xi)

    Xm[:, 1] = sin(M.flatten() * pi * xi / a)

    lambda_m = (M.flatten() + 0.25) * pi
    gamma_m  = (sin(lambda_m) - sinh(lambda_m) ) / (cosh(lambda_m) - cos(lambda_m) )
    Xm[:, 2] = gamma_m *  cos(lambda_m / a * xi) \
             - gamma_m * cosh(lambda_m / a * xi) \
             +            sin(lambda_m / a * xi) \
             -           sinh(lambda_m / a * xi)

    lambda_m = (M.flatten() == 1) * 4.712 \
             + (M.flatten() == 2) * 7.854 \
             + (M.flatten() > 2)  * (2 * M.flatten() + 1) * pi / 2
    gamma_m = (cos(lambda_m) - cosh(lambda_m) ) / (sin(lambda_m) + sinh(lambda_m) )
    Xm[:, 3] = gamma_m *  cos(lambda_m / a * xi) \
             - gamma_m * cosh(lambda_m / a * xi) \
             +            sin(lambda_m / a * xi) \
             -           sinh(lambda_m / a * xi)

    Xm[:, 4] = (xi**2 - a * xi)**2 * xi**(M.flatten() - 1)

    #
    Yn = zeros( [len(M.flatten() ), 5] )

    lambda_n = (M.T.flatten() == 1) * 4.712 \
             + (M.T.flatten() == 2) * 7.854 \
             + (M.T.flatten() > 2)  * (2 * M.T.flatten() + 1) * pi / 2
    gamma_n = (cosh(lambda_n) - cos(lambda_n) ) / (sin(lambda_n) + sinh(lambda_n) )
    Yn[M.T.flatten() == 1, 0] = 1
    Yn[M.T.flatten() == 2, 0] = real(sqrt(3) ) * (1 - 2 * eta / b)
    Yn[M.T.flatten() > 2, 0]  =                           cosh(lambda_n[M.T.flatten() > 2] / b * eta) \
                           +                               cos(lambda_n[M.T.flatten() > 2] / b * eta) \
                           - gamma_n[M.T.flatten() > 2] * sinh(lambda_n[M.T.flatten() > 2] / b * eta) \
                           - gamma_n[M.T.flatten() > 2] *  sin(lambda_n[M.T.flatten() > 2] / b * eta)

    Yn[:, 1] = sin(M.T.flatten() * pi * eta / b)

    lambda_n = (M.T.flatten() + 0.25) * pi
    gamma_n  = (sin(lambda_n) - sinh(lambda_n) ) / (cosh(lambda_n) - cos(lambda_n) )
    Yn[:, 2] = gamma_n *  cos(lambda_n / b * eta) \
             - gamma_n * cosh(lambda_n / b * eta) \
             +            sin(lambda_n / b * eta) \
             -           sinh(lambda_n / b * eta)

    lambda_n = (M.T.flatten() == 1) * 4.712 \
             + (M.T.flatten() == 2) * 7.854 \
             + (M.T.flatten() > 2)  * (2 * M.T.flatten() + 1) * pi / 2
    gamma_n = (cos(lambda_n) - cosh(lambda_n) ) / (sin(lambda_n) + sinh(lambda_n) )
    Yn[:, 3] = gamma_n *  cos(lambda_n / b * eta) \
             - gamma_n * cosh(lambda_n / b * eta) \
             +            sin(lambda_n / b * eta) \
             -           sinh(lambda_n / b * eta)

    Yn[:, 4] = (eta**2 - b * eta)**2 * eta**(M.T.flatten() - 1)

    ## Ritz method
    #
    for o in range(Situations_Ritz):

        #
        mn = 1

        #
        counter = 0

        #
        while error_Ritz[counter, o] >= margin and mn <= mn_max:

            #
            counter = (mn != 1) * (counter + 1)

            #
            index = (tile(linspace(0, mn - 1, mn), (mn, 1) ) + tile(linspace(0, mn_max * (mn - 1), mn), (mn, 1) ).T).flatten().astype(int)

            # Equation 5.44 from [1]
            B_mn =    D_red[0, 0] * (ddXkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]   *   YlYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ]   ) \
                +     D_red[0, 1] * (  XkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ].T *   YlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] \
                +                      XkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]   *   YlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ].T ) \
                +     D_red[1, 1] * (  XkXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]     * ddYlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] ) \
                + 4 * D_red[2, 2] * ( dXkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]    *  dYldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ]  ) \
                + 2 * D_red[0, 2] * (ddXkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]    *   YldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] \
                +                    ddXkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ].T  *   YldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ].T ) \
                + 2 * D_red[1, 2] * (  XkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ].T  * ddYldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ].T \
                +                      XkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]    * ddYldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] )

            # Equation 5.44 from [1]
            C_mn = Xm[index, BC_Ritz[o, 0] ] * Yn[index, BC_Ritz[o, 1] ]

            # Equation 5.44 from [1]
            A_mn = solve(B_mn, C_mn)
            if max( (B_mn @ A_mn - C_mn).flatten(), key=abs) > 1e-18:
                counter -= 1
                break

            # Equation 5.42 and 5.44 from [1]
            w = F_z * einsum('k,ijk->ij', A_mn, xm[:, :, index, BC_Ritz[o, 0] ] * yn[:, :, index, BC_Ritz[o, 1] ] )

            # Maximum deflection
            delta_Ritz[counter, o] = max(w.flatten(), key=abs)

            # Normalized, maximum norm []
            if counter == 0:
                error_Ritz[counter, o] = 100
            else:
                error_Ritz[counter, o] = sqrt( (w - W).flatten() @ (w - W).flatten() / (W.flatten() @ W.flatten() ) ) * 100

            #
            W = w.copy()

            # Because of symmetry considerations and based on the Fourier decomposition of the pressure load even numbers will
            # not contribute to the deformation
            mn += 2

        #
        mn_Ritz[o] = mn - 2

        #
        x_Ritz[o] = median(nodes_x[w == delta_Ritz[counter, o] ] )
        y_Ritz[o] = median(nodes_y[w == delta_Ritz[counter, o] ] )

    # Tableau 30 (Tableau 10, Tableau 10 Medium & Tableau 10 Light)
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

    Colors_Ritz = [0, 3, 4, 6, 6, 7, 7, 9, 12, 12, 13, 13, 15, 15, 15, 15, 18, 19, 21, 22, 24, 24, 25, 25]

    LineStyles = ['solid', 'dashed', 'dotted', 'dashdot']

    LineStyles_Ritz = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 3, 1, 0, 0, 0, 0, 0, 1, 0, 1]

    # Initiate window
    plt.figure(0)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    for o in range(Situations_Ritz):
        plt.plot(linspace(1, mn_Ritz[o], (mn_Ritz[o] + 1) // 2), error_Ritz[0:(mn_Ritz[o] + 1) // 2, o], color=Colors[Colors_Ritz[o] ], linewidth=1,
                 linestyle=LineStyles[LineStyles_Ritz[o] ], label=BoundaryConditions_Ritz[o] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B5 [%]')
    plt.yscale('symlog', linthreshy = margin / 100)
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn, max(mn_Ritz) ) )
    plt.xticks(arange(1, max(mn, max(mn_Ritz) ) + 2, 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/ErrorConvergenceRitz_G.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(0)

    # Initiate window
    plt.figure(0)
    # Full screen
    plt.get_current_fig_manager().full_screen_toggle()
    # Create plots
    for o in range(Situations_Ritz):
        plt.plot(linspace(1, mn_Ritz[o], (mn_Ritz[o] + 1) // 2), delta_Ritz[0:(mn_Ritz[o] + 1) // 2, o], color=Colors[Colors_Ritz[o] ], linewidth=1,
                 linestyle=LineStyles[LineStyles_Ritz[o] ], label=BoundaryConditions_Ritz[o] )
    # Format ylabel and yticks
    plt.ylabel(f'\u03B4\u2098\u2090\u2093 [m]')
    plt.yscale('log')
    # Format xlabel and xticks
    plt.xlabel(f'M = N [-]')
    plt.xlim(1, max(mn, max(mn_Ritz) ) )
    plt.xticks(arange(1, max(mn, max(mn_Ritz) ) + 2, 2) )
    # Create legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
    # Create text wit simulation name and source
    plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0, -0.15, f'Source: PointLoad.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    # Save figure
    plt.savefig(f'{simulation}/Illustrations/PointLoad/DisplacementConvergenceRitz_G.{fileformat}', bbox_inches="tight")
    # Close figure
    plt.close(0)

    #
    if data:
        #
        if exists(f'{simulation}/Data/PointLoad/MaximumDeformation_G.txt'):
            remove(f'{simulation}/Data/PointLoad/MaximumDeformation_G.txt')

        #
        textfile = open(f'{simulation}/Data/PointLoad/MaximumDeformation_G.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Maximum deformation due to a constant pressure load according to different analyses for various boundary conditions') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: PointLoad.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(f'Ritz method \n')

        #
        textfile.write(tabulate( array([BoundaryConditions_Ritz, delta_Ritz[(mn_Ritz - 1) // 2, arange(delta_Ritz.shape[1]) ], x_Ritz, y_Ritz, mn_Ritz, error_Ritz[(mn_Ritz - 1) // 2, arange(error_Ritz.shape[1]) ] ] ).T.tolist(),
                    headers=(f'Boundary conditions', f'\u03B4\u2098\u2090\u2093 [m]', f'x [m]', f'y [m]', f'm = n [-]', f'\u03B5 [%]'),
                    stralign=('left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.2e', '.2e', '.0f', '.5e') ) )

        #
        textfile.close()

    # End the function
    return