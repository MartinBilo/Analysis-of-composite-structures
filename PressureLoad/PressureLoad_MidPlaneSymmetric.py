def PressureLoad_MidPlaneSymmetric(assumeddeflection, force, geometry, mesh, settings, stiffness, data=False, illustrations=False):

    ## Finicky methods with bounds on a, b and D and require special orthotopic laminate

    # Import packages into library
    from   datetime          import datetime
    from   matplotlib        import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as plt
    from   matplotlib.ticker import MaxNLocator
    from   math              import inf, pi
    from   numpy             import arange, array, copy, einsum, empty, linspace, reshape, sqrt, tile, zeros
    from   numpy.linalg      import solve
    from   os                import remove
    from   os.path           import exists
    from   statistics        import median
    from   sys               import float_info
    from   tabulate          import tabulate
    from   textwrap          import fill

    ##
    #
    from AssumedDeflection.SimplifiedLévyCoefficientsSimplySupportedSimplySupported import SimplifiedLévyCoefficientsSimplySupportedSimplySupported
    from AssumedDeflection.SimplifiedLévyCoefficientsClampedSimplySupported         import SimplifiedLévyCoefficientsClampedSimplySupported
    from AssumedDeflection.SimplifiedLévyCoefficientsFreeSimplySupported            import SimplifiedLévyCoefficientsFreeSimplySupported
    from AssumedDeflection.SimplifiedLévyCoefficientsClampedClamped                 import SimplifiedLévyCoefficientsClampedClamped
    from AssumedDeflection.SimplifiedLévyCoefficientsFreeFree                       import SimplifiedLévyCoefficientsFreeFree
    from AssumedDeflection.SimplifiedLévyCoefficientsClampedFree                    import SimplifiedLévyCoefficientsClampedFree

    #
    from AssumedDeflection.LévyCoefficientsSimplySupportedSimplySupported import LévyCoefficientsSimplySupportedSimplySupported
    from AssumedDeflection.LévyCoefficientsClampedSimplySupported         import LévyCoefficientsClampedSimplySupported
    from AssumedDeflection.LévyCoefficientsFreeSimplySupported            import LévyCoefficientsFreeSimplySupported
    from AssumedDeflection.LévyCoefficientsClampedClamped                 import LévyCoefficientsClampedClamped
    from AssumedDeflection.LévyCoefficientsFreeFree                       import LévyCoefficientsFreeFree
    from AssumedDeflection.LévyCoefficientsClampedFree                    import LévyCoefficientsClampedFree

    ##
    #
    xm       = assumeddeflection['xm']
    yn       = assumeddeflection['yn']
    Xm       = assumeddeflection['Xm']
    Yn       = assumeddeflection['Yn']
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
    p_z = force['p_z']

    ## Import geometry properties
    #
    a = geometry['a']
    #
    b = geometry['b']

    ##
    #
    nodes_x = mesh['nodes_x']
    #
    nodes_y = mesh['nodes_y']

    ##
    #
    fileformat = settings['fileformat']
    m          = settings['m']
    margin     = settings['margin']
    mn_max     = settings['mn_max']
    n          = settings['n']
    simulation = settings['simulation']

    ## Import stiffness properties
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
    error_Ritz    = zeros( [mn_max, Situations_Ritz] )
    error_Ritz[0] = inf

    #
    delta_Ritz = zeros( [mn_max, Situations_Ritz] )

    #
    mn_Ritz = zeros(Situations_Ritz, dtype = int)

    #
    x_Ritz = zeros(Situations_Ritz)
    y_Ritz = zeros(Situations_Ritz)

    ## Ritz method
    #
    W = empty(nodes_x.shape)

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
            B_mn =    D[0, 0] * (ddXkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]   *   YlYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ]   ) \
                +     D[0, 1] * (  XkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ].T *   YlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] \
                +                  XkddXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]   *   YlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ].T ) \
                +     D[1, 1] * (  XkXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]     * ddYlddYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] ) \
                + 4 * D[2, 2] * ( dXkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]    *  dYldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ]  ) \
                + 2 * D[0, 2] * (ddXkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]    *   YldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] \
                +                ddXkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ].T  *   YldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ].T ) \
                + 2 * D[1, 2] * (  XkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ].T  * ddYldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ].T \
                +                  XkdXm[index.reshape(-1,1), index, BC_Ritz[o, 0] ]    * ddYldYn[index.reshape(-1,1), index, BC_Ritz[o, 1] ] )

            # Equation 5.44 from [1]
            C_mn = Xm[index, BC_Ritz[o, 0] ] * Yn[index, BC_Ritz[o, 1] ]

            # Equation 5.44 from [1]
            A_mn = solve(B_mn, C_mn)
            if max( (B_mn @ A_mn - C_mn).flatten(), key=abs) > float_info.epsilon:
                counter -= 1
                break

            # Equation 5.42 and 5.44 from [1]
            w = p_z * einsum('k,ijk->ij', A_mn, xm[:, :, index, BC_Ritz[o, 0] ] * yn[:, :, index, BC_Ritz[o, 1] ] )

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

    if illustrations:
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

        #
        if exists(f'{simulation}/Illustrations/PressureLoad/ErrorConvergenceRitz_MidPlaneSymmetric.{fileformat}'):
            remove(f'{simulation}/Illustrations/PressureLoad/ErrorConvergenceRitz_MidPlaneSymmetric.{fileformat}')

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
        # Get min and max of y-axis
        y_b1, y_t1 = plt.ylim()
        # Format xlabel and xticks
        plt.xlabel(f'M = N [-]')
        plt.xlim(1, max(mn_Ritz) )
        plt.xticks(arange(1, max(mn_Ritz) + 2, 2) )
        # Create legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
        # Create text wit simulation name and source
        plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(0, -0.15, f'Source: PressureLoad_MidPlaneSymmetric.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        # Save figure
        plt.savefig(f'{simulation}/Illustrations/PressureLoad/ErrorConvergenceRitz_MidPlaneSymmetric.{fileformat}', bbox_inches="tight")
        # Close figure
        plt.close(0)

    #
    if data:
        #
        if exists(f'{simulation}/Data/PressureLoad/MaximumDeformation_MidPlaneSymmetric.txt'):
            remove(f'{simulation}/Data/PressureLoad/MaximumDeformation_MidPlaneSymmetric.txt')

        #
        textfile = open(f'{simulation}/Data/PressureLoad/MaximumDeformation_MidPlaneSymmetric.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Maximum deformation due to a constant pressure load according to different analyses for various boundary conditions') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: PressureLoad_MidPlaneSymmetric.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        if max(abs(B.flatten() ) ) != 0:
            textfile.write(fill(f'Warning: The laminate under consideration violates the assumed symmetry (B = 0).') )
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