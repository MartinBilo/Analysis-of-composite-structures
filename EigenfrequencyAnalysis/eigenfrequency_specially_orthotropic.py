def eigenfrequency_specially_orthotropic(assumeddeflection, material, settings, stiffness, data=False, illustrations=False):

    # Import modules into library
    from datetime     import datetime
    from matplotlib   import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as plt
    from numpy        import array, errstate, inf, linspace, sort, sqrt, tile, zeros
    from numpy.linalg import eig, solve

    #
    XkXm     = assumeddeflection['XkXm']
    YlYn     = assumeddeflection['YlYn']
    XkddXm   = assumeddeflection['XkddXm']
    YlddYn   = assumeddeflection['YlddYn']
    dXkdXm   = assumeddeflection['dXkdXm']
    dYldYn   = assumeddeflection['dYldYn']
    ddXkddXm = assumeddeflection['ddXkddXm']
    ddYlddYn = assumeddeflection['ddYlddYn']

    #
    rho = material['rho']

    #
    fileformat = settings['fileformat']
    mn_max     = settings['mn_max']
    margin     = settings['margin']
    simulation = settings['simulation']

    #
    D = stiffness['D']

    #
    BoundaryConditions = [f'Free',
                          f'Free (x = 0,a); simply supported (y = 0,b)',
                          f'Free (x = 0,a); clamped (y = 0); simply supported (y = b)',
                          f'Free (x = 0,a); clamped (y = 0,b)',
                          f'Free (x = 0,a); clamped (y = 0,b) [Polynomial]',
                          f'Simply supported (x = 0,a); free (y = 0,b)',
                          f'Simply supported',
                          f'Simply supported (x = 0,a); clamped (y = 0); simply supported (y = b)',
                          f'Simply supported (x = 0,a); clamped (y = 0,b)',
                          f'Simply supported (x = 0,a); clamped (y = 0,b) [Polynomial]',
                          f'Clamped (x = 0); simply supported (x = a); free (y = 0,b)',
                          f'Clamped (x = 0); simply supported (x = a; y = 0,b)',
                          f'Clamped (x = 0; y = 0); simply supported (x = a; y = b)',
                          f'Clamped (x = 0); simply supported (x = a); clamped (y = 0,b)',
                          f'Clamped (x = 0); simply supported (x = a); clamped (y = 0,b) [Polynomial]',
                          f'Clamped (x = 0,a); free (y = 0,b)',
                          f'Clamped (x = 0,a); simply supported (y = 0,b)',
                          f'Clamped (x = 0,a); clamped (y = 0); simply supported (y = b)',
                          f'Clamped',
                          f'Clamped (x = 0,a); clamped (y = 0,b) [Polynomial]',
                          f'Clamped (x = 0,a) [Polynomial]; free (y = 0,b)',
                          f'Clamped (x = 0,a) [Polynomial]; simply supported (y = 0,b)',
                          f'Clamped (x = 0,a) [Polynomial]; clamped (y = 0); simply supported (y = b)',
                          f'Clamped (x = 0,a) [Polynomial]; clamped (y = 0,b)',
                          f'Clamped [Polynomial]']

    #
    BC = array( [ [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                  [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                  [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
                  [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
                  [4, 0], [4, 1], [4, 2], [4, 3], [4, 4] ] )

    #
    situations = len(BC)

    #
    omega = zeros( (mn_max**2, situations) )

    #
    error = zeros( (mn_max**2, situations) )

    #
    error[0, :] = 100

    #
    for i in range(situations):

        #
        o =     D[0, 0] *  ddXkddXm[:, :, BC[i, 0] ]   *     YlYn[:, :, BC[i, 1] ]    \
          +     D[0, 1] * (  XkddXm[:, :, BC[i, 0] ].T *   YlddYn[:, :, BC[i, 1] ]    \
          +                  XkddXm[:, :, BC[i, 0] ]   *   YlddYn[:, :, BC[i, 1] ].T) \
          +     D[1, 1] *      XkXm[:, :, BC[i, 0] ]   * ddYlddYn[:, :, BC[i, 1] ]    \
          + 4 * D[2, 2] *    dXkdXm[:, :, BC[i, 0] ]   *   dYldYn[:, :, BC[i, 1] ]

        #
        p = rho * XkXm[:, :, BC[i, 0] ] * YlYn[:, :, BC[i, 1] ]

        #
        mn = 0

        #
        omega_1 = 0

        #
        while error[mn, i] >= margin and mn < mn_max:

            #
            mn += 1

            #
            index = (tile(linspace(0, mn - 1, mn), (mn, 1) ) + tile(linspace(0, mn_max * (mn - 1), mn), (mn, 1) ).T).flatten().astype(int)

            #
            omega_squared, __ = eig(solve(p[index.reshape(-1,1), index], o[index.reshape(-1,1), index]) )

            # Filter smaller than zero and imaginary and sort
            omega_filtered = sort(sqrt(omega_squared.real[omega_squared > 0] ) )

            #
            omega_2 = omega_1

            #
            omega_1 = omega[0, i]

            #
            omega[:len(omega_filtered), i] = omega_filtered

            #
            if mn == 1 or mn == 2:
                error[mn, i] = 100
            else:
                # Since free free yields a zero eigenfrequency for m = n = 1
                with errstate(divide='ignore'):
                    # To avoid a selection based on symmetric and asymmetric vibration modes
                    error[mn, i] = max(abs(omega[0, i] - omega_1) / omega_1 * 100,
                                       abs(omega[0, i] - omega_2) / omega_2 * 100)

    #
    error[error == inf] = 100

    #
    if illustrations:
        # Tableau 30 (Tableau 10, Tableau 10 Medium & Tableau 10 Light)
        Colors = [( 31 / 255., 119 / 255., 180 / 255.), (114 / 255., 158 / 255., 206 / 255.), (174 / 255., 199 / 255., 232 / 255.),
                  (255 / 255., 127 / 255.,  14 / 255.), (255 / 255., 158 / 255.,  74 / 255.), (255 / 255., 187 / 255., 120 / 255.),
                  ( 44 / 255., 160 / 255.,  44 / 255.), (103 / 255., 191 / 255.,  92 / 255.), (152 / 255., 223 / 255., 138 / 255.),
                  (214 / 255.,  39 / 255.,  40 / 255.), (237 / 255., 102 / 255.,  93 / 255.), (255 / 255., 152 / 255., 150 / 255.),
                  (148 / 255., 103 / 255., 189 / 255.), (173 / 255., 139 / 255., 201 / 255.), (197 / 255., 176 / 255., 213 / 255.),
                  (140 / 255.,  86 / 255.,  75 / 255.), (168 / 255., 120 / 255., 110 / 255.), (196 / 255., 156 / 255., 148 / 255.),
                  (227 / 255., 119 / 255., 194 / 255.), (237 / 255., 151 / 255., 202 / 255.), (247 / 255., 182 / 255., 210 / 255.),
                  (127 / 255., 127 / 255., 127 / 255.), (162 / 255., 162 / 255., 162 / 255.), (199 / 255., 199 / 255., 199 / 255.),
                  (188 / 255., 189 / 255.,  34 / 255.), (205 / 255., 204 / 255.,  93 / 255.), (219 / 255., 219 / 255., 141 / 255.),
                  ( 23 / 255., 190 / 255., 207 / 255.), (109 / 255., 204 / 255., 218 / 255.), (158 / 255., 218 / 255., 229 / 255.) ]

        # Initiate window
        plt.figure()
        # Full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()
        # Create plots
        for i in range(situations):
            tmp = error[2:, i]
            plt.plot(range(2,len(tmp[tmp > 0]) + 2), tmp[tmp > 0], color=Colors[i], linewidth = 1, label = BoundaryConditions[i] )
        # Format xlabel and xticks
        plt.xlabel(f'm = n [-]')
        # Format ylabel and yticks
        plt.ylabel(f'\u03F5 [%]')
        plt.yscale('symlog', linthreshy=margin / 10)
        # Create legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
        # Get min and max of axes
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        # Create text wit simulation name and source
        plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(0, -0.15, f'Source: eigenfrequency_specially_orthotropic.py [v1.0] via composite_analysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        # Save figure
        plt.savefig(f'{simulation}/Illustrations/EigenfrequencyAnalysis/error_specially_orthotropic.{fileformat}', bbox_inches="tight")
        # Close figure
        plt.close()

    return