def MaximumStrainFailureTheoryEnvelope(laminate, material, settings, tad, data=False, illustrations=False):

    # Import packages into library
    from   datetime          import datetime
    from   matplotlib        import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as     plt
    from   numpy             import array, concatenate, deg2rad, einsum, errstate, flip, linspace, ones, pi, repeat, sin, sort, sqrt, tan, \
                                    tile, where, zeros, copy, inf, isinf
    from   os                import mkdir, remove
    from   os.path           import exists, isdir
    from   tabulate          import tabulate
    from   textwrap          import fill

    ## Import the laminate properties
    # The layup of the laminate in degrees
    theta = laminate['theta']

    ##
    #
    epsilon_u_x_c = abs(material['epsilon_u_x_c'] )
    #
    epsilon_u_x_t = material['epsilon_u_x_t']
    #
    epsilon_u_y_c = abs(material['epsilon_u_y_c'] )
    #
    epsilon_u_y_t = material['epsilon_u_y_t']
    #
    gamma_u_xy    = material['gamma_u_xy']

    ##
    #
    data_points = settings['data_points']
    #
    fileformat  = settings['fileformat']
    #
    simulation  = settings['simulation']

    ##
    def SingleFailureLoad(d, tad, epsilon_u_x_c, epsilon_u_x_t, epsilon_u_y_c, epsilon_u_y_t, gamma_u_xy):

        #
        N_x = zeros(2)
        # N_x_neg
        N_x[0] = min( (tad[0, d, :] < 0) * (epsilon_u_x_t / tad[0, d, :] ) - (tad[0, d, :] > 0) * (epsilon_u_x_c / tad[0, d, :] ) + (tad[0, d, :] == 0) * 0, key = abs)
        # N_x_pos
        N_x[1] = min( (tad[0, d, :] > 0) * (epsilon_u_x_t / tad[0, d, :] ) - (tad[0, d, :] < 0) * (epsilon_u_x_c / tad[0, d, :] ) + (tad[0, d, :] == 0) * 0, key = abs)

        #
        N_y = zeros(2)
        # N_y_neg
        N_y[0] = min( (tad[1, d, :] < 0) * (epsilon_u_y_t / tad[1, d, :] ) - (tad[1, d, :] > 0) * (epsilon_u_y_c / tad[1, d, :] ) + (tad[1, d, :] == 0) * 0, key = abs)
        # N_y_pos
        N_y[1] = min( (tad[1, d, :] > 0) * (epsilon_u_y_t / tad[1, d, :] ) - (tad[1, d, :] < 0) * (epsilon_u_y_c / tad[1, d, :] ) + (tad[1, d, :] == 0) * 0, key = abs)

        #
        N_xy = zeros(2)
        # N_xy_neg
        N_xy[0] = min( (tad[2, d, :] < 0) * (gamma_u_xy   / tad[2, d, :] ) - (tad[2, d, :] > 0) * (gamma_u_xy   / tad[2, d, :] ) + (tad[2, d, :] == 0) * 0, key = abs)
        # N_xy_pos
        N_xy[1] = min( (tad[2, d, :] > 0) * (gamma_u_xy   / tad[2, d, :] ) - (tad[2, d, :] < 0) * (gamma_u_xy   / tad[2, d, :] ) + (tad[2, d, :] == 0) * 0, key = abs)

        #
        N = zeros(2)
        # N_neg
        N[0] = min(N_x[0], N_y[0], N_xy[0], key=abs)
        # N_pos
        N[1] = min(N_x[1], N_y[1], N_xy[1], key=abs)

        #
        return(N_x, N_y, N_xy, N)

    ##
    def CylindricalCoordinateSolver(data_points,d1, d2, s1, N, theta, ratio_force, tad, F_t, F_c):

        #
        coefficient_force_min = (einsum('i,j->ij', tad[s1, d1, :], ones(data_points) ) - einsum('i,j->ij', tad[s1, d2, :], ratio_force) )
        coefficient_force_max = (einsum('i,j->ij', tad[s1, d1, :], ones(data_points) ) + einsum('i,j->ij', tad[s1, d2, :], ratio_force) )

        #
        F_c = tile(- F_c, (data_points, 1) ).T
        F_t = tile(  F_t, (data_points, 1) ).T

        #
        n_1 = ( (    0      <= theta) & (theta <       pi / 2) ) * ( ( (coefficient_force_min < 0) * (F_c) + (coefficient_force_min > 0) * (F_t) ) / coefficient_force_min) \
            + ( (    pi / 2 <  theta) & (theta <       pi)     ) * ( ( (coefficient_force_max > 0) * (F_c) + (coefficient_force_max < 0) * (F_t) ) / coefficient_force_max) \
            + ( (    pi     <= theta) & (theta <   3 * pi / 2) ) * ( ( (coefficient_force_min > 0) * (F_c) + (coefficient_force_min < 0) * (F_t) ) / coefficient_force_min) \
            + ( (3 * pi / 2 <  theta) & (theta <=  2 * pi)     ) * ( ( (coefficient_force_max < 0) * (F_c) + (coefficient_force_max > 0) * (F_t) ) / coefficient_force_max)

        n_1_neg = copy(n_1)
        n_1_neg[n_1_neg > 0] = - inf
        n_1_neg = sort(n_1_neg, axis=0)

        n_1_pos = copy(n_1)
        n_1_pos[n_1_pos < 0] = inf
        n_1_pos = sort(n_1_pos, axis=0)

        #
        MN       = zeros( [data_points, 2] )
        MN[:, 0] = where(isinf(n_1_pos[ 0, :]), n_1_neg[- 1, :], n_1_pos[ 0, :])
        MN[:, 1] = ( (theta[0, :] != pi / 2) & (theta[0, :] != 3 * pi / 2) ) * (- MN[:, 0] * tan(theta[0, :] ) ) \
                 + (theta[0, :] == pi / 2) * (N[0] ) \
                 + (theta[0, :] == 3 * pi / 2) * (N[1] )
        MN       = concatenate( (MN, MN[0, :].reshape(1, 2) ), axis=0)

        #
        return(MN)

    ##
    def FailureEnvelope(data_points, N_x, N_y, N_xy, d1, d2, tad, epsilon_u_x_c, epsilon_u_x_t, epsilon_u_y_c, epsilon_u_y_t, gamma_u_xy):

        #
        theta = tile(deg2rad(linspace(0, 360 - 360 / data_points, num = data_points) ), (tad.shape[2], 1) )

        #
        with errstate(divide='ignore'):
            ratio_force = where( (1 - (sin(theta[0, :] ) )**2) != 0, sqrt( (sin(theta[0, :] ) )**2 / (1 - (sin(theta[0, :] ) )**2) ), 0)

        #
        MN_x  = CylindricalCoordinateSolver(data_points, d1, d2, 0, N_x, theta, ratio_force, tad, epsilon_u_x_t, epsilon_u_x_c)

        #
        MN_y  = CylindricalCoordinateSolver(data_points, d1, d2, 1, N_y, theta, ratio_force, tad, epsilon_u_y_t, epsilon_u_y_c)

        #
        MN_xy = CylindricalCoordinateSolver(data_points, d1, d2, 2, N_xy, theta, ratio_force, tad, gamma_u_xy, gamma_u_xy)

        #
        mn       = zeros( [data_points + 1, 3] )
        mn[:, 0] = sqrt(MN_x[:, 0]**2  + MN_x[:, 1]**2)
        mn[:, 1] = sqrt(MN_y[:, 0]**2  + MN_y[:, 1]**2)
        mn[:, 2] = sqrt(MN_xy[:, 0]**2 + MN_xy[:, 1]**2)

        MN       = zeros( [data_points + 1, 2] )
        MN[:, 0] = where( (mn[:, 0] <= mn[:, 1]) & (mn[:, 0] <= mn[:, 2] ), MN_x[:, 0], \
                   where( (mn[:, 1] <= mn[:, 0]) & (mn[:, 1] <= mn[:, 2] ), MN_y[:, 0], MN_xy[:, 0] ) )
        MN[:, 1] = where( (mn[:, 0] <= mn[:, 1]) & (mn[:, 0] <= mn[:, 2] ), MN_x[:, 1], \
                   where( (mn[:, 1] <= mn[:, 0]) & (mn[:, 1] <= mn[:, 2] ), MN_y[:, 1], MN_xy[:, 1] ) )

        #
        return(MN_x, MN_y, MN_xy, MN)

    # Number of plies in the laminate
    number_of_plies = len(theta)

    #
    epsilon_u_x_c = repeat(epsilon_u_x_c * ones(number_of_plies), 2)
    #
    epsilon_u_x_t = repeat(epsilon_u_x_t * ones(number_of_plies), 2)
    #
    epsilon_u_y_c = repeat(epsilon_u_y_c * ones(number_of_plies), 2)
    #
    epsilon_u_y_t = repeat(epsilon_u_y_t * ones(number_of_plies), 2)
    #
    gamma_u_xy = repeat(gamma_u_xy * ones(number_of_plies), 2)

    #
    forces = [f'Nx', f'Ny', f'Nxy', f'Mx', f'My', f'Mxy']

    #
    number_of_forces = len(forces)

    #
    MaximumStrain = {}

    #
    for i in range(0, number_of_forces):
        #
        locals()[f'{forces[i]}_x'], locals()[f'{forces[i]}_y'], locals()[f'{forces[i]}_xy'], locals()[forces[i] ] \
            = SingleFailureLoad(i, tad, epsilon_u_x_c, epsilon_u_x_t, epsilon_u_y_c, epsilon_u_y_t, gamma_u_xy)
        #
        MaximumStrain[forces[i] ] = locals()[forces[i] ]

    #
    for i in range(0, number_of_forces - 1):
        #
        for j in range(i + 1, number_of_forces):
            #
            locals()[f'{forces[i]}{forces[j]}_x'], locals()[f'{forces[i]}{forces[j]}_y'], locals()[f'{forces[i]}{forces[j]}_xy'], locals()[f'{forces[i]}{forces[j]}'] \
                = FailureEnvelope(data_points, locals()[f'{forces[j]}_x'], locals()[f'{forces[j]}_y'], locals()[f'{forces[j]}_xy'], i, j, \
                                  tad, epsilon_u_x_c, epsilon_u_x_t, epsilon_u_y_c, epsilon_u_y_t, gamma_u_xy)
            #
            MaximumStrain[forces[i] + forces[j] ] = locals()[f'{forces[i]}{forces[j]}']

    #
    if data:
                #
        if exists(f'{simulation}/Data/MaterialFailureEnvelopes/MaximumStrainFailureTheory.txt'):
            remove(f'{simulation}/Data/MaterialFailureEnvelopes/MaximumStrainFailureTheory.txt')

        #
        textfile = open(f'{simulation}/Data/MaterialFailureEnvelopes/MaximumStrainFailureTheory.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Singular failure loads according to the Maximum Strain Failure Theory') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: MaximumStrainFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(f'Singular failure loads \n')

        #
        textfile.write(tabulate( array( [ [f'N\u2093  [N/m]', locals()[f'{forces[0]}_x'][0],  locals()[f'{forces[0]}_x'][1], \
                                                             locals()[f'{forces[0]}_y'][0],  locals()[f'{forces[0]}_y'][1], \
                                                             locals()[f'{forces[0]}_xy'][0], locals()[f'{forces[0]}_xy'][1], \
                                                             locals()[forces[0] ][0],        locals()[forces[0] ][1] ], \
                                          [f'N\u1D67  [N/m]', locals()[f'{forces[1]}_x'][0],  locals()[f'{forces[1]}_x'][1], \
                                                             locals()[f'{forces[1]}_y'][0],  locals()[f'{forces[1]}_y'][1], \
                                                             locals()[f'{forces[1]}_xy'][0], locals()[f'{forces[1]}_xy'][1], \
                                                             locals()[forces[1] ][0],        locals()[forces[1] ][1] ], \
                                          [f'N\u2093\u1D67 [N/m]', locals()[f'{forces[2]}_x'][0],  locals()[f'{forces[2]}_x'][1], \
                                                                   locals()[f'{forces[2]}_y'][0],  locals()[f'{forces[2]}_y'][1], \
                                                                   locals()[f'{forces[2]}_xy'][0], locals()[f'{forces[2]}_xy'][1], \
                                                                   locals()[forces[2] ][0],        locals()[forces[2] ][1] ], \
                                          [f'M\u2093  [N]', locals()[f'{forces[3]}_x'][0],  locals()[f'{forces[3]}_x'][1], \
                                                           locals()[f'{forces[3]}_y'][0],  locals()[f'{forces[3]}_y'][1], \
                                                           locals()[f'{forces[3]}_xy'][0], locals()[f'{forces[3]}_xy'][1], \
                                                           locals()[forces[3] ][0],        locals()[forces[3] ][1] ], \
                                          [f'M\u1D67  [N]', locals()[f'{forces[4]}_x'][0],  locals()[f'{forces[4]}_x'][1], \
                                                           locals()[f'{forces[4]}_y'][0],  locals()[f'{forces[4]}_y'][1], \
                                                           locals()[f'{forces[4]}_xy'][0], locals()[f'{forces[4]}_xy'][1], \
                                                           locals()[forces[4] ][0],        locals()[forces[4] ][1] ], \
                                          [f'M\u2093\u1D67 [N]', locals()[f'{forces[5]}_x'][0],  locals()[f'{forces[5]}_x'][1], \
                                                                 locals()[f'{forces[5]}_y'][0],  locals()[f'{forces[5]}_y'][1], \
                                                                 locals()[f'{forces[5]}_xy'][0], locals()[f'{forces[5]}_xy'][1],
                                                                 locals()[forces[5] ][0],        locals()[forces[5] ][1] ] ] ).tolist(),
                    headers=(f'Force', f'\u03B5\u2081', f'', f'\u03B5\u2082', f'', f'\u03B3\u2081\u2082', f'', f'Maximum Strain', f'Failure Theory'),
                    colalign=('left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.5e', '.5e', '.5e', '.5e', '.5e', '.5e', '.5e') ) )

        #
        textfile.close()

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

        #
        LineStyles = ['solid', 'dashed', 'dotted', 'dashdot']

        # Labels
        Labels = [f'N\u2093 [N/m]', f'N\u1D67 [N/m]', f'N\u2093\u1D67 [N/m]', f'M\u2093 [N]', f'M\u1D67 [N]', f'M\u2093\u1D67 [N]']

        #
        if not isdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaximumStrainFailureTheory'):
            mkdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaximumStrainFailureTheory')

        #
        for i in range(0, number_of_forces - 1):
            #
            for j in range(i + 1, number_of_forces):
                #
                if exists(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaximumStrainFailureTheory/{forces[i]}{forces[j]}.{fileformat}'):
                    remove(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaximumStrainFailureTheory/{forces[i]}{forces[j]}.{fileformat}')

                # Initiate window
                plt.figure(0)
                # Full screen
                plt.get_current_fig_manager().full_screen_toggle()
                # Create vertical and horizontal line through origin
                plt.axvline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                plt.axhline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                # Create plots
                plt.plot(locals()[f'{forces[i]}{forces[j]}'][:, 1],    locals()[f'{forces[i]}{forces[j]}'][:, 0],    color=Colors[0],  linewidth=1, linestyle=LineStyles[0], label=f'Maximum Strain Failure Theory')
                plt.plot(locals()[f'{forces[i]}{forces[j]}_x'][:, 1],  locals()[f'{forces[i]}{forces[j]}_x'][:, 0],  color=Colors[21], linewidth=1, linestyle=LineStyles[2], label=f'\u03B5\u2081')
                plt.plot(locals()[f'{forces[i]}{forces[j]}_y'][:, 1],  locals()[f'{forces[i]}{forces[j]}_y'][:, 0],  color=Colors[22], linewidth=1, linestyle=LineStyles[2], label=f'\u03B5\u2082')
                plt.plot(locals()[f'{forces[i]}{forces[j]}_xy'][:, 1], locals()[f'{forces[i]}{forces[j]}_xy'][:, 0], color=Colors[23], linewidth=1, linestyle=LineStyles[2], label=f'\u03B3\u2081\u2082')
                #
                plt.plot(0, locals()[forces[i] ][0], color=Colors[0], marker='.')
                plt.plot(0, locals()[forces[i] ][1], color=Colors[0], marker='.')
                plt.plot(locals()[forces[j] ][0], 0, color=Colors[0], marker='.')
                plt.plot(locals()[forces[j] ][1], 0, color=Colors[0], marker='.')
                #
                plt.plot(0, locals()[f'{forces[i]}_x'][0], color=Colors[21], marker='1')
                plt.plot(0, locals()[f'{forces[i]}_x'][1], color=Colors[21], marker='1')
                plt.plot(locals()[f'{forces[j]}_x'][0], 0, color=Colors[21], marker='1')
                plt.plot(locals()[f'{forces[j]}_x'][1], 0, color=Colors[21], marker='1')
                #
                plt.plot(0, locals()[f'{forces[i]}_y'][0], color=Colors[22], marker='1')
                plt.plot(0, locals()[f'{forces[i]}_y'][1], color=Colors[22], marker='1')
                plt.plot(locals()[f'{forces[j]}_y'][0], 0, color=Colors[22], marker='1')
                plt.plot(locals()[f'{forces[j]}_y'][1], 0, color=Colors[22], marker='1')
                #
                plt.plot(0, locals()[f'{forces[i]}_xy'][0], color=Colors[23], marker='1')
                plt.plot(0, locals()[f'{forces[i]}_xy'][1], color=Colors[23], marker='1')
                plt.plot(locals()[f'{forces[j]}_xy'][0], 0, color=Colors[23], marker='1')
                plt.plot(locals()[f'{forces[j]}_xy'][1], 0, color=Colors[23], marker='1')
                # Format axes
                plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0) )
                # Format xlabel
                plt.xlabel(Labels[j] )
                # Format ylabel
                plt.ylabel(Labels[i] )
                # Create legend
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')
                # Create text wit simulation name and source
                plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0, -0.15, f'Source: MaximumStrainFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                # Save figure
                plt.savefig(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaximumStrainFailureTheory/{forces[i]}{forces[j]}.{fileformat}', bbox_inches='tight')
                # Close figure
                plt.close(0)

    #
    return(MaximumStrain)