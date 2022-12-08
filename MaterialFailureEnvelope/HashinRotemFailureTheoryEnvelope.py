def HashinRotemFailureTheoryEnvelope(laminate, material, settings, tqad, data=False, illustrations=False):


    # Import packages into library
    from   datetime          import datetime
    from   math              import pi
    from   matplotlib        import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as     plt
    from   numpy             import amax, argmin, array, concatenate, deg2rad, einsum, errstate, flip, linspace, maximum, ones, pi, repeat, \
                                    sin, sort, sqrt, tan, tile, where, zeros
    from   numpy.linalg      import norm
    from   os                import mkdir, remove
    from   os.path           import exists, isdir
    from   tabulate          import tabulate
    from   textwrap          import fill

    ## Import the laminate properties
    # The layup of the laminate in degrees
    theta = laminate['theta']

    ##
    #
    X_c = abs(material['X_c'] )
    #
    X_t = material['X_t']
    #
    Y_c = abs(material['Y_c'] )
    #
    Y_t = material['Y_t']
    #
    S   = material['S']

    ##
    #
    data_points = settings['data_points']
    #
    fileformat  = settings['fileformat']
    #
    simulation  = settings['simulation']

    ##
    def SingleFailureLoad(d, tq, X_c, X_t, Y_c, Y_t, S):

        #
        N_ff = zeros(2)
        # Compressive fiber failure
        N_ff[0] = min( (tq[0, d, :] < 0) * (X_t / tq[0, d, :] ) \
                     - (tq[0, d, :] > 0) * (X_c / tq[0, d, :] ) \
                     + (tq[0, d, :] == 0) * 0, key = abs)
        # Tensile fiber failure
        N_ff[1] = min( (tq[0, d, :] > 0) * (X_t / tq[0, d, :] ) \
                     - (tq[0, d, :] < 0) * (X_c / tq[0, d, :] ) \
                     + (tq[0, d, :] == 0) * 0, key = abs)

        #
        N_mf = zeros(2)
        # Compressive matrix failure
        N_mf[0] = - min( (tq[1, d, :] <  0) * (1 / sqrt(tq[1, d, :]**2 / Y_t**2 + tq[2, d, :]**2 / S**2) ) \
                       + (tq[1, d, :] >  0) * (1 / sqrt(tq[1, d, :]**2 / Y_c**2 + tq[2, d, :]**2 / S**2) ) \
                       + (tq[1, d, :] == 0) * 0, key = abs)
        # Tensile matrix failure
        N_mf[1] =   min( (tq[1, d, :] >  0) * (1 / sqrt(tq[1, d, :]**2 / Y_t**2 + tq[2, d, :]**2 / S**2) ) \
                       + (tq[1, d, :] <  0) * (1 / sqrt(tq[1, d, :]**2 / Y_c**2 + tq[2, d, :]**2 / S**2) ) \
                       + (tq[1, d, :] == 0) * 0, key = abs)

        #
        N_of = zeros(2)
        # Compressive failure
        N_of[0] = max(N_ff[0], N_mf[0] )
        # Tensile failure
        N_of[1] = min(N_ff[1], N_mf[1] )

        #
        return(N_ff, N_mf, N_of)

    ##
    def FailureEnvelope(data_points, N_ff, N_mf, d1, d2, tqad, X_c, X_t, Y_c, Y_t, S):

        #
        #                           N_1
        #                          θ = 0, 2 * pi
        #                            ^
        #                            |
        #                        \ θ |
        #                         \<-|
        #                          \ |
        #                           \|
        # θ = pi / 2     ------------+------------> N_2, θ = 3 * pi / 2
        #                            |
        #                            |
        #                            |
        #                            |
        #                            |
        #                          θ = pi
        #

        #
        theta = tile(deg2rad(linspace(0, 360 - 360 / data_points, num = data_points) ), (tqad.shape[2], 1) )

        #       _________________        _________________
        #      /   sin^2 theta          / 1 - cos^2 theta
        #     / ----------------- =    / -----------------
        #   \/   1 - sin^2 theta     \/     cos^2 theta
        #
        # ratio_force = sqrt( sin^2 theta / (1 - sin^2 theta) )
        with errstate(divide='ignore'):
            ratio_force = where( (1 - (sin(theta[0, :] ) )**2) != 0, \
                        sqrt( (sin(theta[0, :] ) )**2 / (1 - (sin(theta[0, :] ) )**2) ), \
                        0)

        ## Fiber failure
        #
        coefficient_force_min = (einsum('i,j->ij', tqad[0, d1, :], ones(data_points) ) - einsum('i,j->ij', tqad[0, d2, :], ratio_force) )
        coefficient_force_max = (einsum('i,j->ij', tqad[0, d1, :], ones(data_points) ) + einsum('i,j->ij', tqad[0, d2, :], ratio_force) )

        X_c = tile(X_c, (data_points, 1) ).T
        X_t = tile(X_t, (data_points, 1) ).T

        #
        n = zeros( [tqad.shape[2], data_points, 2] )
        n[:, :, 0] = where( (theta < pi / 2) | (theta > 3 * pi / 2), \
                    ( (coefficient_force_min < 0) * (- X_c) + (coefficient_force_min > 0) * (X_t) ) / coefficient_force_min, \
                    ( (coefficient_force_min > 0) * (- X_c) + (coefficient_force_min < 0) * (X_t) ) / coefficient_force_min)
        n[:, :, 1] = where( (theta < pi / 2) | (theta > 3 * pi / 2), \
                    ( (coefficient_force_max < 0) * (- X_c) + (coefficient_force_max > 0) * (X_t) ) / coefficient_force_max, \
                    ( (coefficient_force_max > 0) * (- X_c) + (coefficient_force_max < 0) * (X_t) ) / coefficient_force_max)

        #
        n = sort(sort(n, axis=0), axis=2)

        #
        MN_ff       = zeros( [data_points, 2] )
        MN_ff[:, 0] = ( (theta[0, :] < pi / 2) | (theta[0, :] > 3 * pi / 2) ) * (n[ 0, :, 0] ) \
                    + ( (theta[0, :] > pi / 2) & (theta[0, :] < 3 * pi / 2) ) * (n[-1, :, 1] )
        MN_ff[:, 1] = ( (theta[0, :] != pi / 2) & (theta[0, :] != 3 * pi / 2) ) * (- MN_ff[:, 0] * tan(theta[0, :] ) ) \
                    + (theta[0, :] == pi / 2) * (N_ff[0] ) \
                    + (theta[0, :] == 3 * pi / 2) * (N_ff[1] )
        MN_ff       = concatenate( (MN_ff, MN_ff[0, :].reshape(1, 2) ), axis=0)

        ## Matrix failure
        #
        a_c =     tqad[1, d2, :] * tqad[1, d2, :] / Y_c**2 \
            +     tqad[2, d2, :] * tqad[2, d2, :] / S**2
        a_t =     tqad[1, d2, :] * tqad[1, d2, :] / Y_t**2 \
            +     tqad[2, d2, :] * tqad[2, d2, :] / S**2
        b_c = 2 * tqad[1, d1, :] * tqad[1, d2, :] / Y_c**2 \
            + 2 * tqad[2, d1, :] * tqad[2, d2, :] / S**2
        b_t = 2 * tqad[1, d1, :] * tqad[1, d2, :] / Y_t**2 \
            + 2 * tqad[2, d1, :] * tqad[2, d2, :] / S**2
        c_c =     tqad[1, d1, :] * tqad[1, d1, :] / Y_c**2 \
            +     tqad[2, d1, :] * tqad[2, d1, :] / S**2
        c_t =     tqad[1, d1, :] * tqad[1, d1, :] / Y_t**2 \
            +     tqad[2, d1, :] * tqad[2, d1, :] / S**2

        #
        coefficient_force_c = einsum('i,j->ij', a_c, ratio_force**2) \
                            + ( ( (theta <     pi / 2) | ( (theta > pi)     & (theta < 3 * pi / 2) ) ) * (- 1) \
                            +   ( (theta > 3 * pi / 2) | ( (theta > pi / 2) & (theta <     pi    ) ) ) * (  1) ) * einsum('i,j->ij', b_c, ratio_force) \
                            + einsum('i,j->ij', c_c, ones(data_points) )
        coefficient_force_t = einsum('i,j->ij', a_t, ratio_force**2) \
                            + ( ( (theta <     pi / 2) | ( (theta > pi)     & (theta < 3 * pi / 2) ) ) * (- 1) \
                            +   ( (theta > 3 * pi / 2) | ( (theta > pi / 2) & (theta <     pi    ) ) ) * (  1) ) * einsum('i,j->ij', b_t, ratio_force) \
                            + einsum('i,j->ij', c_t, ones(data_points) )

        #
        coefficient_tc = einsum('i,j->ij', tqad[1, d1, :], ones(data_points) ) \
                       + ( ( (theta <     pi / 2) | ( (theta > pi)     & (theta < 3 * pi / 2) ) ) * (- 1) \
                       +   ( (theta > 3 * pi / 2) | ( (theta > pi / 2) & (theta <     pi    ) ) ) * (  1) ) * einsum('i,j->ij', tqad[1, d2, :], ratio_force)

        #
        coefficient_force = (coefficient_tc > 0) \
                          * ( ( (theta <= pi / 2) | (theta >= 3 * pi / 2) ) * coefficient_force_t  \
                          +   ( (theta >  pi / 2) & (theta <  3 * pi / 2) ) * coefficient_force_c) \
                          + (coefficient_tc < 0) \
                          * ( ( (theta <= pi / 2) | (theta >= 3 * pi / 2) ) * coefficient_force_c  \
                          +   ( (theta >  pi / 2) & (theta <  3 * pi / 2) ) * coefficient_force_t) \

        #
        n = sqrt(1 / amax(coefficient_force, axis=0) )

        #
        MN_mf       = zeros( [data_points, 2] )
        MN_mf[:, 0] = ( (theta[0, :] < pi / 2) | (theta[0, :] > 3 * pi / 2) ) * (  n) \
                    + ( (theta[0, :] > pi / 2) & (theta[0, :] < 3 * pi / 2) ) * (- n)
        MN_mf[:, 1] = ( (theta[0, :] != pi / 2) & (theta[0, :] != 3 * pi / 2) ) * (- MN_mf[:, 0] * tan(theta[0, :] ) ) \
                    + (theta[0, :] == pi / 2) * (N_mf[0] ) \
                    + (theta[0, :] == 3 * pi / 2) * (N_mf[1] )
        MN_mf       = concatenate( (MN_mf, MN_mf[0, :].reshape(1, 2) ), axis=0)

        ## Overal failure
        #
        mn       = zeros( [data_points + 1, 2] )
        mn[:, 0] = norm(MN_ff,  axis=1)
        mn[:, 1] = norm(MN_mf,  axis=1)

        index_min = argmin(mn, axis=1)

        MN       = zeros( [data_points + 1, 2] )
        MN[:, 0] = (index_min == 0) * MN_ff[:, 0] + (index_min == 1) * MN_mf[:, 0]
        MN[:, 1] = (index_min == 0) * MN_ff[:, 1] + (index_min == 1) * MN_mf[:, 1]

        #
        return(MN_ff, MN_mf, MN)

    # Number of plies in the laminate
    number_of_plies = len(theta)

    #
    X_c = repeat(X_c * ones(number_of_plies), 2)
    #
    X_t = repeat(X_t * ones(number_of_plies), 2)
    #
    Y_c = repeat(Y_c * ones(number_of_plies), 2)
    #
    Y_t = repeat(Y_t * ones(number_of_plies), 2)
    #
    S   = repeat(S * ones(number_of_plies), 2)

    #
    forces = [f'Nx', f'Ny', f'Nxy', f'Mx', f'My', f'Mxy']

    #
    number_of_forces = len(forces)

    #
    HashinRotem = {}

    #
    for i in range(0, number_of_forces):
        #
        locals()[f'{forces[i]}_ff'], locals()[f'{forces[i]}_mf'], locals()[forces[i] ] = SingleFailureLoad(i, tqad, X_c, X_t, Y_c, Y_t, S)
        #
        HashinRotem[forces[i] ] = locals()[forces[i] ]

    #
    for i in range(0, number_of_forces - 1):
        #
        for j in range(i + 1, number_of_forces):
            #
            locals()[f'{forces[i]}{forces[j]}_ff'], locals()[f'{forces[i]}{forces[j]}_mf'], locals()[f'{forces[i]}{forces[j]}'] \
                = FailureEnvelope(data_points, locals()[f'{forces[j]}_ff'], locals()[f'{forces[j]}_mf'], i, j, tqad, X_c, X_t, Y_c, Y_t, S)
            #
            HashinRotem[forces[i] + forces[j] ] = locals()[f'{forces[i]}{forces[j]}']

    #
    if data:
        #
        if exists(f'{simulation}/Data/MaterialFailureEnvelopes/HashinRotemFailureTheory.txt'):
            remove(f'{simulation}/Data/MaterialFailureEnvelopes/HashinRotemFailureTheory.txt')

        #
        textfile = open(f'{simulation}/Data/MaterialFailureEnvelopes/HashinRotemFailureTheory.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Singular failure loads according to the Hashin-Rotem Failure Theory') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: HashinRotemFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(f'Singular failure loads \n')

        #
        textfile.write(tabulate( array( [ [f'N\u2093  [N/m]', locals()[f'{forces[0]}_ff'][0], locals()[f'{forces[0]}_ff'][1], \
                                                             locals()[f'{forces[0]}_mf'][0], locals()[f'{forces[0]}_mf'][1], \
                                                             locals()[forces[0] ][0],        locals()[forces[0] ][1] ], \
                                          [f'N\u1D67  [N/m]', locals()[f'{forces[1]}_ff'][0], locals()[f'{forces[1]}_ff'][1], \
                                                             locals()[f'{forces[1]}_mf'][0], locals()[f'{forces[1]}_mf'][1], \
                                                             locals()[forces[1] ][0],        locals()[forces[1] ][1] ], \
                                          [f'N\u2093\u1D67 [N/m]', locals()[f'{forces[2]}_ff'][0], locals()[f'{forces[2]}_ff'][1], \
                                                                   locals()[f'{forces[2]}_mf'][0], locals()[f'{forces[2]}_mf'][1], \
                                                                locals()[forces[2] ][0],         locals()[forces[2] ][1] ], \
                                          [f'M\u2093  [N]', locals()[f'{forces[3]}_ff'][0], locals()[f'{forces[3]}_ff'][1], \
                                                           locals()[f'{forces[3]}_mf'][0], locals()[f'{forces[3]}_mf'][1], \
                                                           locals()[forces[3] ][0],         locals()[forces[3] ][1] ], \
                                          [f'M\u1D67  [N]', locals()[f'{forces[4]}_ff'][0], locals()[f'{forces[4]}_ff'][1], \
                                                           locals()[f'{forces[4]}_mf'][0], locals()[f'{forces[4]}_mf'][1], \
                                                           locals()[forces[4] ][0],         locals()[forces[4] ][1] ], \
                                          [f'M\u2093\u1D67 [N]', locals()[f'{forces[5]}_ff'][0], locals()[f'{forces[5]}_ff'][1], \
                                                                 locals()[f'{forces[5]}_mf'][0], locals()[f'{forces[5]}_mf'][1], \
                                                                 locals()[forces[5] ][0],         locals()[forces[5] ][1] ] ] ).tolist(),
                    headers=(f'Force', f'Fiber', f'Failure', f'Matrix', f'Failure', f'Hashin-Rotem', f'Failure Theory'),
                    colalign=('left', 'right', 'left', 'right', 'left', 'right', 'left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.5e', '.5e', '.5e', '.5e', '.5e') ) )

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
        if not isdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/HashinRotemFailureTheory'):
            mkdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/HashinRotemFailureTheory')

        #
        for i in range(0, number_of_forces - 1):
            #
            for j in range(i + 1, number_of_forces):
                #
                if exists(f'{simulation}/Illustrations/MaterialFailureEnvelopes/HashinRotemFailureTheory/{forces[i]}{forces[j]}.{fileformat}'):
                    remove(f'{simulation}/Illustrations/MaterialFailureEnvelopes/HashinRotemFailureTheory/{forces[i]}{forces[j]}.{fileformat}')

                # Initiate window
                plt.figure(0)
                # Full screen
                plt.get_current_fig_manager().full_screen_toggle()
                # Create vertical and horizontal line through origin
                plt.axvline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                plt.axhline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                # Create plots
                plt.plot(locals()[f'{forces[i]}{forces[j]}'][:, 1],    locals()[f'{forces[i]}{forces[j]}'][:, 0],    color=Colors[27], linewidth=1, linestyle=LineStyles[0], label=f'Hashin-Rotem Failure Theory')
                plt.plot(locals()[f'{forces[i]}{forces[j]}_ff'][:, 1], locals()[f'{forces[i]}{forces[j]}_ff'][:, 0], color=Colors[21], linewidth=1, linestyle=LineStyles[2], label=f'Fiber Failure')
                plt.plot(locals()[f'{forces[i]}{forces[j]}_mf'][:, 1], locals()[f'{forces[i]}{forces[j]}_mf'][:, 0], color=Colors[22], linewidth=1, linestyle=LineStyles[2], label=f'Matrix Failure')
                #
                plt.plot(0, locals()[f'{forces[i]}'][0], color=Colors[27], marker='.')
                plt.plot(0, locals()[f'{forces[i]}'][1], color=Colors[27], marker='.')
                plt.plot(locals()[f'{forces[j]}'][0], 0, color=Colors[27], marker='.')
                plt.plot(locals()[f'{forces[j]}'][1], 0, color=Colors[27], marker='.')
                #
                plt.plot(0, locals()[f'{forces[i]}_ff'][0], color=Colors[21], marker='1')
                plt.plot(0, locals()[f'{forces[i]}_ff'][1], color=Colors[21], marker='1')
                plt.plot(locals()[f'{forces[j]}_ff'][0], 0, color=Colors[21], marker='1')
                plt.plot(locals()[f'{forces[j]}_ff'][1], 0, color=Colors[21], marker='1')
                #
                plt.plot(0, locals()[f'{forces[i]}_mf'][0], color=Colors[22], marker='1')
                plt.plot(0, locals()[f'{forces[i]}_mf'][1], color=Colors[22], marker='1')
                plt.plot(locals()[f'{forces[j]}_mf'][0], 0, color=Colors[22], marker='1')
                plt.plot(locals()[f'{forces[j]}_mf'][1], 0, color=Colors[22], marker='1')
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
                plt.text(0, -0.15, f'Source: HashinRotemFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                # Save figure
                plt.savefig(f'{simulation}/Illustrations/MaterialFailureEnvelopes/HashinRotemFailureTheory/{forces[i]}{forces[j]}.{fileformat}', bbox_inches='tight')
                # Close figure
                plt.close(0)

    #
    return(HashinRotem)