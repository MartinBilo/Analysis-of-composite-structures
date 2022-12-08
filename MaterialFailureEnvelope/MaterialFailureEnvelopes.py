def MaterialFailureEnvelopes(laminate, material, stiffness, settings, data=False, illustrations=False):

    # Import packages into library
    from   datetime          import datetime
    from   matplotlib        import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as     plt
    from   numpy             import append, array, concatenate, cos, delete, einsum, insert, arange, ones, pi, repeat, sin, zeros, reshape
    from   numpy.linalg      import norm
    from   os                import mkdir, remove
    from   os.path           import exists, isdir
    from   tabulate          import tabulate
    from   textwrap          import fill

    # Import functions into library
    from MaterialFailureEnvelope.MaximumStrainFailureTheoryEnvelope import MaximumStrainFailureTheoryEnvelope
    from MaterialFailureEnvelope.MaximumStressFailureTheoryEnvelope import MaximumStressFailureTheoryEnvelope
    from MaterialFailureEnvelope.TsaiHillFailureTheoryEnvelope      import TsaiHillFailureTheoryEnvelope
    from MaterialFailureEnvelope.HashinRotemFailureTheoryEnvelope   import HashinRotemFailureTheoryEnvelope
    from MaterialFailureEnvelope.TsaiWuFailureTheoryEnvelope        import TsaiWuFailureTheoryEnvelope

    ##
    #
    theta = laminate['theta']
    #
    z = laminate['z']

    ##
    #
    X             = material['X']
    #
    X_c           = material['X_c']
    #
    X_t           = material['X_t']
    #
    Y             = material['Y']
    #
    Y_c           = material['Y_c']
    #
    Y_t           = material['Y_t']
    #
    S             = material['S']
    #
    epsilon_u_x_c = material['epsilon_u_x_c']
    #
    epsilon_u_x_t = material['epsilon_u_x_t']
    #
    epsilon_u_y_c = material['epsilon_u_y_c']
    #
    epsilon_u_y_t = material['epsilon_u_y_t']
    #
    gamma_u_xy    = material['gamma_u_xy']

    ##
    #
    alpha = stiffness['alpha']
    #
    beta  = stiffness['beta']
    #
    delta = stiffness['delta']
    #
    Q = stiffness['Q']

    ##
    #
    data_points = settings['data_points']
    #
    fileformat  = settings['fileformat']
    #
    simulation  = settings['simulation']


    # Number of plies in the laminate
    number_of_plies = len(theta)

    #
    Z = repeat(z, 2)[1:-1]

    ##
    # i -> force; j -> stress; k -> ply interface
    a = einsum('k,ij->ijk', ones(2 * number_of_plies), alpha) + einsum('k,ij->ijk', Z, beta)

    #
    d = einsum('k,ij->ijk', ones(2 * number_of_plies),  beta) + einsum('k,ij->ijk', Z, delta)

    # in radians
    Theta = repeat(2 * pi * theta / 360, 2)

    # Components of the standard tensor transformation for each ply (equation 3.33 from [1])
    m = cos(Theta)
    n = sin(Theta)

    # Transformation matrix
    t = array( [ [  m * m, n * n,   2 * m * n], \
                 [  n * n, m * m, - 2 * m * n], \
                 [- m * n, m * n, m**2 - n**2] ] )

    ##
    #
    tad = concatenate( (einsum('ijk,jlk->ilk', t, a), \
                        einsum('ijk,jlk->ilk', t, d) ), axis = 1)

    #
    q = repeat(Q, 2, axis = 2)

    ##
    #
    tqad = concatenate( (einsum('ijk,jlk->ilk', t, einsum('ijk,jlk->ilk', q, a) ), \
                         einsum('ijk,jlk->ilk', t, einsum('ijk,jlk->ilk', q, d) ) ), axis = 1)

    #
    MaterialFailure = {}

    #
    Data_neg = []
    Data_pos = []

    #
    Header = [f'N\u2093  [N/m]', f'N\u1D67  [N/m]', f'N\u2093\u1D67 [N/m]', f'M\u2093  [N]', f'M\u1D67  [N]', f'M\u2093\u1D67 [N]']

    #
    Column = []

    #
    def MaterialFailureSummary(neg_array_target, pos_array_target, dic_target, dic_source, column, column_target):
        column_target.append(column)
        column_target.append(f'')
        if dic_target:
            for key, value in dic_source.items():
                if len(value) == 2:
                    neg_array_target.append(value[0] )
                    pos_array_target.append(value[1] )
                    dic_target[key][0] = max(value[0], dic_target[key][0] )
                    dic_target[key][1] = min(value[1], dic_target[key][1] )
                else:
                    Distance1 = norm(value, axis = 1)
                    Distance2 = norm(dic_target[key], axis = 1)
                    dic_target[key][:, 0] = (Distance1 <= Distance2) * value[:, 0] \
                                          + (Distance1 >  Distance2) * dic_target[key][:, 0]
                    dic_target[key][:, 1] = (Distance1 <= Distance2) * value[:, 1] \
                                          + (Distance1 >  Distance2) * dic_target[key][:, 1]
        else:
            dic_target = dic_source.copy()
            for key, value in dic_target.items():
                if len(value) == 2:
                    neg_array_target.append(value[0] )
                    pos_array_target.append(value[1] )

        return(neg_array_target, pos_array_target, dic_target, column_target)

    #
    if epsilon_u_x_c != 0 and epsilon_u_x_t != 0 and epsilon_u_y_c != 0 and epsilon_u_y_t != 0 and gamma_u_xy != 0:
        #
        MaximumStrain = MaximumStrainFailureTheoryEnvelope(laminate, material, settings, tad, data=True, illustrations=True)
        #
        Data_neg, Data_pos, MaterialFailure, Column = MaterialFailureSummary(Data_neg, Data_pos, MaterialFailure, MaximumStrain, 'Maximum Strain Failure Theory', Column)

    #
    if X_c != 0 and X_t != 0 and Y_c != 0 and Y_t != 0 and S != 0:
        #
        MaximumStress = MaximumStressFailureTheoryEnvelope(laminate, material, settings, tqad, data=True, illustrations=True)
        #
        Data_neg, Data_pos, MaterialFailure, Column = MaterialFailureSummary(Data_neg, Data_pos, MaterialFailure, MaximumStress, 'Maximum Stress Failure Theory', Column)

    #
    if X != 0 and Y != 0 and S != 0:
        #
        TsaiHill = TsaiHillFailureTheoryEnvelope(laminate, material, settings, tqad, data=True, illustrations=True)
        #
        Data_neg, Data_pos, MaterialFailure, Column = MaterialFailureSummary(Data_neg, Data_pos, MaterialFailure, TsaiHill, 'Tsai-Hill Failure Theory', Column)

    #
    if X_c != 0 and X_t != 0 and Y_c != 0 and Y_t != 0 and S != 0:
        #
        TsaiWu = TsaiWuFailureTheoryEnvelope(laminate, material, settings, tqad, data=True, illustrations=True)
        #
        Data_neg, Data_pos, MaterialFailure, Column = MaterialFailureSummary(Data_neg, Data_pos, MaterialFailure, TsaiWu, 'Tsai-Wu Failure Theory', Column)

    #
    if X_c != 0 and X_t != 0 and Y_c != 0 and Y_t != 0 and S != 0:
        #
        HashinRotem = HashinRotemFailureTheoryEnvelope(laminate, material, settings, tqad, data=True, illustrations=True)
        #
        Data_neg, Data_pos, MaterialFailure, Column = MaterialFailureSummary(Data_neg, Data_pos, MaterialFailure, HashinRotem, 'Hashin-Rotem Failure Theory', Column)

    Data_neg, Data_pos, __, __ = MaterialFailureSummary(Data_neg, Data_pos, MaterialFailure, MaterialFailure, 'Material Failure', Column)

    Data = zeros( [12, len(Data_pos) // 6] )
    Data[::2, :]  = reshape(Data_neg, (6, len(Data_neg) // 6) )
    Data[1::2, :] = reshape(Data_pos, (6, len(Data_pos) // 6) )
    Data = append(reshape(Column, (12, 1) ), Data, axis = 1)

    #
    if data:
        #
        if exists(f'{simulation}/Data/MaterialFailureEnvelopes/MaterialFailureSummary.txt'):
            remove(f'{simulation}/Data/MaterialFailureEnvelopes/MaterialFailureSummary.txt')

        #
        textfile = open(f'{simulation}/Data/MaterialFailureEnvelopes/MaterialFailureSummary.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Singular failure loads according to various, applicable failure theories') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: MaterialFailureEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(f'Singular failure loads \n')

        #
        textfile.write(tabulate(Data.tolist(),
                    headers=Header,
                    colalign=('left', 'center', 'center', 'center', 'center', 'center', 'center'),
                    numalign=('decimal'),
                    floatfmt=('.5e') ) )

        #
        textfile.close()

    #
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
        forces = [f'Nx', f'Ny', f'Nxy', f'Mx', f'My', f'Mxy']

        #
        number_of_forces = len(forces)

        #
        if not isdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaterialFailureSummary'):
            mkdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaterialFailureSummary')

        #
        for i in range(0, number_of_forces - 1):
            #
            for j in range(i + 1, number_of_forces):
                #
                if exists(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaterialFailureSummary/{forces[i]}{forces[j]}.{fileformat}'):
                    remove(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaterialFailureSummary/{forces[i]}{forces[j]}.{fileformat}')

                # Initiate window
                plt.figure(0)
                # Full screen
                plt.get_current_fig_manager().full_screen_toggle()
                # Create vertical and horizontal line through origin
                plt.axvline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                plt.axhline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                # Create plots
                if epsilon_u_x_c != 0 and epsilon_u_x_t != 0 and epsilon_u_y_c != 0 and epsilon_u_y_t != 0 and gamma_u_xy != 0:
                    plt.plot(MaximumStrain[f'{forces[i]}{forces[j]}'][:, 1], MaximumStrain[f'{forces[i]}{forces[j]}'][:, 0],   color=Colors[0],  linewidth=1, linestyle=LineStyles[0], label=f'Maximum Strain Failure Theory')
                if X_c != 0 and X_t != 0 and Y_c != 0 and Y_t != 0 and S != 0:
                    plt.plot(MaximumStress[f'{forces[i]}{forces[j]}'][:, 1], MaximumStress[f'{forces[i]}{forces[j]}'][:, 0],   color=Colors[3],    linewidth=1, linestyle=LineStyles[0], label=f'Maximum Stress Failure Theory')
                    plt.plot(TsaiWu[f'{forces[i]}{forces[j]}'][:, 1],        TsaiWu[f'{forces[i]}{forces[j]}'][:, 0],          color=Colors[9],    linewidth=1, linestyle=LineStyles[0], label=f'Tsai-Wu Failure Theory')
                    plt.plot(HashinRotem[f'{forces[i]}{forces[j]}'][:, 1],   HashinRotem[f'{forces[i]}{forces[j]}'][:, 0],     color=Colors[27],   linewidth=1, linestyle=LineStyles[0], label=f'Hashin-Rotem Failure Theory')
                if X != 0 and Y != 0 and S != 0:
                    plt.plot(TsaiHill[f'{forces[i]}{forces[j]}'][:, 1],      TsaiHill[f'{forces[i]}{forces[j]}'][:, 0],        color=Colors[6],    linewidth=1, linestyle=LineStyles[0], label=f'Tsai-Hill Failure Theory')
                plt.plot(MaterialFailure[f'{forces[i]}{forces[j]}'][:, 1],   MaterialFailure[f'{forces[i]}{forces[j]}'][:, 0], color='xkcd:black', linewidth=1, linestyle=LineStyles[2], label=f'Material Failure')
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
                plt.text(0, -0.15, f'Source: MaterialFailureEnvelopes.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                # Save figure
                plt.savefig(f'{simulation}/Illustrations/MaterialFailureEnvelopes/MaterialFailureSummary/{forces[i]}{forces[j]}.{fileformat}', bbox_inches='tight')
                # Close figure
                plt.close(0)

    #
    return