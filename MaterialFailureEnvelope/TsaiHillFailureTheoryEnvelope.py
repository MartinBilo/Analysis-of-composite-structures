def TsaiHillFailureTheoryEnvelope(laminate, material, settings, tqad, data=False, illustrations=False):
    """[summary]

    Args:
        material ([type]): [description]
        settings ([type]): [description]
        tqad ([type]): [description]
        data (bool, optional): [description]. Defaults to False.
        illustrations (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    # Import packages into library
    from   datetime          import datetime
    from   matplotlib        import rcParams
    rcParams['font.size'] = 8
    import matplotlib.pyplot as     plt
    from   numpy             import amax, array, concatenate, einsum, errstate, linspace, ones, pi, repeat, sin, sqrt, tan, where, zeros
    from   os                import mkdir, remove
    from   os.path           import exists, isdir
    from   tabulate          import tabulate
    from   textwrap          import fill

    ## Import the laminate properties
    # The layup of the laminate in degrees
    theta = laminate['theta']

    ## Import the material properties
    # Import the failure strength along the fibers in Pascal
    X = material['X']
    # Import the failure strength transverse to the fibers in Pascal
    Y = material['Y']
    # Pure shear failure strength in Pascal
    S = material['S']

    ## Import the settings of the simulation
    # The number of data points of the material failure envelope
    data_points = settings['data_points']
    # The fileformat of the illustrations
    fileformat  = settings['fileformat']
    # The designation of the simulation
    simulation  = settings['simulation']

    # SingularFailureLoad
    def SingularFailureLoad(d, tqad, X, Y, S):
        """Returns the specified compressive and tensile singular material failure load of a laminate according to the Tsai-Hill Failure
           Theory.

        Args:
            d    (float): Number (0, 1, 2, 3, 4 or 5) indicting the singular load under consideration (Nₓ, Nᵧ, Nₓᵧ, Mₓ, Mᵧ or Mₓᵧ
                          respectively).
            tqad (array): Three-dimensional array of size three by, summed throughout the laminate, the number of ply interfaces of each
                          ply by six containing the coefficients of a lamiante relating the six loads to the three internal stresses. See
                          MaterialFailureEnvelopes.py for more information.
            X    (array): One-dimensional array containing the failure strength in Pascal along the fibers for each ply interface of each ply.
            Y    (array): One-dimensional array containing the failure strength in Pascal transverse to the fibers for each ply interface of
                          each ply.
            S    (array): One-dimensional array containing the pure shear failure strength in Pascal for each ply interface of each ply.

        Returns:
            array: One-dimensional array of size two containing the compressive and tensile singular material failure loads of a laminate
                   according to the Tsai-Hill Failure Theory.

        Sources:
            [1]: Kassapoglou, C 2010, Design and analysis of composite structures: With applications to aerospace structures. John Wiley
                 & Sons, Weiheim, Germany.
        """

        # Initialize the array containing the specified compressive and tensile singular material material failure load of the laminate
        N = zeros(2)

        # In MaterialFailureEnvelopes.py it is shown that
        #
        #  σ₁  = tqad[0, d, :] ⋅ N ,
        #
        #  σ₂  = tqad[1, d, :] ⋅ N , and                                                                                        (MFE.X)
        #
        #  τ₁₂ = tqad[2, d, :] ⋅ N .
        #
        # The Tsai-Hill Failure Theory (equation 4.8 from [1]) is defined as
        #
        #  σ₁² / X² - σ₁ ⋅ σ₂ / X² + σ₂² / Y² + τ₁₂² / S² = 1 .                                                                 (SFL.1)
        #
        # Combining and rearranging the previous expressions results in the following expression for the singular material failure load of
        # each ply interface of each ply of a laminate
        #
        #          _                                                                                           _ - ¹/₂
        #         |    tqad[0, d, :]²     tqad[0, d, :] ⋅⋅ tqad[1, d, :]      tqad[0, d, :]²    tqad[2, d, :]²   |
        #  N = ±  |   --------------- -  ------------------------------- +  --------------- + ---------------   |               (SFL.2)
        #         |_        X²                          X²                        Y²                S²         _|

        # Specified compressive singular failure load of the laminate according to the Tsai-Hill Failure Theory (equation SFL.2)
        N[0] = - min( (tqad[0, d, :] * tqad[0, d, :] / X**2 \
                     - tqad[0, d, :] * tqad[1, d, :] / X**2 \
                     + tqad[1, d, :] * tqad[1, d, :] / Y**2 \
                     + tqad[2, d, :] * tqad[2, d, :] / S**2)**(- 1 / 2) )

        # Specified tensile singular failure load of the laminate according to the Tsai-Hill Failure Theory (equation SFL.2)
        N[1] = - N[0]

        # End the function and return the array containing the specified compressive and tensile singular material failure load of the
        # laminate
        return(N)

    # FailureEnvelope
    def FailureEnvelope(data_points, F2, d1, d2, tqad, X, Y, S):
        """[summary]

        Args:
            data_points (float): [description]
            F2 (array): [description]
            d    (float): Number (0, 1, 2, 3, 4 or 5) indicting the first load under consideration (Nₓ, Nᵧ, Nₓᵧ, Mₓ, Mᵧ or Mₓᵧ respectively).
            d2   (float): Number (0, 1, 2, 3, 4 or 5) indicting the second load under consideration (Nₓ, Nᵧ, Nₓᵧ, Mₓ, Mᵧ or Mₓᵧ respectively).
            tqad (array): Three-dimensional array of size three by, summed throughout the laminate, the number of ply interfaces of each
                          ply by six containing the coefficients of a lamiante relating the six loads to the three internal stresses. See
                          MaterialFailureEnvelopes.py for more information.
            X    (array): One-dimensional array containing the failure strength in Pascal along the fibers for each ply interface of each ply.
            Y    (array): One-dimensional array containing the failure strength in Pascal transverse to the fibers for each ply interface of
                          each ply.
            S    (array): One-dimensional array containing the pure shear failure strength in Pascal for each ply interface of each ply.

        Returns:
            array: Two-dimensional array of size datapoints + one by two containing the compressive and tensile singular material failure loads of a laminate
                   according to the Tsai-Hill Failure Theory.

        Sources:
            [1]: Kassapoglou, C 2010, Design and analysis of composite structures: With applications to aerospace structures. John Wiley
                 & Sons, Weiheim, Germany.
        """

        # The plane defined by the two independent loads under consideration (N₁ and N₂), which contains the corresponding material
        # failure envelope, is depicted in figure 1 below. In addition, the angle θ between both material failure loads is specified.
        #
        #                        N₁
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
        # Figure FE.1: Illustration of the N₁N₂-plane and the angle
        #          θ between two material failure loads.
        #
        # The following relationships between two failure loads and the angle θ can be deduced from this figure:
        #           ___________
        #  N₁ =   \/ N₁² + N₂²  ⋅ cos θ  -->  N₁² = (N₁² + N₂²) ⋅ cos² θ , and                  (FE.1)
        #
        #           ___________
        #  N₂ = - \/ N₁² + N₂²  ⋅ sin θ  -->  N₂² = (N₁² + N₂²) ⋅ sin² θ ,                      (FE.2)
        #
        # which can be combined to yield
        #
        #                _____________             ______________
        #               /   sin² θ                /  1 - cos² θ
        #  N₂ = ± N₁   / ------------- =  ± N₁   / -------------- = ± N₁ ⋅ r₂₁ .                (FE.3)
        #            \/   1 - sin² θ           \/      cos² θ
        #
        # For the N₁N₂-plane, this expression can be used to reduce the number of unkown variables in a failure theory from two to one. It can
        # furthermore be surmised that this expression does not hold for θ = π / 2 or θ = 3π / 2 since equation FE.1 and FE.2
        # reduce to
        #
        #  N₁ = 0 , and
        #
        #  N₂ = N₂ ,
        #
        # for these conditions thereby decoupling both material failure loads.

        # Based on the specified number of data points, an array containing the angle θ, in radians, where the failure envelope in the
        # N₁N₂-plane is evaluated
        theta = 2 * pi / 360 * linspace(0, 360 - 360 / data_points, num = data_points)

        # The corresponding ratio between both material failure loads as defined in expression FE.3
        with errstate(divide='ignore'):
            ratio_force = where( (1 - (sin(theta) )**2) != 0, sqrt( (sin(theta) )**2 / (1 - (sin(theta) )**2) ), 0)

        # In MaterialFailureEnvelopes.py it is shown that
        #
        #  σ₁  = tqad[0, d1, :] ⋅ N₁ + tqad[0, d2, :] ⋅ N₂ ,
        #
        #  σ₂  = tqad[1, d1, :] ⋅ N₁ + tqad[1, d2, :] ⋅ N₂ , and                                (MFE.X)
        #
        #  τ₁₂ = tqad[2, d1, :] ⋅ N₁ + tqad[2, d2, :] ⋅ N₂ .
        #
        # From these expression in combination with equation FE.3 it follows that
        #
        #  σ₁  = N₁ ⋅ (tqad[0, d1, :] ± r₂₁ ⋅ tqad[0, d2, :] ) ,
        #
        #  σ₂  = N₁ ⋅ (tqad[1, d1, :] ± r₂₁ ⋅ tqad[1, d2, :] ) , and                            (FE.4)
        #
        #  τ₁₂ = N₁ ⋅ (tqad[2, d1, :] ± r₂₁ ⋅ tqad[2, d2, :] ) .
        #
        # The Tsai-Hill Failure Theory (equation 4.8 from [1]) is defined as
        #
        #  σ₁² / X² - σ₁ ⋅ σ₂ / X² + σ₂² / Y² + τ₁₂² / S² = 1 .                                 (FE.5)
        #
        # Combining and rearranging the previous two expression results in
        #                                                                         ___
        #  N₁² ⋅ (r₂₁² ⋅ a ± r₂₁ ⋅ b + c) = 1  -->  N₁² ⋅ cᶠ = 1  --> N₁ = ± 1 / \/cᶠ) .         (FE.6)
        #

        # The(se) coefficients of the Tsai-Hill Failure Theory are given by
        a =     tqad[0, d2, :] * tqad[0, d2, :] / X**2 \
          -     tqad[0, d2, :] * tqad[1, d2, :] / X**2 \
          +     tqad[1, d2, :] * tqad[1, d2, :] / Y**2 \
          +     tqad[2, d2, :] * tqad[2, d2, :] / S**2
        b = 2 * tqad[0, d1, :] * tqad[0, d2, :] / X**2 \
          -     tqad[0, d1, :] * tqad[1, d2, :] / X**2 \
          -     tqad[1, d1, :] * tqad[0, d2, :] / X**2 \
          + 2 * tqad[1, d1, :] * tqad[1, d2, :] / Y**2 \
          + 2 * tqad[2, d1, :] * tqad[2, d2, :] / S**2
        c =     tqad[0, d1, :] * tqad[0, d1, :] / X**2 \
          -     tqad[0, d1, :] * tqad[1, d1, :] / X**2 \
          +     tqad[1, d1, :] * tqad[1, d1, :] / Y**2 \
          +     tqad[2, d1, :] * tqad[2, d1, :] / S**2

        # The sign of load N₂, and therefore of coefficient b in expression FE.6, can be determined with figure FE.1
        #
        #   0     <= θ <=  π / 2  -->  N₁ > 0  -->  N₂ < 0  -->  N₂ = - r₂₁ ⋅ N₁  -->  - b ,
        #   π / 2 <= θ <=  π      -->  N₁ < 0  -->  N₂ < 0  -->  N₂ = + r₂₁ ⋅ N₁  -->  + b ,
        #   π     <= θ <= 3π / 2  -->  N₁ < 0  -->  N₂ > 0  -->  N₂ = - r₂₁ ⋅ N₁  -->  - b , and   (FE.7)
        #  3π / 2 <= θ <= 2π      -->  N₁ > 0  -->  N₂ > 0  -->  N₂ = + r₂₁ ⋅ N₁  -->  + b ,
        #
        # resulting in the following expression for the force coefficient (cᶠ)
        c_f = einsum('i,j->ij', a, ratio_force**2) \
            + ( (    0      <= theta) & (theta <       pi / 2) * (- 1) \
            +   (    pi / 2 <  theta) & (theta <       pi)     * (  1) \
            +   (    pi     <= theta) & (theta <   3 * pi / 2) * (- 1) \
            +   (3 * pi / 2 <  theta) & (theta <=  2 * pi)     * (  1) ) \
            * einsum('i,j->ij', b, ratio_force) \
            + einsum('i,j->ij', c, ones(data_points) )

        # Initialize the array containing in each row the combination of failure loads of the laminate for each angle theta
        MN = zeros( [data_points, 2] )

        # Based on equation FE.6, the first failure load of the laminate for each angle under consideration is therefore then given by
        # the minimum load, corresponding to the maximum force coefficient, for all ply interfaces. The sign and direction of the failure
        # load is based on figure FE.1
        #
        #   0     <= θ <   π / 2  -->  N₁ > 0
        #            θ =   π / 2  -->  N₁ = 0
        #   π / 2 <  θ <  3π / 2  -->  N₁ < 0                                                                     (FE.8)
        #            θ =  3π / 2  -->  N₁ = 0
        #  3π / 2 <  θ <= 2π      -->  N₁ > 0
        #
        MN[:, 0] = ( ( (theta <  pi / 2) | (theta >  3 * pi / 2) ) * (  1) \
                 +   ( (theta >  pi / 2) & (theta <  3 * pi / 2) ) * (- 1) ) * sqrt(1 / amax(c_f, axis=0) )

        # The second failure load is given by
        #
        #  N₂ = - N₁ ⋅ tan θ                                                                                       (FE.9)
        #
        # as derived from figure FE.1. Like equation FE.3, this expression does not hold for θ = π / 2 or θ = 3π / 2. For these angles
        # the failure loads as obtained with SingularFailureLoad are substituted.
        MN[:, 1] = ( (theta !=     pi / 2) \
                 &   (theta != 3 * pi / 2) ) * (- MN[:, 0] * tan(theta) ) \
                 +   (theta ==     pi / 2)   * (F2[0] ) \
                 +   (theta == 3 * pi / 2)   * (F2[1] )

        # Append the array containing the combination of failure loads for each angle theta with the first row of this array to obtain a
        # closed failure envelope graphically
        MN       = concatenate( (MN, MN[0, :].reshape(1, 2) ), axis=0)

        # End the function and return the array describing the material failure envelope
        return(MN)

    # The number of plies in the laminate
    number_of_plies = len(theta)

    # Expand the variable containing the failure strength along the fibers, if applicable, to each ply and subsequently each ply interface of
    # each ply
    X = repeat(X * ones(number_of_plies), 2)
    # Expand the variable containing the failure strength along the fibers, if applicable, to each ply and subsequently each ply interface of
    # each ply
    Y = repeat(Y * ones(number_of_plies), 2)
    # Expand the variable containing the pure shear failure strength, if applicable, to each ply and subsequently each ply interface of each
    # ply
    S = repeat(S * ones(number_of_plies), 2)

    # List containing the forces taking into consideration
    forces = [f'Nx', f'Ny', f'Nxy', f'Mx', f'My', f'Mxy']

    # The number of forces taken into consideration
    number_of_forces = len(forces)

    # Initialize empty dictionary containing the singular material failure loads and material failure load envelopes
    TsaiHill = {}

    #
    for i in range(0, number_of_forces):
        #
        locals()[forces[i] ] = SingularFailureLoad(i, tqad, X, Y, S)
        #
        TsaiHill[forces[i] ] = locals()[forces[i] ]

    #
    for i in range(0, number_of_forces - 1):
        #
        for j in range(i + 1, number_of_forces):
            #
            locals()[f'{forces[i]}{forces[j]}'] \
                = FailureEnvelope(data_points, locals()[forces[j] ], i, j, tqad, X, Y, S)
            #
            TsaiHill[forces[i] + forces[j] ] = locals()[f'{forces[i]}{forces[j]}']

    #
    if data:
        #
        if exists(f'{simulation}/Data/MaterialFailureEnvelopes/TsaiHillFailureTheory.txt'):
            remove(f'{simulation}/Data/MaterialFailureEnvelopes/TsaiHillFailureTheory.txt')

        #
        textfile = open(f'{simulation}/Data/MaterialFailureEnvelopes/TsaiHillFailureTheory.txt', 'w', encoding='utf8')

        #
        textfile.write(fill(f'Singular failure loads according to the Tsai-Hill Failure Theory') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(fill(f'Source: TsaiHillFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        #
        textfile.write(f'\n\n')

        #
        textfile.write(f'Singular failure loads \n')

        #
        textfile.write(tabulate( array( [ [f'N\u2093  [N/m]', locals()[forces[0] ][0], locals()[forces[0] ][1] ], \
                                          [f'N\u1D67  [N/m]', locals()[forces[1] ][0], locals()[forces[1] ][1] ], \
                                          [f'N\u2093\u1D67 [N/m]', locals()[forces[2] ][0], locals()[forces[2] ][1] ], \
                                          [f'M\u2093  [N]', locals()[forces[3] ][0], locals()[forces[3] ][1] ], \
                                          [f'M\u1D67  [N]', locals()[forces[4] ][0], locals()[forces[4] ][1] ], \
                                          [f'M\u2093\u1D67 [N]', locals()[forces[5] ][0], locals()[forces[5] ][1] ] ] ).tolist(),
                    headers=(f'Force', f'Tsai-Hill', f'Failure Theory'),
                    colalign=('left', 'right', 'left'),
                    numalign=('decimal'),
                    floatfmt=('.5e', '.5e', '.5e') ) )

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

        # Labels
        Labels = [f'N\u2093 [N/m]', f'N\u1D67 [N/m]', f'N\u2093\u1D67 [N/m]', f'M\u2093 [N]', f'M\u1D67 [N]', f'M\u2093\u1D67 [N]']

        #
        if not isdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiHillFailureTheory'):
            mkdir(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiHillFailureTheory')

        #
        for i in range(0, number_of_forces - 1):
            #
            for j in range(i + 1, number_of_forces):
                #
                if exists(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiHillFailureTheory/{forces[i]}{forces[j]}.{fileformat}'):
                    remove(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiHillFailureTheory/{forces[i]}{forces[j]}.{fileformat}')

                # Initiate window
                plt.figure(0)
                # Full screen
                plt.get_current_fig_manager().full_screen_toggle()
                # Create vertical and horizontal line through origin
                plt.axvline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                plt.axhline(color="black", linewidth=0.5, linestyle=(0, (5, 10) ) )
                # Create plots
                plt.plot(locals()[f'{forces[i]}{forces[j]}'][:, 1], locals()[f'{forces[i]}{forces[j]}'][:, 0], color=Colors[6], linewidth=1)
                plt.plot(0, locals()[forces[i] ][0], color=Colors[6], marker='.')
                plt.plot(0, locals()[forces[i] ][1], color=Colors[6], marker='.')
                plt.plot(locals()[forces[j] ][0], 0, color=Colors[6], marker='.')
                plt.plot(locals()[forces[j] ][1], 0, color=Colors[6], marker='.')
                # Format axes
                plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0) )
                # Format xlabel
                plt.xlabel(Labels[j] )
                # Format ylabel
                plt.ylabel(Labels[i] )
                # Create text wit simulation name and source
                plt.text(0, -0.125, f'Simulation: {simulation}', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                plt.text(0, -0.15, f'Source: TsaiHillFailureTheoryEnvelope.py [v1.0] via CompositeAnalysis.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).', fontsize = 6, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
                # Save figure
                plt.savefig(f'{simulation}/Illustrations/MaterialFailureEnvelopes/TsaiHillFailureTheory/{forces[i]}{forces[j]}.{fileformat}', bbox_inches='tight')
                # Close figure
                plt.close(0)

    #
    return(TsaiHill)