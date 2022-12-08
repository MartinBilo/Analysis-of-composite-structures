# Import modules
from datetime import datetime
from numpy    import allclose, array, isclose
from numpy.linalg import inv
from os       import mkdir
from os.path  import isdir
from tabulate import tabulate
from textwrap import fill
from typing   import Dict, List


# The function 'stability_analysis'
def stability_analysis(assumeddeflection : Dict[str, List[float] ], geometry : Dict[str, float], settings : Dict[str, str or float],
                                   stiffness : Dict[str, List[float] ] ) -> None:

    #
    a = geometry['a']
    #
    b = geometry['b']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the extensional stiffness (A-)matrix
    A = stiffness['A']
    # An array of shape-(3, 3) containing the coupling stiffness (B-)matrix
    B = stiffness['B']
    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    #
    if not isdir(f'{simulation}/illustrations/stability_analysis/Whitney_method'):
        mkdir(f'{simulation}/illustrations/stability_analysis/Whitney_method')

    #
    if not isdir(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey'):
        mkdir(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey')

    #
    if not isdir(f'{simulation}/illustrations/stability_analysis/Galerkin_method'):
        mkdir(f'{simulation}/illustrations/stability_analysis/Galerkin_method')

    # #
    # if not isdir(f'{simulation}/data/stability_analysis/Ritz_method'):
    #     mkdir(f'{simulation}/data/stability_analysis/Ritz_method')

    # #
    # if not isdir(f'{simulation}/illustrations/stability_analysis/Ritz_method'):
    #     mkdir(f'{simulation}/illustrations/stability_analysis/Ritz_method')

    # #
    # if not isdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NxNy'):
    #     mkdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NxNy')

    # #
    # if not isdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NNxy'):
    #     mkdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NNxy')

    # #
    # if not isdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NxM'):
    #     mkdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NxM')

    # #
    # if not isdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NyM'):
    #     mkdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/NyM')

    # #
    # if not isdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/MNxy'):
    #     mkdir(f'{simulation}/illustrations/stability_analysis/Ritz_method/MNxy')

    ################################################################################################################################

    #
    from stability_analysis.Galerkin_method    import Galerkin_method
    from stability_analysis.Kassapoglou_survey import Kassapoglou_survey
    from stability_analysis.Lévy_method        import Lévy_method
    # from stability_analysis.Ritz_method        import Ritz_method
    from stability_analysis.Seydel_method      import Seydel_method
    from stability_analysis.Whitney_method     import Whitney_method

    N_Kassapoglou, m_Kassapoglou, n_Kassapoglou = Kassapoglou_survey(geometry, settings, stiffness, data = True, illustrations = True)

    # N_Lévy, m_Lévy, n_Lévy = Lévy_method(geometry, settings, stiffness)

    N_Seydel, xi = Seydel_method(geometry, settings, stiffness, data = True)

    N_Whitney, m_Whitney, n_Whitney = Whitney_method(geometry, settings, stiffness, data = True, illustrations = True)

    # #
    # if not allclose(B, 0):
    #     # The reduced bending stiffness approximation (section 7.8 from [1]) via expressions 2.63 and 2.64 from [1]
    #     stiffness['D'] = D - B @ inv(A) @ B

    # #
    # if isclose(D[0, 2], 0) and isclose(D[1, 2], 0) and allclose(B, 0):
    #         #
    #         if not isdir(f'{simulation}/data/stability_analysis/Kassapoglou_survey'):
    #             mkdir(f'{simulation}/data/stability_analysis/Kassapoglou_survey')

    #         #
    #         if not isdir(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey'):
    #             mkdir(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey')

    #         #
    #         if not isdir(f'{simulation}/data/stability_analysis/Lévy_method'):
    #             mkdir(f'{simulation}/data/stability_analysis/Lévy_method')

    #         #
    #         if not isdir(f'{simulation}/data/stability_analysis/Seydel_method'):
    #             mkdir(f'{simulation}/data/stability_analysis/Seydel_method')

    #         #
    #         if not isdir(f'{simulation}/data/stability_analysis/Whitney_method'):
    #             mkdir(f'{simulation}/data/stability_analysis/Whitney_method')

    #         #
    #         N_Kassapoglou, m_Kassapoglou, n_Kassapoglou = Kassapoglou_survey(geometry, settings, stiffness)

    #         #
    #         N_Lévy, m_Lévy, n_Lévy = Lévy_method()

    #         #
    #         N_Seydel, xi = Seydel_method(geometry, settings, stiffness, data = True)

    #         #
    #         N_Whitney, m_Whitney, n_Whitney = Whitney_method(geometry, settings, stiffness)

    #
    N_Galerkin = Galerkin_method(geometry, settings, stiffness, data = True, illustrations = True)

    # #
    # N_Ritz = Ritz_method(assumeddeflection, geometry, settings, stiffness)

    # # Comparison

    # # Create and open the text file
    # with open(f'{simulation}/data/stability_analysis/Comparison.txt', 'w', encoding = 'utf8') as textfile:
    #     # Add a description of the contents
    #     textfile.write(fill(f'Comparison') )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the source and a timestamp
    #     textfile.write(fill(f'Source: .py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     textfile.write(fill(f'Simply supported; N\u1D6A [N/m]') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Whitney',     N_Whitney[0],     m_Whitney[0],     n_Whitney[0]     ],
    #                                      [f'Lévy',        N_Lévy[0],        m_Lévy[0],        n_Lévy[0]        ],
    #                                      [f'Kassapoglou', N_Kassapoglou[0], m_Kassapoglou[0], n_Kassapoglou[0] ] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'm [-]', f'n [-]'],
    #                 colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.5e', '.5e', '.0f', '.0f') ) )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     textfile.write(fill(f'Simply supported; N\u1D67 [N/m]') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Whitney',     N_Whitney[1],     m_Whitney[1],     n_Whitney[1] ],
    #                                      [f'Lévy',        N_Lévy[1],        m_Lévy[1],        n_Lévy[1]    ],
    #                                      [f'Kassapoglou', N_Kassapoglou[1], m_Kassapoglou[1], n_Kassapoglou[1] ] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'm [-]', f'n [-]'],
    #                 colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.5e', '.5e', '.0f', '.0f') ) )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     aspect_ratio = f'a / b = {a / b:.3f}' if a <= b else f'b / a = {b / a:.3f}'
    #     textfile.write(fill(f'Simply supported; N\u1D6A\u1D67 [N/m]; {aspect_ratio}') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Galerkin',    f'\u00B1{N_Galerkin[-1]:.5e}',   len(N_Galerkin) + 1, f'{error_Galerkin[-1]:.2e}'  ],
    #                                      [f'Ritz',        f'\u00B1{N_Ritz[j[5], 5]:.5e}',  j[5] + 1,            f'{error_Ritz[j[5], 5]:.2e}' ],
    #                                      [f'Kassapoglou', f'\u00B1{N_Kassapoglou[2]:.5e}', f'-',                f'-'] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'm = n [-]', f'\u03F5 [%]'],
    #                 colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.5e', '.5e', '.0f','.2e') ) )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     #
    #     N_xy, __, __, __, AR = shear_simplysupported_infinite(D, a, b)

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     aspect_ratio = f'a / b' if a <= b else f'b / a'
    #     textfile.write(fill(f'Simply supported; N\u1D6A\u1D67 [N/m]; Infinite strip ({aspect_ratio} = \u221E)') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Seydel',      f'\u00B1{N_Seydel[0]:.5e}', xi[0]          ],
    #                                      [f'Kassapoglou', f'\u00B1{N_xy:.5e}',        AR * max(a, b) ] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'Half-wave length [m]'],
    #                 colalign = ('left', 'center', 'center'), floatfmt = ('.5e', '.5e', '.2f') ) )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     textfile.write(fill(f'Simply supported (x = 0,a); clamped (y = 0,b); N\u1D6A [N/m]') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Lévy',        N_Lévy[6],        m_Lévy[6],        n_Lévy[6]           ],
    #                                      [f'Kassapoglou', N_Kassapoglou[9], m_Kassapoglou[9], n_Kassapoglou[9]    ] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'm [-]', f'n [-]'],
    #                 colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.5e', '.5e', '.0f', '.0f') ) )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     textfile.write(fill(f'Simply supported (y = 0,b); clamped (x = 0,a); N\u1D67 [N/m]') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Lévy',        N_Lévy[7],         m_Lévy[7],         n_Lévy[7]            ],
    #                                      [f'Kassapoglou', N_Kassapoglou[12], m_Kassapoglou[12], n_Kassapoglou[12]    ] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'm [-]', f'n [-]'],
    #                 colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.5e', '.5e', '.0f', '.0f') ) )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     textfile.write(fill(f'Simply supported (x = 0,a); free (y = 0,b); N\u1D6A [N/m]') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Lévy',                        N_Lévy[4],        m_Lévy[4],        n_Lévy[4]        ],
    #                                      [f'Kassapoglou',                 N_Kassapoglou[5], m_Kassapoglou[5], n_Kassapoglou[5] ],
    #                                      [f'Kassapoglou [approximation]', N_Kassapoglou[6], m_Kassapoglou[6], n_Kassapoglou[6] ] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'm [-]', f'n [-]'],
    #                 colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.5e', '.5e', '.0f', '.0f') ) )

    #     # Add an empty line
    #     textfile.write(f'\n\n')

    #     # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
    #     textfile.write(fill(f'Simply supported (y = 0,b); free (x = 0,a); N\u1D67 [N/m]') )
    #     textfile.write(f'\n')
    #     textfile.write(tabulate(array( [ [f'Lévy',                        N_Lévy[5],        m_Lévy[5],        n_Lévy[5]        ],
    #                                      [f'Kassapoglou',                 N_Kassapoglou[7], m_Kassapoglou[7], n_Kassapoglou[7] ],
    #                                      [f'Kassapoglou [approximation]', N_Kassapoglou[8], m_Kassapoglou[8], n_Kassapoglou[8] ] ] ).tolist(),
    #                 headers = [f'Method', f'Buckling load', f'm [-]', f'n [-]'],
    #                 colalign = ('left', 'center', 'center', 'center'), floatfmt = ('.5e', '.5e', '.0f', '.0f') ) )

    #
    return
