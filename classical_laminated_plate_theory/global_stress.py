# Import modules
from   datetime          import datetime
from   matplotlib        import rcParams
rcParams['font.size'] = 8
import matplotlib.pyplot as plt
from   numpy             import array, einsum, repeat
from   tabulate          import tabulate
from   textwrap          import fill
from   typing            import Dict, List

# The function 'global_stress'
def global_stress(laminate : Dict[str, List[float] ], settings : Dict[str, str or float], stiffness : Dict[str, List[float] ],
                  strain : Dict[str, List[float] ], data : bool = False, illustrations : bool = False):
    """Computes the global stress at the top and bottom of each ply of a laminate.

    Returns the generated ``stress`` dictionary with a shape-(2N, 3) array containing the three global stresses at the top and bottom of each
    ply based on the local strain (``epsilon_xy``) for a laminate characterized by: the lay-up (``theta``), the z-coordinate of each ply
    (``z``), and the transformed, reduced, stiffness matrix of each ply in the global coordinate system (``Q``). This computation is denoted by
    a moniker (``simulation``). Two optional booleans (``data`` and ``illustrations``) can be defined which respectively indicate if a text
    file (``global_stress.txt``) containing the global stress or if an illustration showing the variation of these stresses throughout the
    laminate should be generated. This illustration uses the stipulated format (``fileformat``) and resolution (``resolution``).

    Parameters
    ----------
    - ``laminate`` : Dict | Dictionary containing:
        - ``theta`` : numpy.ndarray, shape = (N) | The lay-up (in degrees) of the laminate (consisting of N plies).
        - ``z`` : numpy.ndarray, shape = (N + 1) | The z-coordinate of each of the N + 1 ply interfaces.
    - ``settings`` : Dict | Dictionary containing:
        - ``fileformat`` : str | The format of the files as which the generated illustrations are saved.
        - ``resolution`` : float | The resolution of the generated illustrations in dots-per-inch.
        - ``simulation`` : str | The moniker of the simulation.
    - ``stiffness`` : Dict | Dictionary containing:
        - ``Q`` : numpy.ndarray, shape = (3, 3, N) | The transformed, reduced, stiffness matrix in the global 1-2 coordinate system for each of
        the N plies.
    - ``strain`` : Dict | Dictionary containing:
        - ``epsilon_xy`` : numpy.ndarray, shape = (2N, 3) | The strain at the top and bottom of each ply in the local x-y coordinate system.
    - ``data`` : bool, optional | Boolean indicating if a text file containing the three global stresses at the top and bottom of each ply should
    be generated, by default ``False``.
    - ``illustrations`` : bool, optional | Boolean indicating if an illustration containing the variation of the three stresses strains throughout
    the laminate should be generated, by default ``False``.

    Returns
    -------
    - ``stress`` : Dict | Dictionary appended with:
        - ``sigma_12`` : numpy.ndarray, shape = (2N, 3) | The three stresses at the top and bottom of each ply in the global 1-2 coordinate
        system from the top to the bottom of the laminate.

    Output
    ------
    - ``global_stress.txt`` : text file | A text file containing the three global stresses at the top and bottom of each ply from the top to the
    bottom of the laminate.
    - ``global_stress`` : illustration | An illustration showing the variation of the three global stresses throughout the laminate.

    Assumptions
    -----------

    Version
    -------
    - v1.0 :
        - Initial version (05/08/2020) | M. Bilo

    References
    ----------
    [1]: Kassapoglou, Christos. 2010. Design and Analysis of Composite Structures: With Applications to Aerospace Structures. 1st ed.
    Chichester: John Wiley & Sons

    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    [3]: Gerrard, Chris. 2020. accessed 5 August 2020.
    <https://public.tableau.com/profile/chris.gerrard#!/vizhome/TableauColors/ColorPaletteswithRGBValues>

    Verified with
    -------------
    [4]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.3
    """

    # The lay-up of the laminate in degrees
    theta = laminate['theta']
    # The z-coordinate of each ply interface
    z = laminate['z']

    # The format of the files as which the generated illustrations are saved
    fileformat = settings['fileformat']
    # The resolution of the generated illustrations in dots-per-inch
    resolution = settings['resolution']
    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3, N) containing the stiffness tensor of each of the N plies in the global 1-2 coordinate system
    Q = stiffness['Q']

    # An array of shape-(2N, 3) containing the strains at the top and bottom of each ply in the local x-y coordinate system from the top to
    # the bottom of the laminate
    epsilon_xy = strain['epsilon_xy']

    # An array of shape-(2N, 3) containing the three stresses (σ₁, σ₂, and τ₁₂ respectively) at the top and bottom of each ply in the global 1-2
    # coordinate system from the top to the bottom of the laminate (equation 3.32 from [1] or equations 2.94 and 2.98 from [2])
    sigma_12 = einsum('ijk,kj->ki', repeat(Q, 2, axis = 2), epsilon_xy)

    # If a text file containing the three global stresses should be generated or if an illustration showing the variation of these stresses
    # should be generated
    if data or illustrations:
        # An array of shape-(2N) containing the z-coordinates of the top and bottom of each ply, from the top to the bottom of the laminate,
        # based on the z-coordinates of the ply interfaces of the laminate
        Z = repeat(z, 2)[1:-1]
    else:
        # Create this variable as a placeholder to avoid the message that the variable Z is possibly unbound
        Z = None

    # If a text file containing the three global stresses at the top and bottom of each ply should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/global_stress.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'The stresses in the global 1-2 coordinate system at the top and bottom of each ply from the top to the '
                                f'bottom of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: global_stress.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the global stress at the top and bottom of each ply from the top to the bottom of the laminate
            textfile.write(tabulate(array( [repeat(theta, 2), Z, sigma_12[:, 0], sigma_12[:, 1], sigma_12[:, 2] ] ).T.tolist(),
                        headers = [f'\u03B8 [\u00B0]', f'z [m]', f'\u03C3\u2081 [-]', f'\u03C3\u2082 [-]', f'\u03C4\u2081\u2082 [-]'],
                        stralign = 'left', numalign = 'decimal', floatfmt = ('.0f', '.3e', '.5e', '.5e', '.5e') ) )

    # If an illustration showing the variation of the three global stresses throughout the laminate should be generated
    if illustrations:
        # Use the Tableau 30 colors (consisting of Tableau 10, Tableau 10 Medium, and Tableau 10 Light from [3]) for the figure
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

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        # For each ply in the laminate
        for i in range(len(theta) ):
            # Add the variation of the normal stress in the global 1 direction throughout a ply
            plt.plot(sigma_12[2 * i:2 * i + 2, 0], Z[2 * i:2 * i + 2], color = Colors[0], linewidth = 1)

            # Add the variation of the normal stress in the global 2 direction throughout a ply
            plt.plot(sigma_12[2 * i:2 * i + 2, 1], Z[2 * i:2 * i + 2], color = Colors[3], linewidth = 1)

            # Add the variation of the shear strain in the global 1-2 plane throughout a ply
            plt.plot(sigma_12[2 * i:2 * i + 2, 2], Z[2 * i:2 * i + 2], color = Colors[6], linewidth = 1)

        # Connect all line segments describing the variation of the normal stress in the global 1 direction through each ply
        plt.plot(sigma_12[:, 0], Z, color = Colors[0], linewidth = 1, linestyle = 'dashed')

        # Connect all line segments describing the variation of the normal stress in the global 2 direction through each ply
        plt.plot(sigma_12[:, 1], Z, color = Colors[3], linewidth = 1, linestyle = 'dashed')

        # Connect all line segments describing the variation of the shear stress in the global 1-2 plane through each ply
        plt.plot(sigma_12[:, 2], Z, color = Colors[6], linewidth = 1, linestyle = 'dashed')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Add a vertical line through the origin
        plt.axvline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'\u03C3 [-]')

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'z [m]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend( (f'\u03C3\u2081', f'\u03C3\u2082', f'\u03C4\u2081\u2082'),
                    loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, edgecolor='inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                 fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: global_stress.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                 fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/classical_laminated_plate_theory/global_stress.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()

    # Create the stress dictionary
    stress = {}
    # Append the stress dictionary with the global stress at the top and bottom of each ply from the top to the bottom of the laminate
    stress['sigma_12'] = sigma_12

    # End the function and return the generated stress
    return(stress)
