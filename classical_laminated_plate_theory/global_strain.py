# Import modules
from   datetime   import datetime
from   matplotlib import rcParams
rcParams['font.size'] = 8
import matplotlib.pyplot as plt
from   numpy      import array, einsum, ones, repeat
from   tabulate   import tabulate
from   textwrap   import fill
from   typing     import Dict, List

# The function 'global_strain'
def global_strain(laminate : Dict[str, List[float] ], settings : Dict[str, str or float], strain : Dict[str, List[float] ],
                  data : bool = False, illustrations : bool = False) -> Dict[str, List[float] ]:
    """Computes the global strain at the top and bottom of each ply of a laminate.

    Returns the `strain` dictionary appended with a shape-(2N, 3) array containing the three global strains at the top and bottom of each ply
    based on the mid-plane strain and curvatures (`epsilon_0`) for a laminate characterized by the lay-up (`theta`) and the z-coordinate
    of each ply interface (`z`). This computation is denoted by a moniker (`simulation`). Two optional booleans (`data` and
    `illustrations`) can be defined which respectively indicate if a text file (`global_strain.txt`) containing the global strains or if an
    illustration showing the variation of these strains throughout the laminate should be generated. This illustration uses the stipulated
    format (`fileformat`) and resolution (`resolution`).

    Parameters
    ----------
    - `laminate` : Dict | Dictionary containing:
        - `theta` : numpy.ndarray, shape = (N) | The lay-up (in degrees) of the laminate (consisting of N plies).
        - `z` : numpy.ndarray, shape = (N + 1) | The z-coordinate of each of the N + 1 ply interfaces.
    - `settings` : Dict | Dictionary containing:
        - `fileformat` : str | The format of the files as which the generated illustrations are saved.
        - `resolution` : float | The resolution of the generated illustrations in dots-per-inch.
        - `simulation` : str | The moniker of the simulation.
    - `strain` : Dict | Dictionary containing:
        - `epsilon_0` : numpy.ndarray, shape = (6) | The mid-plane strains and curvatures.
    - `data` : bool, optional | Boolean indicating if a text file containing the three global strains at the top and bottom of each ply should
    be generated, by default `False`.
    - `illustrations` : bool, optional | Boolean indicating if an illustration containing the variation of the three global strains throughout
    the laminate should be generated, by default `False`.

    Returns
    -------
    - `strain` : Dict | Dictionary appended with:
        - `epsilon_12` : numpy.ndarray, shape = (2N, 3) | The three strains at the top and bottom of each ply in the global 1-2 coordinate
        system from the top to the bottom of the laminate.

    Output
    ------
    - `global_strain.txt` : text file | A text file containing the three global strains at the top and bottom of each ply from the top to the
    bottom of the laminate.
    - `global_strain` : illustration | An illustration showing the variation of the three global strains throughout the laminate.

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

    [2]: Gerrard, Chris. 2020. accessed 5 August 2020.
    <https://public.tableau.com/profile/chris.gerrard#!/vizhome/TableauColors/ColorPaletteswithRGBValues>

    Verified with
    -------------
    [3]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.3
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

    # An array of shape-(6) containing the three mid-plane/membrane strains and three constant curvatures
    epsilon_0 = strain['epsilon_0']

    # The number of plies, N, comprising the laminate
    number_of_plies = len(theta)

    # An array of shape-(2N) containing the z-coordinates of the top and bottom of each ply, from the top to the bottom of the laminate, based
    # on the z-coordinates of the ply interfaces of the laminate
    Z = repeat(z, 2)[1:-1]

    # An array of shape-(2N, 3) containing the three strains (ε₁, ε₂, and γ₁₂ respectively) at the top and bottom of each ply in the global 1-2
    # coordinate system from the top to the bottom of the laminate (equation 3.48 from [1])
    epsilon_12 = einsum('i,j->ij', ones(2 * number_of_plies), epsilon_0[:3] ) \
               + einsum('i,j->ij',                         Z, epsilon_0[3:] )

    # If a text file containing the three global strains at the top and bottom of each ply should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/global_strain.txt', 'w', encoding='utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'Strains in the global 1-2 coordinate system at the top and bottom of each ply from the top to the bottom '
                                f'of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: global_strains.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the global strain at the top and bottom of each ply from the top to the bottom of the laminate
            textfile.write(tabulate(array( [repeat(theta, 2), Z, epsilon_12[:, 0], epsilon_12[:, 1], epsilon_12[:, 2] ] ).T.tolist(),
                        headers = [f'\u03B8 [\u00B0]', f'z [m]', f'\u03B5\u2081 [-]', f'\u03B5\u2082 [-]', f'\u03B3\u2081\u2082 [-]'],
                        stralign = 'center', numalign = 'decimal', floatfmt = ('.0f', '.3e', '.5e', '.5e', '.5e') ) )

    # If an illustration showing the variation of the three global strains throughout the laminate should be generated
    if illustrations:
        # Use the Tableau 30 colors (consisting of Tableau 10, Tableau 10 Medium, and Tableau 10 Light from [2]) for the figure
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

        # Add the variation of the normal strain in the global 1 direction throughout the laminate
        plt.plot(epsilon_12[:, 0], Z, color = Colors[0], linewidth = 1, label = f'\u03B5\u2081')

        # Add the variation of the normal strain in the global 2 direction throughout the laminate
        plt.plot(epsilon_12[:, 1], Z, color = Colors[3], linewidth = 1, label = f'\u03B5\u2082')

        # Add the variation of the shear strain in the global 1-2 plane throughout the laminate
        plt.plot(epsilon_12[:, 2], Z, color = Colors[6], linewidth = 1, label = f'\u03B3\u2081\u2082')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Add a vertical line through the origin
        plt.axvline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'\u03B5 [-]')

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'z [m]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                 fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: global_strain.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                 fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/classical_laminated_plate_theory/global_strain.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()

    # Append the strain dictionary with the global strain at the top and bottom of each ply from the top to the bottom of the laminate
    strain['epsilon_12'] = epsilon_12

    # End the function and return the appended strain dictionary
    return(strain)
