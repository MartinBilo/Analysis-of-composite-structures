# Import modules
from datetime  import datetime
from jax.numpy import array, cos, deg2rad, einsum, ones, sin, zeros
from tabulate  import tabulate
from textwrap  import fill
import numpy   as np

# The function 'ply_stiffness'
def ply_stiffness(laminate : dict[list[float] ], material : dict[list[float], list[float], list[float], list[float], list[str] ],
                  settings : dict[str], data : bool = False) -> tuple[dict[list[float] ], dict[list[float] ] ] :
    """Computes the transformed, reduced, stiffness matrix in the global 1-2 coordinate system for each of the N plies of a laminate.

    Returns the ``material`` dictionary appended with the Poisson's ratio in the longitudinal direction for the transverse direction (``nu_21``)
    for all plies combined (float), if applicable, or each of the N plies separately (shape-(N) array) and a ``stiffness`` dictionary with an
    array of shape-(3, 3, N) containing the transformed, reduced, stiffness matrix in the global 1-2 coordinate system (``Q``) for each of the
    N plies characterized by: the lay-up (``theta``), the Poisson's ratio in the transverse direction for the longitudinal direction (``nu_12``),
    the Young's modulus in the longitudinal (``E_1``) and transverse direction (``E_2``) and the shear modulus (``G_12``). The computation is
    furthermore denoted by a moniker (``simulation``). Lastly, an optional boolean (``data``) can be defined which indicates if a text file
    (``ply_stiffness.txt``) containing the components of the stiffness tensor should be generated.

    Parameters
    ----------
    - `laminate` : Dict | Dictionary containing:
        - `theta` : jax.numpy.ndarray, shape = N | The lay-up (in degrees) of the laminate (consisting of N plies).
    - `material` : Dict | Dictionary containing:
        - `nu_12` : float or jax.numpy.ndarray, shape = N | The Poisson's ratio in the transverse direction for the longitudinal direction for
        all plies combined (float), if applicable, or each of the N plies separately (shape-(N) array).
        - `E_1` : float or jax.numpy.ndarray, shape = N | The Young's modulus in the longitudinal direction for all plies combined (float), if
        applicable, or each of the N plies separately (shape-(N) array).
        - `E_2` : float or jax.numpy.ndarray, shape = N | The Young's modulus in the transverse direction for all plies combined (float), if
        applicable, or each of the N plies separately (shape-(N) array).
        - `G_12` : float or jax.numpy.ndarray, shape = N | The shear modulus for all plies combined (float), if applicable, or each of the N
        plies separately (shape-(N) array).
        - `composite` : str or List[str], length = N | The material classification for all plies combined (str), if applicable, or each of the
        N plies separately (length-(N) list).
    - `settings` : Dict | Dictionary containing:
        - `simulation` : str | The moniker of the simulation.
    - `data` : bool, optional | Boolean indicating if a text file containing the in-plane and flexural thermal coefficients should be created
    as output, by default `False`.

    Returns
    -------
    - `material` : Dict | Dictionary appended with:
        - `nu_21` : float or jax.numpy.ndarray, shape = N | The Poisson's ratio in the longitudinal direction for the transverse direction for
        all plies combined (float), if applicable, or each of the N plies separately (shape-(N) array).
    - `stiffness` : Dict | Dictionary containing:
        - `Q` : jax.numpy.ndarray, shape = (3, 3, N) | The transformed, reduced, stiffness matrix in the global 1-2 coordinate system for each
        of the N plies.

    Output
    ------
    - 'ply_stiffness.txt' : text file | A text file containing the components of the stiffness tensor in the global 1-2 coordinate system for
    each unique material and direction in the laminate.

    Assumptions
    -----------

    Version
    -------
    - v1.0 :
        - Initial version (04/08/2020) | M. Bilo
    - v1.1 :
        - Updated documentation (13/11/2024) | M. Bilo

    References
    ----------
    [1]: Kassapoglou, Christos. 2013. Design and Analysis of Composite Structures: With Applications to Aerospace Structures. 2nd ed.
    Chichester: John Wiley & Sons

    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    Verified with
    -------------
    [3]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.2
    """

    # Version of the script
    version = f'v1.1'

    # The local x-y and global 1-2, right-handed, orthogonal, Cartesian, coordinate system for a ply rotated by an angle θ are defined in
    # figure 1. For the local classification the fibers are aligned with the x direction whereas the global system is lined up with the length
    # and width of the rectangular laminate, the 1 and 2 direction respectively. Conforming to the right-handed coordinate system, the angle θ
    # is defined positive from the x-axis towards the y-axis.
    #
    #                        x
    #                    θ = 0°, 360°
    #                  1     ^
    #                   \  θ |
    #                    \<->|
    #                     \  |
    #                      \ |
    #                       \| z
    # y, θ = 90° <-----------o------------- θ = 270°
    #                    ^  /|
    #                  θ | / |
    #                    v/  |
    #                    /   |
    #                   2    |
    #                     θ = 180°
    #
    # Figure 1: Local x-y and global 1-2 coordinate system for a ply rotated by an angle θ (based on figures 3.1, 3.2 and 3.4 from [1]). NOTE:
    # Even though the global 1-2 coordinate system is orthogonal, this is unfortunately not apparent from the figure.

    # The lay-up of the laminate in degrees
    theta = laminate['theta']

    # The Poisson's ratio in the local x-direction for the local y-direction of a / each ply (figure 1)
    nu_xy = material['nu_xy']
    # The Young's modulus in the local x-direction of a / each ply (figure 1)
    E_x = material['E_x']
    # The Young's modulus in the local y-direction of a / each ply (figure 1)
    E_y = material['E_y']
    # The shear modulus in the local x-y plane of a / each ply (figure 1)
    G_xy = material['G_xy']
    # The classification of the material(s) comprising the laminate
    composite = material['composite']

    # The moniker of the simulation
    simulation = settings['simulation']

    # The Poisson's ratio in the local y-direction for the local x-direction of a / each ply (figure 1) (combining equations 3.24, 3.27, 3.29
    # from [1] or equation 2.67 from [2] which follows from an identical derivation)
    #
    #              Eᵧ
    # νᵧₓ = νₓᵧ ⋅ -----
    #              Eₓ
    #
    nu_yx = nu_xy * E_y / E_x

    # The number of plies, N, comprising the laminate
    number_of_plies = len(theta)

    # Two arrays of shape-(N) containing recurring components of the standard, tensor transformation matrix for each of the N plies (equations
    # 2.97a and 2.97b from [2])
    #
    # c = cos(θ), and
    #
    # s = sin(θ)
    #
    c = cos(deg2rad(theta) )
    s = sin(deg2rad(theta) )

    # An array of shape-(3, 3, N) containing the transformation matrix for each of the N plies (equation 2.96 from [2])
    #      _                                    _
    #     |     c ⋅ c       s ⋅ s       2 ⋅ c ⋅ s  |
    # T = |     s ⋅ s       c ⋅ c     - 2 ⋅ c ⋅ s  |
    #     |_  - c ⋅ s       c ⋅ s        c² - s² _|
    #
    T = array( [ [  c * c, s * s,   c * s + c * s],
                 [  s * s, c * c, - c * s - c * s],
                 [- c * s, c * s,   c * c - s * s] ] )

    # An array of shape-(3, 3, N) containing the inverse of the transformation matrix for each of the N plies (equation 2.95 from [2])
    #        _                                    _
    #       |     c ⋅ c       s ⋅ s     - 2 ⋅ c ⋅ s  |
    # T⁻¹ = |     s ⋅ s       c ⋅ c       2 ⋅ c ⋅ s  |
    #       |_    c ⋅ s     - c ⋅ s        c² - s² _|
    #
    T_inverse = array( [ [c * c,   s * s, - c * s - c * s],
                         [s * s,   c * c,   c * s + c * s],
                         [c * s, - c * s,   c * c - s * s] ] )

    # An array of shape-(3, 3) containing the Reuter matrix (equation 2.101 from [2])
    #      _                 _     _             _
    #     |   R₁₁   0    0    |   |   1   0   0   |
    # R = |    0   R₂₂   0    | = |   0   1   0   |
    #     |_   0    0   R₃₃  _|   |_  0   0   2  _|
    #
    R = array( [ [1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 2] ] )

    # An array of shape-(3, 3) containing the inverse of the Reuter matrix (derived from expression 2.101 from [2])
    #        _                    _     _                     _
    #       |    1                 |   |                       |
    #       |   ---    0     0     |   |     1     0     0     |
    #       |   R₁₁                |   |                       |
    #       |          1           |   |                       |
    # R⁻¹ = |    0    ---    0     | = |     0     1     0     |
    #       |         R₂₂          |   |                       |
    #       |                1     |   |                 1     |
    #       |    0     0    ---    |   |     0     0    ---    |
    #       |_              R₃₃   _|   |_                2    _|
    #
    R_inverse = array( [ [1, 0,     0],
                         [0, 1,     0],
                         [0, 0, 1 / 2] ] )

    # An array of shape-(3, 3, N) containing the reduced stiffness matrix in the local x-y coordinate system for each of the N plies (figure 1)
    # (expression 3.31 from [1], or equations 2.78 and 2.93 from [2])
    #      _                                            _
    #     |        Eₗ               νₗₜ ⋅ Eₜ                ₗ|
    #     |   ------------      -------------       0    |
    #     |    1 - νₗₜ ⋅ νₜₗ         1 - νₗₜ ⋅ νₜₗ             ₗₗ|
    # _   |      νₗₜ ⋅ Eₜ               Eₜ                  ₗ|
    # Q = |   ------------      -------------       0    |
    #     |    1 - νₗₜ ⋅ νₜₗ         1 - νₗₜ ⋅ νₜₗ             ₗₗ|
    #     |                                              |
    #     |        0                  0             Gₗₜ   ₗ|
    #     |_                                            _|
    #
    Q_bar = array( [ [        E_x / (1 - nu_xy * nu_yx) *  ones(number_of_plies),
                      nu_xy * E_y / (1 - nu_xy * nu_yx) *  ones(number_of_plies),
                                                          zeros(number_of_plies) ],
                     [nu_xy * E_y / (1 - nu_xy * nu_yx) *  ones(number_of_plies),
                              E_y / (1 - nu_xy * nu_yx) *  ones(number_of_plies),
                                                          zeros(number_of_plies) ],
                     [                                    zeros(number_of_plies),
                                                          zeros(number_of_plies),
                                                   G_xy *  ones(number_of_plies) ] ] )

    # An array of shape-(3, 3, N) containing the transformed, reduced, stiffness matrix in the global 1-2 coordinate system for each of the N
    # plies (figure 1) (expressions 2.102 and 2.103 from [2] and equivalent to equation 3.33 from [1] and expressions 2.104a-f from [2])
    #         _
    # Q = T⁻¹ Q R T R⁻¹
    #
    Q = einsum('onl,nml,mk,kil,ij->ojl', T_inverse, Q_bar, R, T, R_inverse)

    # If a text file containing the transformed, reduced, stiffness matrix of the laminate should be generated
    if data:
        # If the number of plies is unequal to the indicated number of materials in the laminate
        if number_of_plies != len(composite):
            # Expand the list containing the classification of the material(s) comprising the laminate to all plies
            composite = np.array( [composite for i in range(number_of_plies) ] )

        # Determine the unique material(s) in the laminate
        unique_materials = np.unique(composite)

        # An array of shape-(N, 7) containing the ply orientation and corresponding components of the transformed, reduced, stiffness tensor in
        # the global 1-2 coordinate system for each of the N plies
        TMP = array( [theta, Q[0, 0, :], Q[1, 1, :], Q[0, 1, :], Q[2, 2, :], Q[0, 2, :], Q[1, 2, :] ] ).T

        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/ply_stiffness.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'Transformed, reduced, stiffness tensor in the global 1-2 coordinate system for each unique material and'
                                f' direction in the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: ply_stiffness.py [{version}] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # For all unique material(s) in the laminate
            for mat in unique_materials:
                # For the material(s) under consideration, determine the first occurrences of unique ply directions from the top to the bottom
                # of the laminate
                __, index_unique_theta = np.unique(theta[ [i for i, x in enumerate(composite) if x == mat] ], return_index = True)

                # For the material(s) under consideration, filter the array containing the ply orientation and corresponding components of the
                # transformed, reduced, stiffness tensor of the entire laminate
                tmp = TMP[ [i for i, x in enumerate(composite) if x == mat] , :]

                # Add an empty line
                textfile.write(f'\n\n')

                # Add the classification of the material
                textfile.write(f'{mat}\n')

                # Add the components of the stiffness tensor in the global 1-2 coordinate system for each unique lay-up direction for the
                # material under consideration
                textfile.write(tabulate(tmp[index_unique_theta, :].tolist(),
                                        headers = [f'\u0398 [\u00B0]',
                                                   f'Q\u2081\u2081 [N/m\u00B2]', f'Q\u2082\u2082 [N/m\u00B2]', f'Q\u2081\u2082 [N/m\u00B2]',
                                                   f'Q\u2086\u2086 [N/m\u00B2]', f'Q\u2081\u2086 [N/m\u00B2]', f'Q\u2082\u2086 [N/m\u00B2]' ],
                                        numalign = ('decimal'), floatfmt = ('.0f', '.5e', '.5e', '.5e', '.5e', '.5e', '.5e') ) )

    # Append the material dictionary with the Poisson's ratio in the transverse direction for the longitudinal direction of a / each ply
    material['nu_yx'] = nu_yx

    # Create the stiffness dictionary
    stiffness = {}
    # Append the stiffness dictionary with the stiffness tensor in the global 1-2 coordinate system for each ply of the laminate
    stiffness['Q'] = Q

    # End the function and return the appended material dictionary and the generated stiffness dictionary
    return(material, stiffness)
