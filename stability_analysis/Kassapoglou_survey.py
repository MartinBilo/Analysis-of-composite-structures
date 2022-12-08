# Import modules
from   datetime          import datetime
from   matplotlib        import rcParams
rcParams['font.size'] = 8
import matplotlib.pyplot as plt
from   numpy             import append, arctan, argmin, array, ceil, concatenate, cos, einsum, errstate, expand_dims, finfo, floor, isclose, \
                                linspace, meshgrid, ones, pi, roots, sin, sqrt, take_along_axis, tan, zeros
from   numpy.ma          import masked_equal
from   scipy.optimize    import fsolve
from   tabulate          import tabulate
from   textwrap          import fill
from   typing            import Dict, List, Tuple


# Simplysupported (single normal force)
def normal_force_simplysupported(kappa : float, b : float, D : List[float], direction : int = 1) -> Tuple[float, float, float]:
    #
    #           -----------------------
    #           |     ^    ss         |
    #           |     |               |
    # N₁ -----> | ss  | b          ss | <----- N₁
    #           |     |               |
    #           |     v    ss         |
    #           -----------------------
    #

    # The buckling load for an uniaxial, compressive, load in the 1-direction for a simply supported laminate is given by (equation 6.7 from
    # [1])
    #              _                                  _
    #         π²  |                                r⁴  |
    # N₁ = - ---- | D₁₁m² + 2(D₁₂ + 2D₆₆)r² + D₂₂ ---- | .
    #         a²  |_                               m² _|
    #
    # The number of half-waves in the 1-direction that minimizes the buckling load can be found by equating the derivative of this load with
    # respect to the number of half-waves to zero
    #               _                 _                    _______
    #  δNᵪ      π²  |               r⁴  |                4 /  D₂₂
    # ---- = - ---- | 2D₁₁m - 2D₂₂ ---- | = 0 --> m = r   /  -----  = κ
    #  δm       a²  |_              m³ _|               \/    D₁₁
    #
    # which, as expected and required, is identical to the expression derived for κ (kappa) for the function 'Whitney_method' in
    # Whitney_method.py.

    #
    m_min = masked_equal(array( [floor(kappa), ceil(kappa) ] ), 0)

    # The buckling load for an uniaxial, compressive, load in the 1-direction can be rewritten in line with the other equations in table 6.1
    # from [1] to yield
    #              _                                  _
    #         π²  |                                r⁴  |      π²    _______
    # Nᵪ = - ---- | D₁₁m² + 2(D₁₂ + 2D₆₆)r² + D₂₂ ---- | = - ---- \/ D₁₁D₂₂ K
    #         a²  |_                               m² _|      b²
    #
    # where
    #
    #      m²     2(D₁₂ + 2D₆₆)     κ²
    # K = ---- + --------------- + ---- .
    #      κ²      \/ D₁₁D₂₂        m²

    #
    with errstate(divide = 'ignore'):
        K = m_min**2 / kappa**2 + 2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) + kappa**2 / m_min**2

    #
    if isclose(direction, 1):
        m = f'{int(m_min[argmin(K) ] ) }'
    else:
        m = f'1'

    #
    if isclose(direction, 1):
        n = f'1'
    else:
        n = f'{int(m_min[argmin(K) ] ) }'

    # (table 6.1 from [1])
    N = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * K.min()

    #
    return(N, m, n)


# Clampedsimplysupported (single normal force)
def normal_force_clamped_simplysupported(kappa : float, b : float, D : List[float], direction : int = 1) -> Tuple[float, float, float]:
    #
    #           -----------------------
    #           |     ^    ss         |
    #           |     |               |
    # N₁ -----> | c   | b           c | <----- N₁
    #           |     |               |
    #           |     v    ss         |
    #           -----------------------
    #

    # The buckling load for an uniaxial, compressive, load in the 1-direction for a clamped & simply supported laminate is given by (table 6.1
    # from [1])
    #
    #          π²    _______
    # N₁ =  - ---- \/ D₁₁D₂₂ K .
    #          b²
    #
    # where the constant K is given by
    #
    #      4      2(D₁₂ + 2D₆₆)     3
    # K = ---- + --------------- + --- κ²                                                                   for 0 < κ <= 1.662, and
    #      κ²      \/ D₁₁D₂₂        4
    #
    #      m⁴ + 8m² + 1     2(D₁₂ + 2D₆₆)       κ²
    # K = -------------- + --------------- + --------                                                       for 1.662 < κ.
    #       κ² (m + 1)       \/ D₁₁D₂₂        m² + 1
    #
    # The parameter κ (kappa) is defined as (table 6.1 from [1] and the function 'Whitney_method' in Whitney_method.py)
    #           _______
    #        4 /  D₂₂
    # κ = r   /  -----  .
    #       \/    D₁₁
    #
    # The number of half-waves in the 1-direction that minimizes the buckling load can be found by equating the derivative of the constant K
    # with respect to the number of half-waves to zero
    #
    #  δK
    # ---- = 0 ∀ m                                    --> m = 1                                             for 0 < κ <= 1.662, and
    #  δm
    #
    #  δK     - m⁴ + 4m³ - 8m² + (16 - 2κ⁴)m - 1
    # ---- = ------------------------------------ = 0 --> - m⁴ + 4m³ - 8m² + (16 - 2κ⁴)m - 1 = 0 ∧ m ≠ -1   for 1.662 < κ.
    #  δm                κ² (m + 1)²

    #
    m_roots    = roots(array( [-1, 4, -8, 16 - 2 * kappa**4, -1] ) )
    m_filtered = m_roots[isclose(m_roots.imag, 0) & (m_roots.real > 0) ].real
    m_min      = masked_equal(concatenate( (floor(m_filtered), ceil(m_filtered) ) ), 0)

    if m_min.size == 0:
        m_min = array( [1] )

    #
    with errstate(divide = 'ignore'):
        k = (m_min**4 + 8 * m_min**2 + 1) / (kappa**2 * (m_min + 1) ) + kappa**2 / (m_min**2 + 1)

    #
    K = 2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) \
      + (kappa <  1.662) * (4 / kappa**2 + 3 / 4 * kappa**2) \
      + (kappa >= 1.662) * k.min()

    #
    if isclose(direction, 1):
        m = f'{int( (kappa <  1.662) * 1 + (kappa >= 1.662) * m_min[argmin(k) ] ) }'
    else:
        m = f'1'

    #
    if isclose(direction, 1):
        n = f'1'
    else:
        n = f'{int( (kappa <  1.662) * 1 + (kappa >= 1.662) * m_min[argmin(k) ] ) }'

    #
    N = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * K

    #
    return(N, m, n)


# Simplysupportedclamped (single, normal force)
def normal_force_simplysupported_clamped(kappa : float, b : float, D : List[float], direction : int = 1) -> Tuple[float, float, float]:
    #
    #           -----------------------
    #           |     ^    c          |
    #           |     |               |
    # N₁ -----> | ss  | b          ss | <----- N₁
    #           |     |               |
    #           |     v    c          |
    #           -----------------------
    #

    # The buckling load for an uniaxial, compressive, load in the 1-direction for a simply supported & clamped laminate is given by (table 6.1
    # from [1])
    #
    #          π²    _______
    # N₁ =  - ---- \/ D₁₁D₂₂ K
    #          b²
    #
    # where the constant K is given by
    #
    #      m²    2(D₁₂ + 2D₆₆)     16   κ²
    # K = --- + --------------- + ---- ---- .
    #      κ²     \/ D₁₁D₂₂        3    m²
    #
    # The parameter κ (kappa) is defined as (table 6.1 from [1] and the function 'Whitney_method' in Whitney_method.py)
    #           _______
    #        4 /  D₂₂
    # κ = r   /  -----  .
    #       \/    D₁₁
    #
    # The number of half-waves in the 1-direction that minimizes the buckling load can be found by equating the derivative of the constant K
    # with respect to the number of half-waves to zero
    #                                         ____
    #  δK     2m     32   κ²               4 / 16
    # ---- = ---- - ---- ---- = 0 --> m =   / ---- κ .
    #  δm     κ²     3    m³              \/   3

    #
    m_roots = 4 / 3**(1 / 4) * kappa
    m_min   = masked_equal(array( [floor(m_roots), ceil(m_roots) ] ), 0)

    #
    K = m_min**2 / kappa**2 + 2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) + 16 / 3 * kappa**2 / m_min**2

    #
    if isclose(direction, 1):
        m = f'{int(m_min[argmin(K) ] ) }'
    else:
        m = f'1'

    #
    if isclose(direction, 1):
        n = f'1'
    else:
        n = f'{int(m_min[argmin(K) ] ) }'

    #
    N = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * K.min()

    #
    return(N, m, n)


# Clamped (single, normal force)
def normal_force_clamped(kappa : float, b : float, D : List[float], direction : int = 1) -> Tuple[float, float, float]:
    #
    #           -----------------------
    #           |     ^    c          |
    #           |     |               |
    # N₁ -----> | c   | b           c | <----- N₁
    #           |     |               |
    #           |     v    c          |
    #           -----------------------
    #

    # The buckling load for an uniaxial, compressive, load in the 1-direction for a clamped laminate is given by (table 6.1 from [1])
    #
    #          π²    _______
    # Nᵪ =  - ---- \/ D₁₁D₂₂ K
    #          b²
    #
    # where the constant K is given by
    #
    #      4      8(D₁₂ + 2D₆₆)
    # K = ---- + --------------- + 4κ²                                  for 0 < κ <= 1.094, and
    #      κ²     3 \/ D₁₁D₂₂
    #
    #      m⁴ + 8m² + 1     2(D₁₂ + 2D₆₆)       κ²
    # K = -------------- + --------------- + --------                   for 1.094 < κ.
    #      κ² (m² + 1)       \/ D₁₁D₂₂        m² + 1
    #
    # The parameter κ (kappa) is defined as (table 6.1 from [1] and the function 'Whitney_method' in Whitney_method.py)
    #           _______
    #        4 /  D₂₂
    # κ = r   /  -----  .
    #       \/    D₁₁
    #
    # The number of half-waves in the 1-direction that minimizes the buckling load can be found by equating the derivative of the constant K
    # with respect to the number of half-waves to zero
    #
    #  δK
    # ---- = 0 ∀ m                          --> m = 1                   for 0 < κ <= 1.094, and
    #  δm
    #                                                   ______________
    #  δK       2m (m⁴ - 2m² - 7 + k⁴)                 /      _______
    # ---- = - ------------------------ = 0 --> m =   / 1 ± \/ 8 - κ⁴   for 1.094 < κ <= 8¹⸍⁴, and
    #  δm            κ² (m² + 1)²                   \/
    #
    #                                       --> m = 1                   for 8¹⸍⁴ < κ.

    #
    m_roots = array( [sqrt(1 - min(sqrt(max(8 - kappa**4, 0) ), 1) ), sqrt(1 + sqrt(max(8 - kappa**4, 0) ) ) ] )
    m_min   = masked_equal(concatenate( (floor(m_roots), ceil(m_roots) ) ), 0)

    #
    with errstate(divide = 'ignore'):
        k = (m_min**4 + 8 * m_min**2 + 1) / (kappa**2 *(m_min**2 + 1) ) + kappa**2 / (m_min**2 + 1)

    #
    K = (kappa <  1.094) * (4 / kappa**2 + 8 * (D[0, 1] + 2 * D[2, 2] ) / (3 * sqrt(D[0, 0] * D[1, 1] ) ) + 4 * kappa**2) \
      + (kappa >= 1.094) * (2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) + k.min() )

    #
    if isclose(direction, 1):
        m = f'{int( (kappa <  1.094) * 1 + (kappa >= 1.094) * m_min[argmin(k) ] ) }'
    else:
        m = f'1'

    #
    if isclose(direction, 1):
        n = f'1'
    else:
        n = f'{int( (kappa <  1.094) * 1 + (kappa >= 1.094) * m_min[argmin(k) ] ) }'

    #
    N = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * K

    #
    return(N, m, n)


# Simplysupportedfree (single, normal force) table and section 6.2
def normal_simplysupported_free(kappa : float, b : float, D : List[float] ) -> Tuple[float, float, float]:
    #
    #           -----------------------
    #           |     ^   free        |
    #           |     |               |
    # N₁ -----> | ss  | b          ss | <----- N₁
    #           |     |               |
    #           |     v    ss         |
    #           -----------------------
    #

    # The constant K (table 6.1 from [2])
    K = 12 / pi**2 * D[2, 2] / sqrt(D[0, 0] * D[1, 1] ) + 1 / kappa**2

    # The uniaxial, in-plane, normal, buckling load in the 1-direction (N₁) (table 6.1 from [2])
    N = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * K

    # The number of half-waves in the 1-direction (m)
    m = f'-'

    # The number of half-waves in the 2-direction (n)
    n = f'-'

    #
    return(N, m, n)


# Simplysupportedfree (single, normal force) table and section 6.2
def normal_simplysupported_free_approximation(kappa : float, b : float, D : List[float], direction : int = 1) -> Tuple[float, float, float]:
    # section 6.2.1 from Kassapoglou [1]

    #
    #           -----------------------
    #           |     ^   free        |
    #           |     |               |
    # N₁ -----> | ss  | b          ss | <----- N₁
    #           |     |               |
    #           |     v    ss         |
    #           -----------------------
    #

    # The buckling load for an uniaxial, compressive, load in the 1-direction for a simply supported & simply supported and free laminate is
    # given by (equation 6.12 from [1])
    #              _                                      _
    #         π²  |                                  λ⁴r⁴  |
    # N₁ = - ---- | D₁₁m² + 2(D₁₂ + 2D₆₆)λ²r² + D₂₂ ------ |.
    #         a²  |_                                  m²  _|
    #
    # The number of half-waves in the 1-direction that minimizes the buckling load can be found by equating the derivative of this load with
    # respect to the number of half-waves to zero
    #               _                    _                     _______
    #  δN₁      π²  |               λ⁴r⁴  |                 4 /  D₂₂
    # ---- = - ---- | 2D₁₁m - 2D₂₂ ------ | = 0 --> m = λr   /  -----  = λκ.
    #  δm       a²  |_                m³ _|                \/    D₁₁

    #
    delta = 5 / 12

    #
    m_min = masked_equal(array( [floor(kappa * delta), ceil(kappa * delta) ] ), 0)

    # The buckling load for an uniaxial, compressive, load in the 1-direction can be rewritten in line with the other equations in table 6.1
    # from [1] to yield
    #
    #         π²    _______
    # Nᵪ = - ---- \/ D₁₁D₂₂ K
    #         b²
    #
    # where the constant K is given by
    #
    #      m²     2(D₁₂ + 2D₆₆)λ²    λ⁴κ²
    # K = ---- + ---------------- + ------ .
    #      κ²       \/ D₁₁D₂₂         m²
    with errstate(divide = 'ignore'):
        K = m_min**2 / kappa**2 + 2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) * delta**2 + kappa**2 / m_min**2 * delta**4

    #
    if isclose(direction, 1):
        m = f'{int(m_min[argmin(K) ] ) }'
    else:
        m = f'5/12'

    #
    if isclose(direction, 1):
        n = f'5/12'
    else:
        n = f'{int(m_min[argmin(K) ] ) }'

    #
    N = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * K.min()

    #
    return(N, m, n)


# Simplysupported (single, bending moment)
def bending_simplysupported(kappa : float, D : List[float], direction : int = 1) -> Tuple[float, float, float]:
    #
    #           -----------------------
    #       ^   |     ^    ss         |   ^
    #      /    |     |               |    \
    #  M₁ (     | ss  | b          ss |     ) M₁
    #      \    |     |               |    /
    #       \   |     v    ss         |   /
    #           -----------------------
    #

    # The buckling load for a uniaxial, moment in the 1-direction/along the 2-direction of a simply supported laminate is given by (table 6.1
    # from [1])
    #              _______
    # M₁ =  - π² \/ D₁₁D₂₂ K
    #
    # where the constant K is given by
    #               ______________________________________________________________________
    #              / | m²     2(D₁₂ + 2D₆₆)     κ²  | | m²     8(D₁₂ + 2D₆₆)        κ²  |
    # K = 0.047π² /  |---- + --------------- + ---- | |---- + --------------- + 16 ---- |  .
    #           \/   |_κ²      \/ D₁₁D₂₂        m² _| |_κ²      \/ D₁₁D₂₂           m² _|
    #
    # The parameter κ (kappa) is defined as (table 6.1 from [1] and the function 'Whitney_method' in Whitney_method.py)
    #           _______
    #        4 /  D₂₂
    # κ = r   /  -----  .
    #       \/    D₁₁
    #
    # The number of half-waves in the 1-direction that minimizes the buckling load can be found by equating the derivative of the constant K
    # with respect to the number of half-waves to zero
    #                                                                                                 ___________________
    #  δK                (2κ² - m²) (2κ² + m²) (5dκ²m² + 8κ⁴ + 2m⁴)               ___          κ     /  ___________
    # ---- = - 0.047²π⁴ ------------------------------------------- = 0 --> m = \/ 2  κ ∨ m = ---   / \/ 25d² - 64  - 5d
    #  δm                                  κ⁴m⁵K                                               2  \/
    #
    # where the stiffness d is given by
    #
    #      2(D₁₂ + 2D₆₆)
    # d = --------------- .
    #       \/ D₁₁D₂₂

    #
    d = 2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] )

    #
    m_roots    = array( [sqrt(2) * kappa, kappa / 2 * sqrt(sqrt(25 * d**2 - 64) + 5 * d) ] )
    m_filtered = m_roots[isclose(m_roots.imag, 0) ]
    m_min      = masked_equal(concatenate( (floor(m_filtered), ceil(m_filtered) ) ), 0)

    #
    with errstate(divide = 'ignore'):
        K = 0.047 * pi**2 * sqrt( (m_min**2 / kappa**2 +     d +      kappa**2 / m_min**2) \
                                * (m_min**2 / kappa**2 + 4 * d + 16 * kappa**2 / m_min**2) )

    #
    if isclose(direction, 1):
        m = f'{int(m_min[argmin(K) ] ) }'
    else:
        m = f'1'

    #
    if isclose(direction, 1):
        n = f'1'
    else:
        n = f'{int(m_min[argmin(K) ] ) }'

    #
    M = pi**2 * sqrt(D[0, 0] * D[1, 1] ) * K.min()

    #
    return(M, m, n)


# Simply supported square (shear; a/b = 1; 0 < beta <= 1)
def shear_simplysupported_square(D : List[float], a : float, b : float) -> Tuple[float, float, float]:
    #                       __________\ N₁₂
    #           ----------------------- |\
    #           |     ^    ss         | |
    #           |     |               | |
    #         | | ss  | a = b      ss | |
    #         | |     |               |
    #         | |     v    ss         |
    #        \| -----------------------
    #       N₁₂ ___________
    #           \

    #
    A    = - 0.27 + 0.185 *   (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] )
    B    =   0.82 + 0.46  *   (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) \
           - 0.2          * ( (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) )**2

    #
    beta = (min(D[0, 0], D[1, 1] ) / max(D[0, 0], D[1, 1] ) )**(1 / 4)

    #
    K    = 8.2 + 5 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) * 10**(- (A / beta + B * beta) )

    #
    m = f'-'

    #
    n = f'-'

    #
    N_xy = 4 / (b if D[1, 1] >= D[0, 0] else a)**2 * (min(D[0, 0], D[1, 1]) * max(D[0, 0], D[1, 1])**3)**(1 / 4) * K

    #
    return(N_xy, m, n)


# Simply supported rectangle (shear; 0.5 <= a/b < 1)
def shear_simplysupported_rectangle(D : List[float], a : float, b : float) -> Tuple[float, float, float]:
    #                       __________\ N₁₂
    #           ----------------------- |\
    #           |     ^    ss         | |
    #           |     |               | |
    #         | | ss  | b          ss | |
    #         | |     |               |
    #         | |     v    ss         |
    #        \| -----------------------
    #       N₁₂ ___________
    #           \

    #
    D1 = (b > a) * (D[0, 0] + D[1, 1] * (a / b)**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * (a / b)**2) \
       + (b < a) * (D[1, 1] + D[0, 0] * (b / a)**4 + 2 * (D[0, 1] + 2 * D[2, 2] ) * (b / a)**2)

    #
    D2 = (b > a) * (D[0, 0] + 81 * D[1, 1] * (a / b)**4 + 18 * (D[0, 1] + 2 * D[2, 2] ) * (a / b)**2) \
       + (b < a) * (D[1, 1] + 81 * D[0, 0] * (b / a)**4 + 18 * (D[0, 1] + 2 * D[2, 2] ) * (b / a)**2)

    #
    D3 = (b > a) * (81 * D[0, 0] + D[1, 1] * (a / b)**4 + 18 * (D[0, 1] + 2 * D[2, 2] ) * (a / b)**2) \
       + (b < a) * (81 * D[1, 1] + D[0, 0] * (b / a)**4 + 18 * (D[0, 1] + 2 * D[2, 2] ) * (b / a)**2)

    #
    m = f'-'

    #
    n = f'-'

    #
    N_xy = (pi**4 * (b / a**3 if b > a else a / b**3) ) / sqrt(14.28 / D1**2 + 40.96 / (D1 * D2) + 40.96 / (D1 * D3) )

    #
    return(N_xy, m, n)


# Simply supported infinite (shear; a / b = 0) section 6.4
def shear_simplysupported_infinite(D : List[float], a : float, b : float) -> Tuple[float, float, float, float, float]:
    #                                              __________\ N₁₂
    #           ---------------------------------------------- |\
    #           |        ^            ss                     | |
    #           |        |                                   | |
    #         | | ss     | a, b -> ∞                      ss | |
    #         | |        |                                   |
    #         | |        v            ss                     |
    #        \| ----------------------------------------------
    #       N₁₂ ___________
    #           \
    kappa = (a <= b) * (D[1, 1] / D[0, 0] )**(1 / 2) \
          + (a >  b) * (D[0, 0] / D[1, 1] )**(1 / 2)

    #
    def tana(x, D, kappa):
        # Equation 6.29 and 6.30 combined
        y = arctan( (x**4 + 2 * kappa * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) * x**2 + kappa**2)**(- 1) \
          * (    3 * x**4 + 2 * kappa * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) * x**2 - kappa**2) \
          + (        x**4 + 2 * kappa * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) * x**2 + kappa**2)**(- 1 / 2) \
          * (6 * x**2 - 2 * (D[0, 1] + 2 * D[2, 2] ) ) )
        return(y)

    # Equations 6.29 and 6.30 combined
    tan_alpha = tan(fsolve(tana, 0, args = (D, kappa) )[0] )

    # Equation 6.29
    AR = (tan_alpha**4 + 2 * kappa * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) * tan_alpha**2 + kappa**2)**(- 1 / 4)

    #
    m = f'-'

    #
    n = f'-'

    # Equation 6.28
    N_xy = pi**2 * sqrt(D[0, 0] * D[1, 1] ) / (2 * AR**2 * (a if a <= b else b)**2 * tan_alpha) * (1 / kappa \
         * (1 + 6 * tan_alpha**2 * AR**2 + tan_alpha**4 * AR**4) + 2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) \
         * (AR**2 + AR**4 * tan_alpha**2) + kappa * AR**4)

    #
    return(N_xy, m, n, tan_alpha, AR)


def shear_simplysupported(D : List[float], a : float, b : float) -> Tuple[float, float, float]:
    #                       __________\ N₁₂
    #           ----------------------- |\
    #           |         ss          | |
    #           |                     | |
    #         | | ss               ss | |
    #         | |                     |
    #         | |         ss          |
    #        \| -----------------------
    #       N₁₂ ___________
    #           \

    # Aspect ratio
    r = min(a, b) / max(a, b)

    #
    if isclose(a, b):
        N_xy = shear_simplysupported_square(D, a, b)[0]
    elif (1 / 2 <= r) & (r < 1):
        N_xy = shear_simplysupported_rectangle(D, a, b)[0]
    else:
        N_xy = (1 - 2 * r) * shear_simplysupported_infinite(D, a, b)[0] + 2 * r * shear_simplysupported_rectangle(D, a, b)[0]

    #
    m = f'-'

    #
    n = f'-'

    #
    return(N_xy, m, n)


# The function 'normal_shear_simplysupported_approximation'
def normal_shear_simplysupported_approximation(number_of_data_points : float, a : float, b : float, D : List[float],
                                               kappa : float, theta : List[float] ) -> Tuple[List[float], List[float] ]:
    # The biaxial loading ratio (k) is defined as (equation 5.71 from [1] and page 121 from [2])
    #
    #      N₁₂
    # k = ---- .
    #      N
    #
    # This ratio as a function of the angle θ in the NN₁₂-plane (see figure X.X) is thus given by
    #
    #          1
    # k = - ------- .
    #        tan θ

    # An array of shape-(number_of_data_points) containing the biaxial loading ratio (k) taking into account the singularity at θ = 0, θ = 90°,
    # and θ = 180°
    k = ( (theta != 0) & (theta != pi / 2) & (theta != pi) ) * - 1 / (tan(theta) \
      + ( (theta == 0) | (theta == pi / 2) | (theta == pi) ) * finfo(float).eps)

    # An array of shape-(number_of_data_points, 2) containing the variable component K of the in-plane, normal, buckling load N (equation 6.34
    # from [1]) for each biaxial loading ratio k, and thus angle θ, under consideration
    K       = zeros( [number_of_data_points, 2] )
    K[:, 0] = (5 - sqrt(9 + 65536 / 81 * a**2 / b**2 * k**2 / pi**4) ) / (2 - 8192 / 81 * a**2 / b**2 * k**2 / pi**4)
    K[:, 1] = (5 + sqrt(9 + 65536 / 81 * a**2 / b**2 * k**2 / pi**4) ) / (2 - 8192 / 81 * a**2 / b**2 * k**2 / pi**4)

    # An array of shape-(number_of_data_points) containing the in-plane, normal, buckling load (N) (equation 6.34 from [1]) for each biaxial
    # loading ratio (k), and thus angle θ, under consideration
    N_x = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * (1 / kappa**2 + 2 * (D[0, 1] + 2 * D[2, 2] ) / sqrt(D[0, 0] * D[1, 1] ) + kappa**2) \
        * take_along_axis(K, expand_dims(argmin(abs(K), axis = 1), axis = -1), axis = -1).flatten()

    # Correct the values of the in-plane, normal, buckling load at the boundaries of sections (N) (θ = 90° and θ = 270°) due to the presence of
    # singularities (equations 6.34 and 6.35 from [1])
    N_x[theta == 0]      = 0
    N_x[theta == pi / 2] = - pi**2 / b**2 * sqrt(D[0, 0] * D[1, 1] ) * (1 / kappa**2 + 2 * (D[0, 1] + 2 * D[2, 2] ) \
                         / sqrt(D[0, 0] * D[1, 1] ) + kappa**2)
    N_x[theta == pi]     = 0

    # An array of shape-(number_of_data_points) containing the in-plane, shear, buckling load (N₁₂) (page 134 from [1]) for each biaxial
    # loading ratio (k), and thus angle θ, under consideration
    N_xy = k * N_x

    # Correct the values of the in-plane, normal, buckling load at the boundaries of sections (N₁₂) (θ = 90° and θ = 270°) due to the presence
    # of singularities (equations 6.34 and 6.37 from [1])
    N_xy[theta == 0]      = 9 * pi**4 * sqrt(D[0, 0] * D[1, 1] ) / (32 * a * b) * (1 / kappa**2 + 2 * (D[0, 1] + 2 * D[2, 2] ) \
                          / sqrt(D[0, 0] * D[1, 1] ) + kappa**2)
    N_xy[theta == pi / 2] = 0
    N_xy[theta == pi]     = - N_xy[theta == 0]

    # End the function and return the arrays of shape-(number_of_data_points) containing the biaxial, in-plane, normal (N) and shear buckling
    # load (N₁₂) for a simply supported, rectangular, specially orthotropic laminate for each value of the angle θ under consideration
    return(N_x, N_xy)


# The function 'normal_moment_interaction_curve'
def normal_moment_interaction_curve(x, N, M, theta):
    # Return the residual of the interaction curve for an in-plane, normal load (N) and in-plane, moment (M) (table 6.1 from [2])
    return(x**1.76 - x * ( (theta < pi / 2) * M - (theta > pi / 2) * M) * tan(theta) / N - 1)


# The function 'Kassapoglou_survey'
def Kassapoglou_survey(geometry : Dict[str, float], settings : Dict[str, str or float], stiffness : Dict[str, List[float] ],
                       data : bool = False, illustrations : bool = False) -> Tuple[List[float], List[float], List[float] ]:
    """[summary]

    Table KS.1: Table
     #    Boundary conditions                                               Applied load   Unit
    ---  ----------------------------------------------------------------- -------------- ------
    0     Simply supported                                                      Nᵪ          N/m
    1     Simply supported                                                      Nᵧ          N/m
    2     Simply supported                                                      Nᵪᵧ         N/m
    3     Simply supported                                                      Mᵪ          Nm
    4     Simply supported                                                      Mᵧ          Nm
    5     Simply supported (x = 0,a; y = 0); free (y = b)                       Nᵪ          N/m
    6     Simply supported (x = 0,a; y = 0); free (y = b) [approximation]       Nᵪ          N/m
    7     Simply supported (x = 0; y = 0,b); free (x = a)                       Nᵧ          N/m
    8     Simply supported (x = 0; y = 0,b); free (x = a) [approximation]       Nᵧ          N/m
    9     Simply supported (x = 0,a); clamped (y = 0,b)                         Nᵪ          N/m
    10    Simply supported (x = 0,a); clamped (y = 0,b)                         Nᵧ          N/m
    11    Simply supported (y = 0,b); clamped (x = 0,a)                         Nᵪ          N/m
    12    Simply supported (y = 0,b); clamped (x = 0,a)                         Nᵧ          N/m
    13    Clamped                                                               Nᵪ          N/m
    14    Clamped                                                               Nᵧ          N/m

    Parameters
    ----------
    geometry : Dict[str, float]
        [description]
    settings : Dict[str, str or float]
        [description]
    stiffness : Dict[str, List[float] ]
        [description]

    Returns
    -------
    Tuple[List[float], List[float], List[float] ]
        [description]
    """


    # Kassapoglou [2]

    # The plate dimension (width) in the x-direction
    a = geometry['a']
    # The plate dimension (length) in the y-direction
    b = geometry['b']

    #
    number_of_data_points = settings['number_of_data_points']
    #
    number_of_elements_x = settings['number_of_elements_x']
    #
    number_of_elements_y = settings['number_of_elements_y']
    # The format of the files as which the generated illustrations are saved
    fileformat = settings['fileformat']
    # The resolution of the generated illustrations in dots-per-inch
    resolution = settings['resolution']
    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    #
    boundary_conditions = [f'Simply supported',
                           f'Simply supported',
                           f'Simply supported',
                           f'Simply supported',
                           f'Simply supported',
                           f'Simply supported (x = 0,a; y = 0); free (y = b)',
                           f'Simply supported (x = 0,a; y = 0); free (y = b) [approximation]',
                           f'Simply supported (x = 0; y = 0,b); free (x = a)',
                           f'Simply supported (x = 0; y = 0,b); free (x = a) [approximation]',
                           f'Simply supported (x = 0,a); clamped (y = 0,b)',
                           f'Simply supported (x = 0,a); clamped (y = 0,b)',
                           f'Simply supported (y = 0,b); clamped (x = 0,a)',
                           f'Simply supported (y = 0,b); clamped (x = 0,a)',
                           f'Clamped',
                           f'Clamped']

    #
    applied_force = [f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A\u1D67 [N/m]',
                     f'M\u1D6A  [Nm]',
                     f'M\u1D67  [Nm]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]']

    #
    kappa_x = a / b * (D[1, 1] / D[0, 0] )**(1 / 4)
    kappa_y = b / a * (D[0, 0] / D[1, 1] )**(1 / 4)

    #
    Nmn = [normal_force_simplysupported(kappa_x, b, D, direction = 1),
           normal_force_simplysupported(kappa_y, a, D, direction = 2),
           shear_simplysupported(D, a, b),
           bending_simplysupported(kappa_x, D),
           bending_simplysupported(kappa_y, D),
           normal_simplysupported_free(kappa_x, b, D),
           normal_simplysupported_free_approximation(kappa_x, b, D, direction = 1),
           normal_simplysupported_free(kappa_y, a, D),
           normal_simplysupported_free_approximation(kappa_y, a, D, direction = 2),
           normal_force_simplysupported_clamped(kappa_x, b, D, direction = 1),
           normal_force_clamped_simplysupported(kappa_y, a, D, direction = 2),
           normal_force_clamped_simplysupported(kappa_x, b, D, direction = 1),
           normal_force_simplysupported_clamped(kappa_y, a, D, direction = 2),
           normal_force_clamped(kappa_x, b, D, direction = 1),
           normal_force_clamped(kappa_y, a, D, direction = 2) ]

    N_Kassapoglou = array( [load[0] for load in Nmn] )
    m_Kassapoglou = [m[1] for m in Nmn]
    n_Kassapoglou = [n[2] for n in Nmn]

    N_table = [None] * len(N_Kassapoglou)
    for i in range(len(N_Kassapoglou) ):
        N_table[i] = f'{N_Kassapoglou[i]:.5e}'

    for i in [2, 3, 4]:
        N_table[i] = f'\u00B1{N_Kassapoglou[i]:.5e}'

    #
    delta = 5 / 12

    #
    x, y = meshgrid(linspace(0, 1, number_of_elements_x), linspace(0, 1, number_of_elements_y) )

    #
    w_x = sin(float(m_Kassapoglou[0] ) * pi * x) * sin(float(n_Kassapoglou[0] ) * pi * y)

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    #
    plt.imshow(w_x, cmap='PuBu_r', extent=[0, a, 0, b], interpolation='bilinear', aspect = 'equal', origin = 'lower')

    # Format the xlabel
    plt.xlabel(f'x [m]')

    # Format the ylabel
    plt.ylabel(f'y [m]')

    # Add text with the moniker of the simulation name, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

    # Save the figure
    plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/Deformation_Nx.{fileformat}',
                dpi = resolution, bbox_inches = "tight")

    # Close the figure
    plt.close()

    #
    w_y = sin(float(m_Kassapoglou[1] ) * pi * x) * sin(float(n_Kassapoglou[1] ) * pi * y)

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    #
    plt.imshow(w_y, cmap='PuBu_r', extent=[0, a, 0, b], interpolation='bilinear', aspect = 'equal', origin = 'lower')

    # Format the xlabel
    plt.xlabel(f'x [m]')

    # Format the ylabel
    plt.ylabel(f'y [m]')

    # Add text with the moniker of the simulation name, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

    # Save the figure
    plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/Deformation_Ny.{fileformat}',
                dpi = resolution, bbox_inches = "tight")

    # Close the figure
    plt.close()

    #
    w_x = sin(float(m_Kassapoglou[6] ) * pi * x) * sin(delta * pi * y)

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    #
    plt.imshow(w_x, cmap='PuBu_r', extent=[0, a, 0, b], interpolation='bilinear', aspect = 'equal', origin = 'lower')

    # Format the xlabel
    plt.xlabel(f'x [m]')

    # Format the ylabel
    plt.ylabel(f'y [m]')

    # Add text with the moniker of the simulation name, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

    # Save the figure
    plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/Deformation_Nx_free.{fileformat}',
                dpi = resolution, bbox_inches = "tight")

    # Close the figure
    plt.close()

    #
    w_y = sin(delta * pi * x) * sin(float(n_Kassapoglou[8] ) * pi * y)

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    #
    plt.imshow(w_y, cmap='PuBu_r', extent=[0, a, 0, b], interpolation='bilinear', aspect = 'equal', origin = 'lower')

    # Format the xlabel
    plt.xlabel(f'x [m]')

    # Format the ylabel
    plt.ylabel(f'y [m]')

    # Add text with the moniker of the simulation name, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

    # Save the figure
    plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/Deformation_Ny_free.{fileformat}',
                dpi = resolution, bbox_inches = "tight")

    # Close the figure
    plt.close()

    #
    S = shear_simplysupported_infinite(D, a, b)

    # Create and open the figure
    plt.figure()

    # Make the figure full screen
    fig = plt.get_current_fig_manager()
    fig.full_screen_toggle()

    #
    x, y = meshgrid(linspace(0, 1 / S[4] + S[3], number_of_elements_x), linspace(0, 1, number_of_elements_y) )

    #
    w = sin(pi * y) * sin(pi * S[4] * (x - y * S[3] ) )

    #
    if a <= b:
        #
        plt.imshow(w, cmap='PuBu_r', extent=[0, 1 / S[4] + S[3], 0, a], interpolation='bilinear', aspect = 'equal', origin = 'lower')

        # Format the xlabel
        plt.xlabel(f'y [m]')

        # Format the ylabel
        plt.ylabel(f'x [m]')
    else:
        #
        plt.imshow(w, cmap='PuBu_r', extent=[0, 1 / S[4] + S[3], 0, b], interpolation='bilinear', aspect = 'equal', origin = 'lower')

        # Format the xlabel
        plt.xlabel(f'x [m]')

        # Format the ylabel
        plt.ylabel(f'y [m]')

    # Add text with the moniker of the simulation name, the source and a timestamp
    plt.text(0, -0.125, f'Simulation: {simulation}',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
    plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
            fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

    # Save the figure
    plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/Deformation_Nxy.{fileformat}',
                dpi = resolution, bbox_inches = "tight")

    # Close the figure
    plt.close()

    #
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/stability_analysis/Kassapoglou_survey.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'The stresses in the local x-y coordinate system at the top and bottom of each ply from the top to the bottom '
                                f'of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
            textfile.write(tabulate(array( [boundary_conditions, applied_force, N_table, m_Kassapoglou, n_Kassapoglou] ).T.tolist(),
                        headers  = [f'Boundary conditions', f'Applied force', f'Buckling load', f'm [-]', f'n [-]'],
                        colalign = ('left', 'center', 'center', 'center', 'center'),
                        floatfmt = ('.0f', '.0f', '.5e', '.2f', '.2f') ) )

    #
    if illustrations:
        # Use the Tableau 30 colors (consisting of Tableau 10, Tableau 10 Medium, and Tableau 10 Light from [3]) for the figures
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

        # The plane defined by the two loads under consideration (N₁ and N₂), which contains the interaction curve dictating buckling, is
        # depicted in figure 1 below. In addition, the angle θ between both loads is specified.
        #
        #                         N₁
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
        # Figure KS.1: Illustration of the N₁N₂-plane and the angle
        #                θ between the two loads.
        #
        # From this figure can be derived that the angle θ is related to both loads via
        #
        #            N₁
        # tan θ = - ----
        #            N₂
        #
        # resulting in
        #
        # N₁ = - N₂ ⋅ tan θ .
        #
        # It can furthermore be seen that
        #          ___________
        # N₁ =   \/ N₁² + N₂²  ⋅ cos θ  -->  N₁² = (N₁² + N₂²) ⋅ cos² θ
        #
        # which can be rewritten to yield
        #
        #               ______________
        #              /  1 - cos² θ
        # N₂ = ± N₁   / --------------
        #           \/      cos² θ
        #
        # where the sign is dependent on the section of the N₁N₂-plane
        #
        #   0° < θ < 180° --> - ,
        # 180° < θ < 360° --> + .

        #
        #                       __________\ N₁₂
        #           ----------------------- |\
        #      ^    |     ^    ss         | |  ^
        #     /     |     |               | |   \
        # M₁ (    | | ss  | b          ss | |    ) M₁
        #     \   | |     |               |     /
        #         | |     v    ss         |
        #        \| -----------------------
        #       N₁₂ ___________
        #           \

        # The interaction curve describing buckling due to an in-plane shear force and in-plane moment is given by (table 6.1 from [2])
        #
        #  M²      N₁₂²
        # ----- + ------ = 1 .
        #  M₀²     N₁₂₀²
        #
        # Combining this expression with the relationship between N₁, N₂, and θ yields
        #          _                            _ -1
        #         |   1    1 - cos² θ       1    |
        # N₁₂ = ± | ----- ------------ + ------- |
        #         |_ M₀²     cos² θ       N₁₂₀² _|
        #
        # where the sign is dependent on the section of the N₁₂M-plane
        #
        #   0° < θ <  90° --> + ,
        #  90° < θ < 270° --> - ,
        # 270° < θ < 360° --> + .

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the MN₁₂-plane in which the
        # interaction curve is bounded (0° <= θ <= 360°)
        theta = 2 * pi / 360 * linspace(0, 360 - 360 / number_of_data_points, num = number_of_data_points)

        # An array of shape-(number_of_data_points) containing the sign of the in-plane, shear, buckling load (N₁₂)
        sign = ones(number_of_data_points)
        sign[ (theta > pi / 2) & (theta < 3 * pi / 2) ] = - 1

        # An array of shape-(number_of_data_points, 2) containing the in-plane, shear, buckling load (N₁₂) for a moment in the 1 and 2-direction
        # (M₁ and M₂) respectively
        N_xy = einsum('i,ij->ij', sign,
            sqrt(1 / (einsum('j,i->ij', 1 / N_Kassapoglou[3:5]**2, (1 - cos(theta)**2) / cos(theta)**2) + 1 / N_Kassapoglou[2]**2) ) )

        # An array of shape-(number_of_data_points, 2) containing the in-plane, buckling moment in the 1 and 2-direction (M₁ and M₂ respectively)
        # for the in-plane, shear, buckling load (N₁₂)
        M = einsum('ij,i->ij', - N_xy, tan(theta) )

        # Correct the values of the in-plane, buckling moment at the boundary of sections (θ = 90° and θ = 270°) due to the presence of
        # singularities
        M[theta ==     pi / 2] = - N_Kassapoglou[3:5]
        M[theta == 3 * pi / 2] =   N_Kassapoglou[3:5]

        # Append the arrays containing the in-plane, shear, buckling load (N₁₂) and the in-plane, buckling moment in the 1 and 2-direction (M₁
        # and M₂ respectively) with the values at θ = 0° to close the envelope describing the buckling load at θ = 360°
        N_xy = append(N_xy, N_xy[0, :].reshape(1, -1), axis = 0)
        M    = append(   M,    M[0, :].reshape(1, -1), axis = 0)

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        # Add the variation of the in-plane, buckling moment in the 1-direction (M₁)
        plt.plot(M[:, 0], N_xy[:, 0], color = Colors[0], linewidth = 1, linestyle = LineStyles[0], label = f'M\u2081')

        # Add the variation of the in-plane, buckling moment in the 2-direction (M₂)
        plt.plot(M[:, 1], N_xy[:, 1], color = Colors[3], linewidth = 1, linestyle = LineStyles[1], label = f'M\u2082')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Add a vertical line through the origin
        plt.axvline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'M [Nm]')

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'N\u2081\u2082 [N/m]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/M_vs_N12.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()

        #                       __________\ N₁₂
        #           ----------------------- |\
        #           |     ^    ss         | |
        #           |     |               | |
        # N₁ ---> | | ss  | b          ss | | <--- N₁
        #         | |     |               |
        #         | |     v    ss         |
        #        \| -----------------------
        #       N₁₂ ___________
        #           \

        # The interaction curve describing buckling due to an in-plane, shear force and in-plane, normal force is given by (table 6.1 from [2])
        #
        #  N₁₂²     N
        # ------ + ---- = 1 .
        #  N₁₂₀²    N₀
        #
        # Combining this expression with the relationship between N₁, N₂, and θ yields
        #                    _______________
        #  N₁₂²     N₁₂     /  1 - cos² θ
        # ------ ± -----   / --------------  - 1 = 0 --> a₀ ⋅ N₁₂² ± b₀ ⋅ N₁₂ - 1 = 0
        #  N₁₂₀²    N₀   \/      cos² θ
        #
        # where the sign is dependent on the section of the NN₁₂-plane
        #
        #   0° < θ <  90° --> + ,
        #  90° < θ < 180° --> - .

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NN₁₂-plane in which the
        # interaction curve is bounded (0° <= θ <= 180°)
        theta = pi / 180 * linspace(0, 180, num = number_of_data_points // 2)

        # An array of shape-(number_of_data_points) containing the sign of the in-plane, shear, buckling load (N₁₂)
        sign = ones(number_of_data_points // 2)
        sign[pi / 2 <  theta] = - 1

        # The components of the previously derived quadratic equation
        a_0 = 1 / N_Kassapoglou[2]**2
        b_0 = einsum('j,i->ij', - 1 / N_Kassapoglou[0:2], sqrt(1 - cos(theta)**2) / cos(theta) )

        # An array of shape-(number_of_data_points, 2) containing the in-plane, shear, buckling load (N₁₂) for an in-plane, normal force in the
        # 1 and 2-direction (N₁ and N₂) respectively
        N_xy = (- b_0 + einsum('i,ij->ij', sign, sqrt(b_0**2 + 4 * a_0) ) ) / (2 * a_0)

        # An array of shape-(number_of_data_points, 2) containing the in-plane, normal force in the 1 and 2-direction (N₁ and N₂ respectively)
        # for the in-plane, shear, buckling load (N₁₂)
        N = einsum('ij,i->ij', - N_xy, tan(theta) )

        # Correct the values of the in-plane, normal force in the 1 and 2-direction at the intersection of the sections under consideration
        # (θ = 90°) due to the presence of a singularity
        N[theta == pi / 2] = N_Kassapoglou[0:2]

        # Inititate an arrays of shape-(number_of_data_points // 2, 2) containing the in-plane, normal, buckling, force in the 1 and
        # 2-direction (N₁ and N₂) respectively
        N_approximation    = zeros( (number_of_data_points // 2, 2) )

        # Inititate an arrays of shape-(number_of_data_points // 2, 2) containing the corresponding in-plane, shear, buckling, load in the
        # 12-plane (N₁₂)
        N_xy_approximation = zeros( (number_of_data_points // 2, 2) )

        # Determine the in-plane, normal, buckling, force in the 1-direction (N₁) and the corrsponding in-plane, shear, buckling, load in the
        # 12-plane (N₁₂)
        N_approximation[:, 0], N_xy_approximation[:, 0] = \
            normal_shear_simplysupported_approximation(number_of_data_points // 2, a, b, D, kappa_x, theta)

        # Determine the in-plane, normal, buckling, force in the 2-direction (N₂) and the corrsponding in-plane, shear, buckling, load in the
        # 12-plane (N₁₂)
        N_approximation[:, 1], N_xy_approximation[:, 1] = \
            normal_shear_simplysupported_approximation(number_of_data_points // 2, b, a, D, kappa_y, theta)

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        # Add the variation of the in-plane, normal, buckling load in the 1-direction (N₁) according to the interaction curve
        plt.plot(N[:, 0], N_xy[:, 0], color = Colors[0], linewidth = 1, linestyle = LineStyles[0], label = f'Interaction curve (N\u2081)')

        # Add the variation of the in-plane, normal, buckling load in the 2-direction (N₂) according to the interaction curve
        plt.plot(N[:, 1], N_xy[:, 1], color = Colors[3], linewidth = 1, linestyle = LineStyles[0], label = f'Interaction curve (N\u2082)')

        # Add the variation of the in-plane, normal, buckling load in the 1-direction (N₁) according to the approximation (section 6.5 from [2])
        plt.plot(N_approximation[:, 0], N_xy_approximation[:, 0], color = Colors[0], linewidth = 1, linestyle = LineStyles[1],
            label = f'Approximation (N\u2081)')

        # Add the variation of the in-plane, normal, buckling load in the 2-direction (N₂) according to the approximation (section 6.5 from [2])
        plt.plot(N_approximation[:, 1], N_xy_approximation[:, 1], color = Colors[3], linewidth = 1, linestyle = LineStyles[1],
            label = f'Approximation (N\u2082)')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'N [N/m]')

        # Set the upper limit of the x-axis
        plt.xlim(right = 0)

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'N\u2081\u2082 [N/m]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/N_vs_N12.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()

        #
        #           -----------------------
        #        ^  |     ^    ss         |  ^
        #       /   |     |               |   \
        # N₁ -----> | ss  | b          ss | <----- N₁
        #       \   |     |               |   /
        #     M₁ \  |     v    ss         |  / M₁
        #           -----------------------
        #

        # The interaction curve describing buckling due to an in-plane moment and in-plane, normal force is given by (table 6.1 from [2])
        #  _    _ 1.76
        # |  M   |      N
        # | ---- |  +  ---- = 1 .
        # |_ M₀ _|      N₀
        #
        # Combining this expression with the relationship between N₁, N₂, and θ yields
        #  _    _ 1.76                     _    _ 1.76
        # |  M   |       M tan θ          |  M   |      M M₀ tan θ               1.76    M₀ tan θ
        # | ---- |  -  ---------- = 1 --> | ---- |  -  ------------ - 1 = 0 --> x   -   ---------- x - 1 = 0
        # |_ M₀ _|         N₀             |_ M₀ _|        M₀ N₀                             N₀
        #
        # where x = M / M₀ .

        # An array of shape-(number_of_data_points) containing the angle θ in radians for the section of the NM-plane in which the
        # interaction curve is bounded (0° <= θ <= 180°)
        theta = pi / 180 * linspace(0, 180, num = number_of_data_points // 2)

        # Initialize an array of shape-(number_of_data_points / 2) containing the roots of the previously derived equation describing the
        # interaction curve for an in-plane moment and an in-plane, normal force in the 1 and 2-direction respectively
        x = zeros( (number_of_data_points // 2, 2) )

        # An array of shape-(2) containing the initial estimate for the root of the interaction curve for the 1 and 2-direction at θ = 0°
        x0 = ones(2)

        # For all angles θⁱ under consideration
        for i in range(number_of_data_points // 2):

            # If θ > 0°
            if i != 0:
                # Set the estimate for the roots of the derived interaction curve for the 1 and 2-direction for the current angle θⁱ equal to
                # the roots of the previous angle θⁱ⁻¹
                x0[:] = x[i - 1, :]

            # Determine the root of the derived interaction curve for the 1-direction for the current angle θⁱ
            x[i, 0] = fsolve(normal_moment_interaction_curve, x0[0], args=(N_Kassapoglou[0], N_Kassapoglou[3], theta[i] ) )

            # Determine the root of the derived interaction curve for the 2-direction for the current angle θⁱ
            x[i, 1] = fsolve(normal_moment_interaction_curve, x0[1], args=(N_Kassapoglou[1], N_Kassapoglou[4], theta[i] ) )

        # Initialize an array of shape-(number_of_data_points / 2, 2) containing the in-plane, moment in the 1 and 2-direction (M₁ and M₂
        # respectively)
        M = zeros( (number_of_data_points // 2, 2) )

        # An array of shape-(number_of_data_points, 2) containing the in-plane moment in the 1 and 2-direction (M₁ and M₂ respectively)
        M[:, 0] = ( (theta < pi / 2) * N_Kassapoglou[3] - (theta > pi / 2) * N_Kassapoglou[3] ) * x[:, 0]
        M[:, 1] = ( (theta < pi / 2) * N_Kassapoglou[4] - (theta > pi / 2) * N_Kassapoglou[4] ) * x[:, 1]

        # An array of shape-(number_of_data_points, 2) containing the in-plane, normal force in the 1 and 2-direction (N₁ and N₂ respectively)
        # for the corresponding in-plane, moment (M₁ and M₂ respectively)
        N = einsum('ij,i->ij', - M, tan(theta) )

        # Correct the values of the in-plane, normal force in the 1 and 2-direction (N₁ and N₂ respectively) at the intersection of the
        # sections under consideration (θ = 90°) due to the presence of a singularity
        N[theta == pi / 2] = N_Kassapoglou[0:2]

        # Create and open the figure
        plt.figure()

        # Make the figure full screen
        fig = plt.get_current_fig_manager()
        fig.full_screen_toggle()

        # Add the variation of the in-plane, normal, buckling load in the 1-direction (N₁) according to the interaction curve
        plt.plot(N[:, 0], M[:, 0], color = Colors[0], linewidth = 1, label = f'1-direction')

        # Add the variation of the in-plane, normal, buckling load in the 2-direction (N₂) according to the interaction curve
        plt.plot(N[:, 1], M[:, 1], color = Colors[3], linewidth = 1, label = f'2-direction')

        # Add a horizontal line through the origin
        plt.axhline(color = "black", linewidth = 0.5, linestyle = (0, (5, 10) ) )

        # Format the xlabel
        plt.xlabel(f'N [N/m]')

        # Set the upper limit of the x-axis
        plt.xlim(right = 0)

        # Format the xticks
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0) )

        # Format the ylabel
        plt.ylabel(f'M [Nm]')

        # Format the yticks
        plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0) )

        # Add a legend
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fancybox = True, edgecolor = 'inherit')

        # Add text with the moniker of the simulation name, the source and a timestamp
        plt.text(0, -0.125, f'Simulation: {simulation}',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)
        plt.text(0, -0.150, f'Source: stability_analysis/Kassapoglou_survey.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")})',
                fontsize = 6, horizontalalignment = 'left', verticalalignment = 'center', transform = plt.gca().transAxes)

        # Save the figure
        plt.savefig(f'{simulation}/illustrations/stability_analysis/Kassapoglou_survey/N_vs_M.{fileformat}',
                    dpi = resolution, bbox_inches = "tight")

        # Close the figure
        plt.close()

    #
    return(N_Kassapoglou, m_Kassapoglou, n_Kassapoglou)
