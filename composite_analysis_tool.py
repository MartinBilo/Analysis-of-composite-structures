# composite_analysis_tool
#
#
#
#
#

# Impormpackages into library
from numpy   import array
from os      import mkdir
from os.path import isdir

#
force            = {}
force['delta_C'] = 0 # [2]
force['delta_T'] = 0
force['xi']      = 0.125
force['eta']     = 0.125
force['N_x']     = 1000
force['N_y']     = 1000
force['N_xy']    = 0
force['M_x']     = 0
force['M_y']     = 0
force['M_xy']    = 0
force['p_z']     = 1e2
force['F_z']     = 1

#
geometry      = {}
geometry['a'] = 0.069
geometry['b'] = 1.026

#
laminate          = {}
# -90 <= theta <= 90
laminate['theta'] = array( [45, 45, 0, 45, 0, 45, 0, 45, 45] ) # array( [-45, 30, 0] )
laminate['t']     = 0.2e-3

# Table 2.1 from [1]
material                  = {}
material['composite']     = [f'Graphite/Epoxy']
material['nu_xy']         = 0.048
material['E_x']           = 55e9
material['E_y']           = 55e9
material['G_xy']          = 1.6e9
material['X']             = 1500e6
material['X_c']           = 1500e6
material['X_t']           = 1500e6
material['Y']             = 143e6
material['Y_c']           = 246e6
material['Y_t']           = 40e6
material['S']             = 68e6
material['alpha_x']       = 0.02e-6
material['alpha_y']       = 22.5e-6
material['beta_x']        = 0.00
material['beta_y']        = 0.60
# https://docs-emea.rs-online.com/webdocs/1153/0900766b81153da2.pdf
material['epsilon_u_x_c'] = 0.008
material['epsilon_u_x_t'] = 0.0085
material['epsilon_u_y_c'] = 0.008
material['epsilon_u_y_t'] = 0.0085
material['gamma_u_xy']    = 0.018
material['rho']           = 1.6e3

#
mesh                         = {}
mesh['number_of_elements_x'] = 10
mesh['number_of_elements_y'] = 10

#
settings                        = {}
# 'eps':  'Encapsulated Postscript'
# 'jpeg': 'Joint Photographic Experts Group'
# 'jpg':  'Joint Photographic Experts Group'
# 'pdf':  'Portable Document Format'
# 'pgf':  'PGF code for LaTeX'
# 'png':  'Portable Network Graphics'
# 'ps':   'Postscript',
# 'raw':  'Raw RGBA bitmap'
# 'rgba': 'Raw RGBA bitmap'
# 'svg':  'Scalable Vector Graphics'
# 'svgz': 'Scalable Vector Graphics'
# 'tif':  'Tagged Image File Format'
# 'tiff': 'Tagged Image File Format'
settings['fileformat']            = f'pdf'
settings['margin']                = 0.1
settings['mn_max']                = 40
settings['mn_min']                = 5
settings['number_of_data_points'] = 360
settings['number_of_elements_x']  = 100
settings['number_of_elements_y']  = 100
settings['resolution']            = 600
settings['simulation']            = f'verification'

##
##
##

simulation = settings['simulation']

if not isdir(f'{simulation}'):
    mkdir(f'{simulation}')
    mkdir(f'{simulation}/data')
    mkdir(f'{simulation}/illustrations')

else:
    if not isdir(f'{simulation}/data'):
        mkdir(f'{simulation}/data')

    if not isdir(f'{simulation}/illustrations'):
        mkdir(f'{simulation}/illustrations')

##
## Classical Laminated Plate Theory
##

# Import functions into library
from classical_laminated_plate_theory.ply_stiffness            import ply_stiffness
from classical_laminated_plate_theory.ABD_matrix               import ABD_matrix
from classical_laminated_plate_theory.laminate_classification  import laminate_classification
from classical_laminated_plate_theory.ABD_matrix_inverse       import ABD_matrix_inverse
from classical_laminated_plate_theory.laminate_stiffness       import laminate_stiffness
from classical_laminated_plate_theory.moisture_coefficients    import moisture_coefficients
from classical_laminated_plate_theory.thermal_coefficients     import thermal_coefficients
from classical_laminated_plate_theory.midplane_strain          import midplane_strain
from classical_laminated_plate_theory.global_strain            import global_strain
from classical_laminated_plate_theory.local_strain             import local_strain
from classical_laminated_plate_theory.global_stress            import global_stress
from classical_laminated_plate_theory.local_stress             import local_stress

#
if not isdir(f'{simulation}/data/classical_laminated_plate_theory'):
    mkdir(f'{simulation}/data/classical_laminated_plate_theory')

#
if not isdir(f'{simulation}/illustrations/classical_laminated_plate_theory'):
    mkdir(f'{simulation}/illustrations/classical_laminated_plate_theory')

#
material, stiffness = ply_stiffness(laminate, material, settings, data=True)
#
laminate, stiffness = ABD_matrix(laminate, settings, stiffness, data=True)
#
laminate, stiffness = laminate_classification(laminate, material, settings, stiffness, data=True)
#
stiffness = ABD_matrix_inverse(settings, stiffness, data=True)
#
laminate = laminate_stiffness(laminate, settings, stiffness, data=True)
#
stiffness = moisture_coefficients(laminate, material, settings, stiffness, data=True)
#
stiffness = thermal_coefficients(laminate, material, settings, stiffness, data=True)
#
strain = midplane_strain(force, settings, stiffness, data=True)
#
strain = global_strain(laminate, settings, strain, data=True, illustrations=True)
#
strain = local_strain(laminate, settings, strain, data=True, illustrations=True)
#
stress = global_stress(laminate, settings, stiffness, strain, data=True, illustrations=True)
#
stress = local_stress(laminate, settings, stress, data=True, illustrations=True)

##
## Material Failure Envelopes
##

# # Import functions into library
# from MaterialFailureEnvelope.MaterialFailureEnvelopes import MaterialFailureEnvelopes

# #
# if not isdir(f'{simulation}/data/MaterialFailureEnvelopes'):
#     mkdir(f'{simulation}/data/MaterialFailureEnvelopes')

# #
# if not isdir(f'{simulation}/illustrations/MaterialFailureEnvelopes'):
#     mkdir(f'{simulation}/illustrations/MaterialFailureEnvelopes')

# #
# MaterialFailureEnvelopes(laminate, material, stiffness, settings, data=True, illustrations=True)

assumeddeflection = {}
#
# Assumed Deflection
#

# # Import functions into library
# from AssumedDeflection.AssumedDeflection                 import AssumedDeflection

# #
# assumeddeflection, mesh, settings = AssumedDeflection(geometry, mesh, settings)

##
## Pressure Load
##

# # Import functions into library
# from PressureLoad.PressureLoad_SpeciallyOrthotropic import PressureLoad_SpeciallyOrthotropic
# from PressureLoad.PressureLoad_MidPlaneSymmetric    import PressureLoad_MidPlaneSymmetric
# from PressureLoad.PressureLoad                      import PressureLoad

# #
# if not isdir(f'{simulation}/data/PressureLoad'):
#     mkdir(f'{simulation}/data/PressureLoad')

# #
# if not isdir(f'{simulation}/illustrations/PressureLoad'):
#     mkdir(f'{simulation}/illustrations/PressureLoad')

# #
# PressureLoad_SpeciallyOrthotropic(assumeddeflection, force, geometry, mesh, settings, stiffness, data=True, illustrations=True)

# #
# PressureLoad_MidPlaneSymmetric(assumeddeflection, force, geometry, mesh, settings, stiffness, data=True, illustrations=True)

# #
# PressureLoad(assumeddeflection, force, geometry, laminate, mesh, settings, stiffness, data=True, illustrations=True)

# ##
# ## Point Load
# ##

# # Import functions into library
# from PointLoad.PointLoad_SpeciallyOrthotropic import PointLoad_SpeciallyOrthotropic
# from PointLoad.PointLoad                      import PointLoad

# #
# if not isdir(f'{simulation}/data/PointLoad'):
#     mkdir(f'{simulation}/data/PointLoad')

# #
# if not isdir(f'{simulation}/illustrations/PointLoad'):
#     mkdir(f'{simulation}/illustrations/PointLoad')

# #
# PointLoad_SpeciallyOrthotropic(assumeddeflection, force, geometry, mesh, settings, stiffness, data=False, illustrations=False)

# #
# PointLoad(assumeddeflection, force, geometry, laminate, mesh, settings, stiffness, data=False, illustrations=False)

# #
# # Eigenfrequency analysis
# #

# # Import functions into library
# from EigenfrequencyAnalysis.eigenfrequency_specially_orthotropic                 import eigenfrequency_specially_orthotropic
# from EigenfrequencyAnalysis.eigenfrequency_anisotropic                 import eigenfrequency_anisotropic
# from EigenfrequencyAnalysis.eigenfrequency                 import eigenfrequency
# from EigenfrequencyAnalysis.eigenfrequency_specially_orthotropic_approximation import eigenfrequency_specially_orthotropic_approximation


# #
# if not isdir(f'{simulation}/data/EigenfrequencyAnalysis'):
#     mkdir(f'{simulation}/data/EigenfrequencyAnalysis')

# #
# if not isdir(f'{simulation}/illustrations/EigenfrequencyAnalysis'):
#     mkdir(f'{simulation}/illustrations/EigenfrequencyAnalysis')

# #
# eigenfrequency_specially_orthotropic(assumeddeflection, material, settings, stiffness, illustrations=True)

# #
# eigenfrequency_anisotropic(assumeddeflection, material, settings, stiffness, illustrations=True)

# #
# eigenfrequency(assumeddeflection, material, settings, stiffness, illustrations=True)

# #
# eigenfrequency_specially_orthotropic_approximation(geometry, laminate, material, settings, stiffness)

#
# Stability Analysis
#

# Import function(s)
from stability_analysis.stability_analysis import stability_analysis

#
if not isdir(f'{simulation}/data/stability_analysis'):
    mkdir(f'{simulation}/data/stability_analysis')

#
if not isdir(f'{simulation}/illustrations/stability_analysis'):
    mkdir(f'{simulation}/illustrations/stability_analysis')

#
stability_analysis(assumeddeflection, geometry, settings, stiffness)