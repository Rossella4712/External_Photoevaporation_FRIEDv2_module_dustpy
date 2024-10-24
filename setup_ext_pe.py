import numpy as np
from dustpy import constants as c
import os

from .functions_ext_pe import get_MassLoss_ResampleGrid
from .functions_ext_pe import MassLoss_FRIED, TruncationRadius, LimitInRadius

################################################################################################
# Helper routine to add external photoevaporation to your Simulation object in one line.
################################################################################################
def param_ext_pe_FRIED(sim, UV_Flux = 1000.):
    sim.addgroup('EPE', description = "external photoevaporation")
    sim.EPE.addgroup('FRIED', description = "FRIED grid used to calculate mass loss rates due to external photoevaporation")
    sim.EPE.FRIED.addfield('UV_Flux', UV_Flux, description = 'UV Flux [G_0]', constant=True)
    return

def setup_ext_pe_FRIED(sim, fried_dir = str(os.path.dirname(__file__))+"/FRIEDV2/",
                       fried_filenames = ["FRIEDV2_0p1Msol_fPAH1p0_growth.dat",
                                               "FRIEDV2_0p3Msol_fPAH1p0_growth.dat",
                                               "FRIEDV2_0p6Msol_fPAH1p0_growth.dat",
                                               "FRIEDV2_1p0Msol_fPAH1p0_growth.dat",
                                               "FRIEDV2_1p5Msol_fPAH1p0_growth.dat",
                                               "FRIEDV2_3p0Msol_fPAH1p0_growth.dat"],
                                               SigmaFloor = 1.e-100):
    '''
    Add external photoevaporation using the FRIED grid (Haworth et al., 2018) and the Sellek et al.(2020) implementation.
    This setup routine also performs the interpolation in the stellar mass and UV flux parameters.

    Call the setup function after the initialization and then run, as follows:

    sim.initialize()
    setup_extphoto_FRIED(sim)
    sim.run()
    ----------------------------------------------

    fried_filename:             FRIED grid from Haworth+(2018), download from: http://www.friedgrid.com/Downloads/
    UV_target [G0]:             External UV Flux

    SigmaFloor:                 Re-adjust the floor value of the gas surface density to improve the simulation performance

    ----------------------------------------------
    '''
    ##################################
    # SET THE FRIED GRID
    ##################################
    # Obtain a resampled version of the FRIED grid for the simulation stellar mass and UV_flux.
    # Set the external photoevaporation Fields

    # Define a parameter space for the resampled radial and Sigma grids
    grid_radii = np.concatenate((np.array([5]), np.linspace(10,1000, num = 106)))
    grid_Sigma = np.concatenate((np.array([1e-8,1e-7,1e-6,1e-5,1e-4]), np.logspace(-3, 3, num = 100), np.array([5e3,1e4])))
    grid_Sigma1AU = grid_Sigma*grid_radii

    # Obtain the mass loss grid.
    # Also obtain the interpolator(M400, r) function to include in the FRIED class as a hidden function
    grid_MassLoss, grid_MassLoss_Interpolator = get_MassLoss_ResampleGrid(fried_dir, fried_filenames,
                                                        Mstar_target = sim.star.M/c.M_sun, UV_target = sim.EPE.FRIED.UV_Flux,
                                                        grid_radii = grid_radii, grid_Sigma = grid_Sigma, grid_Sigma1AU = grid_Sigma1AU)
    sim.EPE.FRIED.addgroup('Table', description = "(Resampled) Table of the mass loss rates for a given radial-Sigma grid.")
    sim.EPE.FRIED.Table.addfield("radii", grid_radii, description ="Outer disk radius input to calculate FRIED mass loss rates [AU], (array, nr)")
    sim.EPE.FRIED.Table.addfield("Sigma", grid_Sigma, description = "Surface density grid to calculate FRIED mass loss rates [g/cm^2] (array, nSigma)")
    sim.EPE.FRIED.Table.addfield("MassLoss", grid_MassLoss, description = "FRIED Mass loss rates [log10 (M_sun/year)] (grid, nr*nSigma)")

    # We use this hidden _Interpolator function to avoid constructing the FRIED interpolator multiple times
    sim.EPE.FRIED._Interpolator = grid_MassLoss_Interpolator
    
    # Add the truncation radius
    sim.EPE.FRIED.addfield('rTrunc', sim.grid.r[-1], description = 'Truncation radius [cm]')
    sim.EPE.FRIED.addfield('rLim_in', sim.grid.r[-1], description = 'Radius enclosing 0.9 disc mass [cm]')

    # Add the Mass Loss Rate field from the FRIED Grid
    sim.EPE.FRIED.addfield('MassLoss', np.zeros_like(sim.grid.r), description = 'Mass loss rate obtained by interpolating the FRIED Table at each grid cell [g/s]')

    sim.EPE.FRIED.rLim_in.updater = LimitInRadius
    sim.EPE.FRIED.rTrunc.updater = TruncationRadius
    sim.EPE.FRIED.MassLoss.updater =  MassLoss_FRIED
    # sim.updater = ['star', 'grid', 'FRIED', 'gas', 'dust']
    sim.EPE.FRIED.updater = ['rLim_in','rTrunc' ,'MassLoss']

    ###################################
    # ASSIGN GAS AND DUST LOSS RATES
    ###################################

    sim.gas.S.addfield("EPE", np.zeros_like(sim.gas.Sigma), description="Change in gas surface density through ext. photoevap", save = True)
    sim.dust.S.addfield("EPE", np.zeros_like(sim.dust.Sigma), description="Change in dust surface density through ext. photoevap", save = True)

    ##################################
    # ADJUST THE GAS FLOOR VALUE
    ##################################
    # Setting higher floor value than the default avoids excessive mass loss rate calculations at the outer edge.
    # This speeds the code significantly, while still reproducing the results from Sellek et al.(2020)

    sim.gas.SigmaFloor = SigmaFloor
    return