import numpy as np
from dustpy import constants as c
from scipy.interpolate import LinearNDInterpolator
from astropy.table import Table, vstack


#####################################
#
# FRIED GRID ROUTINES
#
#####################################

def get_Sigma_1AU(Sigma_out, r_out):
    return Sigma_out*r_out

def Set_FRIED_Interpolator(r_Table, Sigma1AU_Table, MassLoss_Table):
    '''
    Returns the interpolator function, constructed from the FRIED grid data
    The interpolator takes the (M400 [Jupiter mass], r_out[au]) variables
    The interpolator returns the external photoevaporation mass loss rate [log10 (M_sun/year)]

    The interpolation is performed on the loglog space.
    '''
    # Following Sellek et al.(2020) implementation, Sigma(1AU) is used to set the interpolator
    
    #Sigma_1AU_Table = get_Sigma_1AU(Sigma_Table, r_Table)
    #min_mass_loss = min(MassLoss_Table)

    # Interpolation in the [log(Sigma1AU), log(r_out) -> log(MassLoss)] parameter space
    Interpolator = LinearNDInterpolator(list(zip(np.log10(Sigma1AU_Table), np.log10(r_Table))), MassLoss_Table, fill_value=-12.5)

    # Return a Lambda function that converts the linear Sigma1AU,r inputs to the logspace to perform the interpolation.
    return lambda S1AU, r: Interpolator(np.log10(S1AU), np.log10(r))

#####################################
# FRIED GRID ROUTINES - CALLED ONLY ON SETUP
#####################################

def get_MassLoss_SellekGrid(r_grid, Sigma_grid, r_Table, Sigma1AU_Table, MassLoss_Table):
    '''
    Obtain the MassLoss grid in radius and Sigma, following the interpolation of Sellek et al.(2020) Eq.5
    Note: This only work for a single stellar mass and UV Flux, already available in the FRIED grid
    ----------------------------------------
    r_grid, Sigma_grid:                      Radial[AU] and Surface density [g/cm^2] grid to obtain the Mass Loss interpolation
    r_Table, Sigma_Table, MassLoss_Table:    FRIED Grid data columns (masked to match a single stellar mass and UV Flux)

    returns
    MassLoss_SellekGrid [log Msun/year]:     Mass loss rates as shown Figure 1. in Sellek(2020)
    Sigma_min, Sigma_max [g/cm^2]:           Surface density limits of the FRIED grid for the given parameter space
    ----------------------------------------
    '''

    # Obtain the FRIED interpolator function that returns the Mass loss, given a (Sigma1AU, r) input
    FRIED_Interpolator = Set_FRIED_Interpolator(r_Table, Sigma1AU_Table, MassLoss_Table)

    # Find out the shape of the table in the r_Table parameter range
    shape_FRIED = (int(r_Table.size/np.unique(r_Table).size), np.unique(r_Table).size)

    # Obtain the values of r_Table, and the corresponding minimum and maximum value of Sigma in the Fried grid for each r_Table
    r_Table = r_Table.reshape(shape_FRIED)[0]                           # Dimension: unique(Table.r_Table)
    
    Sigma_max = 1e5 * (r_Table**(-1))
    Sigma_min = r_Table**(-1)
    
    # Give a buffer factor, since the FRIED interpolator should not extrapolate outside the original
    buffer_max = 0.9 # buffer for the Sigma upper grid limit
    buffer_min = 1.1 # buffer for the Sigma lower grid limit

    # The interpolation of the grid limits is performed on the logarithmic space
    # See the FRIED grid (r_Table vs. Sigma) data distribution for reference
    f_Sigma_FRIED_max = lambda r_interp: 10**np.interp( np.log10(r_interp),np.log10(r_Table), np.log10(buffer_max * Sigma_max) )  
    f_Sigma_FRIED_min = lambda r_interp: 10**np.interp( np.log10(r_interp),np.log10(r_Table), np.log10(buffer_min * Sigma_min) ) 


    # Calculate the density limits and the corresponding mass loss rates for the custom radial grid
    Sigma_max = f_Sigma_FRIED_max(r_grid)
    Sigma_min = f_Sigma_FRIED_min(r_grid)
    
    
    #S1AU_max = get_Sigma_1AU(Sigma_max, r_grid)
    #S1AU_min = get_Sigma_1AU(Sigma_min, r_grid)

    #MassLoss_max = FRIED_Interpolator(S1AU_max, r_grid)  # Upper limit of the mass loss rate from the fried grid
    #MassLoss_min = FRIED_Interpolator(S1AU_min, r_grid)  # Lower limit of the mass loss rate from the fried grid

    # Mask the regions where the custom Sigma grid is outside the FRIED boundaries
    mask_max= Sigma_grid >= Sigma_max
    mask_min= Sigma_grid <= Sigma_min

    # Calculate the mass loss rate for each grid cell according to the FRIED grid
    # Note that the mass loss rate is in logarithmic-10 space
    #S1AU_grid = get_Sigma_1AU(Sigma_grid, r_grid)

    
    re_Sigma = np.minimum(Sigma_grid, Sigma_max)
    re_Sigma = np.maximum(re_Sigma, Sigma_min)
    S1AU_grid = get_Sigma_1AU(re_Sigma, r_grid)


    MassLoss_SellekGrid = FRIED_Interpolator(S1AU_grid, r_grid) # Mass loss rate from the FRIED grid
    #MassLoss_SellekGrid[r_grid<5] = -13

    mask_out = r_grid>=450.
    min_mass_loss = min(MassLoss_Table)
    #i_rmax = np.where( r_grid>=450. )[0][0]

    MassLoss_SellekGrid[mask_out] = MassLoss_SellekGrid[mask_out][0]

    ot_regime = mask_min * (MassLoss_SellekGrid > min_mass_loss)

    #MassLoss_SellekGrid[mask_max] = MassLoss_max[mask_max]
    MassLoss_SellekGrid[ot_regime] = MassLoss_SellekGrid[ot_regime] + np.log10(Sigma_grid / Sigma_min)[ot_regime] #MassLoss_min[ot_regime] + np.log10(Sigma_grid / Sigma_min)[ot_regime]
    #MassLoss_SellekGrid[MassLoss_SellekGrid < -13] = -13

    return MassLoss_SellekGrid, Sigma_min, Sigma_max

def get_mask_StarUV(Mstar_value, UV_value, Mstar_Table, UV_Table):
    '''
    Construct a boolean mask that indicates rows of the FRIED Grid where Mstar_value and UV_value are present
    Mstar_value, UV_value must be available values of the FRIED Grid
    '''
    mask_Mstar = Mstar_Table == Mstar_value
    mask_UV = UV_Table == UV_value
    mask = mask_UV * mask_Mstar

    return mask

def get_weights_StarUV(Mstar_value, UV_value, Mstar_lr, UV_lr):

    '''
    Returns the interpolation weights for a given Mstar-UV value pair, within a rectangle Mstar-UV rectangle.
    The bi-linear interpolation is performed in the logpace of the Mstar-UV space
    '''

    logspace_weight = lambda value, left, right: (np.log10(value) - np.log10(left)) / (np.log10(right) - np.log10(left))

    f_Mstar = logspace_weight(Mstar_value, Mstar_lr[0], Mstar_lr[1])
    f_UV = logspace_weight(UV_value, UV_lr[0], UV_lr[1])

    f_weights = np.array([1. - f_Mstar, f_Mstar])[:, None] * np.array([1. - f_UV, f_UV])[None, :]
    return f_weights


def get_MassLoss_ResampleGrid(fried_dir, fried_filenames,
                              Mstar_target = 1., UV_target = 1000.,
                              grid_radii = None, grid_Sigma = None, grid_Sigma1AU = None):
    '''
    Resample the FRIED grid into a new radial-Sigma grid for a target stellar mass and UV Flux
    --------------------------------------------
    fried_filename:                      FRIED grid from Haworth+(2018), download from: http://www.friedgrid.com/Downloads/
    Mstar_target [M_sun]:                Target stellar mass to reconstruct the FRIED grid
    UV_target [G0]:                      Target external UV flux to reconstruct the FRIED grid

    grid_radii[array (nr), AU]:                     Target radial grid array to reconstruct the FRIED grid
    grid_Sigma[array (nSig), g/cm^2]:               Target Sigma grid array to reconstruct the FRIED grid

    returns
    grid_MassLoss [array (nr, nSig), log(Msun/yr)]: Resampled Mass loss grid.
    grid_MassLoss_Interpolator:                     A function that returns the interpolated value based on the grid_MassLoss
                                                    The interpolator inputs are M400 [Jupiter mass] and r [AU]
    --------------------------------------------

    '''
    
    if grid_radii is None:
        grid_radii = np.linspace(5, 1000, num = 100)
    if grid_Sigma is None:
        grid_Sigma = np.logspace(-3, 4, num = 100)
        grid_Sigma1AU = grid_Sigma*grid_radii 


    #FRIED_Grid = np.loadtxt(fried_filename, unpack=True, skiprows=1)

    fried_grids=[]
    for name in fried_filenames:
        grid = Table.read(fried_dir+name, format="ascii")
        fried_grids.append(grid)
        
    FRIED_Grid = fried_grids[0]
    for f in fried_grids[1:]:
        FRIED_Grid = vstack( [FRIED_Grid, f] )


    Table_Mstar = np.array( FRIED_Grid["col1"] )
    Table_rout = np.array( FRIED_Grid["col2"] )
    Table_Sigma1AU = np.array( FRIED_Grid["col3"] )
    Table_Sigma = np.array( FRIED_Grid["col4"] )
    Table_UV = np.array( FRIED_Grid["col5"] )
    Table_MassLoss = np.array( FRIED_Grid["col6"] )

    #################################################################################
    # CREATE THE RADII-SIGMA MESHGRID AND THE SHAPE OF THE OUTPUT
    #################################################################################

    grid_radii_ = grid_radii

    grid_radii, grid_Sigma1AU = np.meshgrid(grid_radii, grid_Sigma1AU, indexing = "ij")
    grid_radii_, grid_Sigma = np.meshgrid(grid_radii_, grid_Sigma, indexing = "ij")

    # This is the mass loss grid that we want to use for interpolation during simulation time
    grid_MassLoss = np.zeros_like(grid_radii)


    #################################################################################
    # FIND THE CLOSEST VALUES FOR THE STELLAR MASS AND UV FLUX IN THE FRIED GRID
    #################################################################################

    unique_Mstar = np.unique(Table_Mstar)
    unique_UV = np.unique(Table_UV)

    i_Mstar = np.searchsorted(unique_Mstar, Mstar_target)
    i_UV = np.searchsorted(unique_UV, UV_target)

    # Left/Right values around the available UV and Star Flux
    Mstar_lr = unique_Mstar[[i_Mstar - 1 , i_Mstar]]
    UV_lr = unique_UV[[i_UV - 1, i_UV]]

    #################################################################################
    # CONSTRUCT A MASS LOSS GRID FOR EACH OF THE CLOSEST STELLAR MASSES AND FLUXES
    #################################################################################

    grid_MassLoss_StarUV = []
    for Mstar_value in Mstar_lr:
        grid_MassLoss_StarUV.append([])
        for UV_value in UV_lr:
            # Mask the FRIED grid for available Mstar and UV values
            mask = get_mask_StarUV(Mstar_value, UV_value, Table_Mstar, Table_UV)

            # Save the MassLoss grid into a collection
            # The function also returns the surface density limits, but we do not need them here
            grid_MassLoss_dummy = get_MassLoss_SellekGrid(grid_radii, grid_Sigma, Table_rout[mask], Table_Sigma1AU[mask], Table_MassLoss[mask])[0]
            grid_MassLoss_StarUV[-1].append(grid_MassLoss_dummy)
    grid_MassLoss_StarUV = np.array(grid_MassLoss_StarUV)


    #################################################################################
    # GET THE FINAL MASS LOSS GRID FOR THE TARGET UV FLUX AND STELLAR MASS
    #################################################################################

    interpolation_weights = get_weights_StarUV(Mstar_target, UV_target, Mstar_lr, UV_lr)
    grid_MassLoss = (interpolation_weights[:, :, None, None] * grid_MassLoss_StarUV).sum(axis=(0,1))

    # Return both the resampled grid for mass loss rates, and an interpolator function for it.
    # This is to avoid building the interpolator multiple times during the simulation run
    grid_MassLoss_Interpolator = Set_FRIED_Interpolator(grid_radii.flatten(), grid_Sigma1AU.flatten(), grid_MassLoss.flatten())

    return grid_MassLoss, grid_MassLoss_Interpolator


##########################################################################
# UPDATER OF THE MASS LOSS FROM THE FRIED GRID AND TRUNCATION RADIUS
##########################################################################

# Called every timestep
def MassLoss_FRIED(sim):
    '''
    Calculates the instantaneous mass loss rate from the FRIED Grid (Haworth+, 2018) for each grid cell,

    '''

    # Interpolate the FRIED grid using the simulation radii and Sigma
    r_AU = sim.grid.r/c.au
    Sigma_g = sim.gas.Sigma
    #M500 = get_M500(Sigma_g, r_AU)
    Sigma_1AU = get_Sigma_1AU(Sigma_g, r_AU)

    # Calls the interpolator hidden inside the FRIED class
    # This way it is not necessary to construct the interpolator every timestep, which is really time consuming
    MassLoss = sim.EPE.FRIED._Interpolator(Sigma_1AU, r_AU)

    # Convert to cgs
    MassLoss = np.power(10, MassLoss) * c.M_sun/c.year

    return MassLoss

def LimitInRadius(sim):
    Sigma_gas_disc = sim.gas.Sigma
    ring_area = sim.grid.A
    mass_gas = (ring_area * Sigma_gas_disc)
    mass_gas_tot = np.sum(mass_gas)
    mass_gas_cum = np.cumsum(mass_gas)
    i_lim_in = np.where(mass_gas_cum > (0.9*mass_gas_tot))[0][0]
    return sim.grid.r[i_lim_in]

def TruncationRadius(sim):
    '''
    Find the photoevaporative radii.
    See Sellek et al. (2020) Figure 2 for reference.
    '''

    # Near the FRIED limit, the truncation radius is extremely sensitive to small variations in the MassLoss profile.
    # If the profile is completely constant, the truncation radius becomes the last grid cell

    MassLoss = sim.EPE.FRIED.MassLoss * c.year / (2*np.pi)   #MassLossFRIED in g/s here <---
    # round to 10^-12 solar masses per year
    #MassLoss = np.round(MassLoss, 12)
    
    ir_ext = np.size(MassLoss) - np.argmax(MassLoss[::-1]) - 1
    R_lim_in = sim.EPE.FRIED.rLim_in

    if sim.grid.r[ir_ext] < R_lim_in:
        return R_lim_in
    else:
        return sim.grid.r[ir_ext]

#####################################
# GAS LOSS RATE
#####################################

def SigmaDot_ExtPhoto(sim):
    '''
    Compute the Mass Loss Rate profile using Sellek+(2020) approach, using the mass loss rates from the FRIED grid of Haworth+(2018)
    '''
    sim.EPE.FRIED.update()
    # Mask the regions that should be subject to external photoevaporation
    mask = sim.grid.r >= sim.EPE.FRIED.rTrunc

    # Obtain Mass at each radial ring and total mass outside the photoevaporative radius
    mass_profile = sim.grid.A * sim.gas.Sigma
    mass_ext = np.sum(mass_profile[mask])

    # Total mass loss rate.
    mass_loss_ext = np.sum((sim.EPE.FRIED.MassLoss * mass_profile)[mask] / mass_ext)

    # Obtain the surface density profile using the mass of each ring as a weight factor
    # Remember to add the (-) sign to the surface density mass loss rate

    if (sim.t/(c.year*1e6)) < 1. :
        SigmaDot = np.zeros_like(sim.grid.r)
    else :
        SigmaDot = np.zeros_like(sim.grid.r)
        SigmaDot[mask] = -sim.gas.Sigma[mask] *  mass_loss_ext / mass_ext

    # return the surface density loss rate [g/cm²/s]
    return SigmaDot

#####################################
# DUST ENTRAINMENT AND LOSS RATE
#####################################

def PhotoEntrainment_Size(sim):
    '''
    Returns a radial array of the dust entrainment size.
    See Eq. 11 from Sellek+(2020)
    '''
    v_th = np.sqrt(8/np.pi) * sim.gas.cs                    # Thermal speed
    F = sim.gas.Hp / np.sqrt(sim.gas.Hp**2 + sim.grid.r**2) # Geometric Solid Angle
    rhos = sim.dust.rhos[0,0]                               # Dust material density

    # Calculate the total mass loss rate (remember to add the (-) sign)
    M_loss = -np.sum(sim.grid.A * sim.gas.S.ext)

    a_ent = v_th / (c.G * sim.star.M) * M_loss /(4 * np.pi * F * rhos)
    return a_ent

def PhotoEntrainment_Fraction(sim,a_ent):
    '''
    Returns fraction of dust grains that are entrained with the gas for each species at each location.
    * Must be multiplied by the dust-to-gas ratio to account for the mass fraction
    * Must be zero for grain sizes larger than the entrainment size
    * In Sellek+2020 the mass fraction is used to account for the dust distribution as well, but in dustpy that information comes for free in the sim.dust.Sigma array
    * Currently this factor must be either 1 (entrained) or 0 (not entrained)
    '''
    mask = sim.dust.a < a_ent[:, None] # Mask indicating which grains are entrained
    f_ent = np.where(mask, 1., 0.)
    return f_ent

def SigmaDot_ExtPhoto_Dust(sim):
    a_ent = PhotoEntrainment_Size(sim)
    f_ent = PhotoEntrainment_Fraction(sim,a_ent)                    # Factor to mask the entrained grains.
    d2g_ratio = sim.dust.Sigma / sim.gas.Sigma[:, None]             # Dust-to-gas ratio profile for each dust species
    SigmaDot_Dust = f_ent * d2g_ratio * sim.gas.S.ext[:, None]      # Dust loss rate [g/cm²/s]
    return SigmaDot_Dust

def S_gas_epe(sim):
    S = SigmaDot_ExtPhoto(sim)
    sim.gas.S.EPE = S
    return S

def S_dust_epe(sim):
    S = SigmaDot_ExtPhoto_Dust(sim)
    sim.dust.S.EPE = S
    return S