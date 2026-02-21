import numpy as np

def charge_derivative(charge1, chargem1, chargex, chargey, chargez, chargemx, chargemy, chargemz, E):
    '''
    Originally written by Author: Rui Zhang Date: 04/28/2025
    Published in Zhang, Rui, et al. "RASCBEC: Raman spectroscopy calculation via born effective charge." Computer Physics Communications 307 (2025): 109425
    Takes eight rotated Born charges under applied field in the index that is ijk=ikj and return.
    '''
    #charges
    c1, cm1, cx, cy, cz, cmx, cmy, cmz = (np.array(ch).T for ch in (charge1, chargem1, chargex, chargey, chargez, chargemx, chargemy, chargemz))

    #charge derivative 3*3*3 tensor
    dq = np.zeros((3, 3, 3))

    for i in range(3):
        for j in range(3):
             dq[i,j,j] = (c1[i,j]-cm1[i,j])/E

    dq[2,0,1] = dq[2,1,0] = 0.5*(np.sqrt(2)*(cx[2,0]-cmx[2,0])-(c1[2,0]-cm1[2,0])-(c1[2,1]-cm1[2,1]))/E
    dq[0,0,1] = dq[0,1,0] = 0.5*(cx[0,0]-cmx[0,0]-cx[1,0]+cmx[1,0]-c1[0,0]+cm1[0,0]-c1[0,1]+cm1[0,1])/E
    dq[1,1,0] = dq[1,0,1] = 0.5*(cx[0,0]-cmx[0,0]+cx[1,0]-cmx[1,0]-c1[1,0]+cm1[1,0]-c1[1,1]+cm1[1,1])/E

    dq[0,1,2] = dq[0,2,1] = 0.5*(np.sqrt(2)*(cy[0,1]-cmy[0,1])-(c1[0,1]-cm1[0,1])-(c1[0,2]-cm1[0,2]))/E
    dq[1,1,2] = dq[1,2,1] = 0.5*(cy[1,1]-cmy[1,1]-cy[2,1]+cmy[2,1]-c1[1,1]+cm1[1,1]-c1[1,2]+cm1[1,2])/E
    dq[2,2,1] = dq[2,1,2] = 0.5*(cy[1,1]-cmy[1,1]+cy[2,1]-cmy[2,1]-c1[2,1]+cm1[2,1]-c1[2,2]+cm1[2,2])/E

    dq[1,2,0] = dq[1,0,2] = 0.5*(np.sqrt(2)*(cz[1,2]-cmz[1,2])-(c1[1,0]-cm1[1,0])-(c1[1,2]-cm1[1,2]))/E
    dq[2,2,0] = dq[2,0,2] = 0.5*(cz[2,2]-cmz[2,2]-cz[0,2]+cmz[0,2]-c1[2,2]+cm1[2,2]-c1[2,0]+cm1[2,0])/E
    dq[0,0,2] = dq[0,2,0] = 0.5*(cz[2,2]-cmz[2,2]+cz[0,2]-cmz[0,2]-c1[0,2]+cm1[0,2]-c1[0,0]+cm1[0,0])/E

    return dq

def gaussian(x, x0, sigma, amplitude):
    """
    Gaussian function.
    
    Parameters:
    -----------
    x : array-like
        x-axis values (energy grid)
    x0 : float
        Center position of the Gaussian
    sigma : float
        Standard deviation (controls width)
    amplitude : float
        Height/intensity of the Gaussian
        
    Returns:
    --------
    array
        Gaussian function values
    """
    return amplitude * np.exp(-((x - x0)**2) / (2 * sigma**2))

def apply_gaussian_broadening(x_data, y_data, x_grid, sigma):
    """
    Apply Gaussian broadening by replacing each data point with a Gaussian.
    
    Parameters:
    -----------
    x_data : array-like
        Original energy values in cm⁻¹
    y_data : array-like
        Original Raman intensity values
    x_grid : array-like
        Uniform grid for the broadened spectrum
    sigma : float
        Standard deviation for Gaussian broadening (FWHM ≈ 2.355 * sigma)
        
    Returns:
    --------
    x_grid : array
        Energy grid
    y_broadened : array
        Broadened intensity values
    individual_gaussians : list of arrays
        List containing each individual Gaussian component
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    x_grid = np.asarray(x_grid)
    
    # Initialize broadened spectrum
    y_broadened = np.zeros_like(x_grid)
    individual_gaussians = []
    
    # Add Gaussian for each data point
    for x0, amplitude in zip(x_data, y_data):
        gauss = gaussian(x_grid, x0, sigma, amplitude)
        y_broadened += gauss
        individual_gaussians.append(gauss)
    
    return x_grid, y_broadened

rot_deg = np.pi/4
dE = 0.1

rot_axes = np.repeat(np.array(['x', 'z', 'x', 'y']),2)
rot_angles = -45 * np.concatenate([[0, 0], np.ones(6)])

E_fields = np.array([
    [dE, dE, dE],
    [-dE, -dE, -dE],
    [np.sqrt(2) * dE, 1e-5, 1e-5],
    [-np.sqrt(2) * dE, 1e-5, 1e-5],
    [1e-5, np.sqrt(2) * dE, 1e-5],
    [1e-5, -np.sqrt(2) * dE, 1e-5],
    [1e-5, 1e-5, np.sqrt(2) * dE],
    [1e-5, 1e-5, -np.sqrt(2) * dE],
])

labels = ['e', 'me', 'x', 'mx', 'y', 'my', 'z', 'mz']