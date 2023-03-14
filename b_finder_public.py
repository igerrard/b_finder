#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to find the turbulence driving parameter ('tdp' in this script), from column density and centroid velocity maps.
Makes a map of centroid velocity dispersion, Mach number, volume density dispersion, column density dispersion, brunt factor and driving parameter.

This script is based off the work of Brunt, Federrath & Price (2010), Federrath et al. (2016), and Steward & Federrath (2022), and is the pipeline used in Gerrard et al., 2023. 
We are solving the equation:
##############################
#                            #
#   sigma(rho)/rho_0 = b*M   #
#                            #
##############################

b_finder can be run as a script, called in full from another function, or you can just call calc_tdp which does the bulk of the work.

__author__="Isabella Gerrard"
__version__="4.0"
__email__="isabella.gerrard@anu.edu.au"
__status__="Public"
"""

import numpy as np
from scipy import fftpack
import argparse
import os
import sys

def get_kernel_mask(centre, radius, w, h):
    """
    Creates a mask for a particular kernel instance
    """
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)
    mask = dist <= radius
    return mask

def kernel_weights(grid, centre, FWHM):
    """
    Populates an array with Gaussian weighting values 
    """
    y      = grid[0]
    x      = grid[1]
    Rsq    = (x-centre[0])**2 + (y-centre[1])**2
    sigma  = FWHM/2.355
    kernel = np.exp(-Rsq/(2*sigma**2))
    return kernel

def gradient_fit(zin, weights = None):
    """
    Creates a planefit to a set of 2D points, taking into account the ambient space.
    This method is described in Steward & Federrath (2022).
    """
    zin = np.array(zin)
    if weights is not None:
        weights = np.array(weights)
    # Find the number of x values and the number of y values of zin
    xn = np.shape(zin)[1]
    yn = np.shape(zin)[0]
    # Create x and y arrays corresponding to coordinates of z
    xin = np.tile(np.array(range(xn), dtype = float), (yn,1))
    yin = np.tile(np.array(range(yn), dtype = float).reshape(yn,1), (1, xn)).reshape(yn, xn)
    # Replace indices for which zin is a nan with nans for x and y
    xin[np.isnan(zin)] = np.nan
    yin[np.isnan(zin)] = np.nan
    # Get the effective size of z
    z_size   = zin.size
    num_nans = (np.isnan(zin)).sum()
    n        = z_size - num_nans
    # If there is not enough data, return the average of the given data
    if n < 3:
        print("planefit: too few points")
        r = np.nanmean(zin)
        return r
    # Shift x, y, z to the center of gravity frame
    x0 = float(np.nansum(xin)) / n
    y0 = float(np.nansum(yin)) / n
    z0 = float(np.nansum(zin)) / n
    x  = xin - x0
    y  = yin - y0
    z  = zin - z0
    # Check if have the weights
    if weights is not None:
        s   = np.nansum(weights)
        sx  = np.nansum(weights * x)
        sy  = np.nansum(weights * y)
        sz  = np.nansum(weights * z)
        sxx = np.nansum(weights * x * x)
        syy = np.nansum(weights * y * y)
        sxy = np.nansum(weights * x * y)
        sxz = np.nansum(weights * x * z)
        syz = np.nansum(weights * y * z)
    else:
        s   = float(n)
        sx  = np.nansum(x)
        sy  = np.nansum(y)
        sz  = np.nansum(z)
        sxx = np.nansum(x * x)
        syy = np.nansum(y * y)
        sxy = np.nansum(x * y)
        sxz = np.nansum(x * z)
        syz = np.nansum(y * z)
    D  = s  * (sxx * syy - sxy * sxy) + sx * (sxy * sy  - sx  * syy) + sy * (sx  * sxy - sxx * sy)
    r0 = sz * (sxx * syy - sxy * sxy) + sx * (sxy * syz - sxz * syy) + sy * (sxz * sxy - sxx * syz)
    r1 = s  * (sxz * syy - sxy * syz) + sz * (sxy * sy  - sx  * syy) + sy * (sx  * syz - sxz * sy)
    r2 = s  * (sxx * syz - sxz * sxy) + sx * (sxz * sy  - sx  * syz) + sz * (sx  * sxy - sxx * sy)
    r  = np.array([r0, r1, r2]) / D
    # Now shift x, y, and z back
    r[0]     = r[0] + z0   - r[1] * x0   - r[2] * y0
    planefit = r[0] + r[1] * xin  + r[2] * yin
    return planefit

def get_power_spectrum(map):
    """ 
    Computes power spectrum of some 2D field F(x,y), in this case column density
    """
    # construct map to do power spectrum analysis on and set NaNs to zero
    scaled_map              = np.copy(map) 
    indices_nan             = np.where(np.isnan(scaled_map))
    scaled_map[indices_nan] = 0.0 
    data                    = scaled_map.astype(float)
   
     # 2D Fourier transformation
    ft             = fftpack.fft2(data)/(data.shape[0]*data.shape[1])
     # shift the transform so k = (0,0) is in the center
    ft_c           = fftpack.fftshift(ft) 
    # compute the 2D power spectrum
    power_spectrum = np.abs(ft_c*np.conjugate(ft_c)) 

    # define k (wave number) grid
    nx = power_spectrum.shape[0]
    ny = power_spectrum.shape[1]
    kx = np.linspace(-(nx//2), nx//2+(nx%2-1), nx)
    ky = np.linspace(-(ny//2), ny//2+(ny%2-1), ny)
    kx_grid, ky_grid = np.meshgrid(kx,ky,indexing="ij")
    k = np.hypot(kx_grid,ky_grid)
    
    # trim LHS (in case of even input data dimension; i.e., k=0 is in the center)
    power_spectrum = power_spectrum[(-(nx%2)+1):,(-(ny%2)+1):]
    k = k[(-(nx%2)+1):,(-(ny%2)+1):]
    return power_spectrum, k

def get_Brunt_sqrt_R(power_spectrum, k, exclude_k0=True):
    """
    # Computes Brunt factor, see Brunt, Federrath & Price (2010)
    """
    # deal with wave number (k) input shape to get spacing of wave number map
    if k.shape[0] > 1 and k.shape[1] > 1:  
        dkx = k[0,0]-k[0,1]
        dky = k[0,0]-k[1,0]
    else:
        dkx = 1
        dky = 1
    dk      = np.hypot(dkx,dky)

    # compute total of 2D and 3D power, excluding k=0 (excluding the mean)
    P   = np.copy(power_spectrum)
    if exclude_k0: P[P.shape[0]//2, P.shape[1]//2] = 0.0 
    P2D = 2 * np.pi * np.sum(P  ) * dk
    P3D = 4 * np.pi * np.sum(P*k) * dk
    # sqrt of ratio of 2D-to-3D power
    Brunt_sqrt_R  = np.sqrt(P2D / P3D) 
    return Brunt_sqrt_R

def get_corrected_velocity(M1, kernel):
    """
    Gradient correct the centoid velocity map, and cacluate goodness-of-fit cs
    """
    M1_fit       = gradient_fit(M1, weights=kernel)
    M1_corrected = M1 - M1_fit
    return M1_corrected

def get_corrected_column_density(M0, kernel):
    """
    Gradient correct the column density map, and cacluate goodness-of-fit cs
    """
    M0_norm      = M0/np.nanmean(M0)
    M0_fit       = 10**(gradient_fit(np.log10(M0_norm), weights=kernel))
    M0_corrected = M0_norm/M0_fit
    return M0_corrected

def weighted_std(data, kernel, mask):
    """
    Calculate kernel-weighted standard deviation of 3D quantities.
    Assumes data and weights has nans and is 2d arrays
    """
    data     = data[mask]
    kernel   = kernel[mask]
    if True in np.isnan(data):
        return np.nan
    else:
        mean = np.average(data, weights=kernel)
        ms   = np.average(data**2, weights=kernel)
    return np.sqrt(ms-mean**2)

def calc_tdp(mask, input_kernel, input_M0, input_M1, cs, vel_correction):
    """
    This function takes the data and, cleans it and produces the turbulence driving parameter in a single kernel instance.
    """
    # Apply the kernel mask to the data
    M0     = np.array(np.where(mask, input_M0,     np.nan))   
    M1     = np.array(np.where(mask, input_M1,     np.nan))
    kernel = np.array(np.where(mask, input_kernel, np.nan))
    
    if len(M0[~np.isnan(M0)]) <= 3:    # Don't bother if there are less than 3 data points, gradient fit will fail.
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan
   
    # Gradient correct the M0 and M1 data
    M1_corrected = get_corrected_velocity(      M1, kernel)
    M0_corrected = get_corrected_column_density(M0, kernel)

    # Compute power spectum of column density
    PSpec, k = get_power_spectrum(M0_corrected) 
    # Compute the brunt factor
    R_sqrt   = get_Brunt_sqrt_R(PSpec, k) 

    # Compute column density dispersion    
    sigr_2d = weighted_std(M0_corrected, kernel, mask)
    # Compute centroid velocity dispersion
    sig_cv  = weighted_std(M1_corrected, kernel, mask)
    # Compute volume density dispersion
    sigr_3d = sigr_2d/R_sqrt 
    # Compute Mach number
    mach    = sig_cv*vel_correction/cs 
    # Compute driving parameter
    tdp     = sigr_3d/mach 

    return R_sqrt, sigr_3d, mach, tdp, sig_cv, sigr_2d

def process_file(m0, m1, sens=None, name=None, N_bpk=20, correct_v=3.3, pix2beam=30/7, diameter=3., sound_speed=8.3):
    """
    Driver function to do the b calculation across some map.
    m0: required, numpy array of M0 map
    m1: required, numpy array of M1 map
    sens: optional, numpay array with same shape as M0 and M1 containing some SNR or sensitivity map associated with the data, expected to have values between 0 and 1
    name: basename for the output files
    N_bpk: number of instrument beams per kernel, size of the FWHM
    correct_v: conversion factor to go from 1D velocity dispersion to 3D velocity dispersion, taken from Federrath & Stewart (2022, Tab. E1. lines 4-6)
    pix2beam: number of pixels per beam in the data
    diameter: kernel truncation diameter, in this case it would be 3*N_bpk.
    sound_speed: sound speed of the gas in km/s. It can be either one number, or if you have spatial information on the sound speed with the same resolution as the moment maps, you can use that (untested)
    """

    if name is None: name = m1[:-13] # change this to something sensible for your data
    if os.path.isfile(name+'_tdp_FWHM'+str(N_bpk)+'.npy'): sys.exit("file already processed") # check if the file has already been processed.

    # load the arrays if they are saved to file 
    if isinstance(m0, str): 
        M0 = np.load(m0)
    else:
        M0 = m0
    if isinstance(m1, str): 
        M1 = np.load(m1)
    else:
        M1 = m1
    if isinstance(sens, str): 
        sensitivity = np.load(sens)
    else:
        sensitivity = sens

    # Load either a map of the sound speed, or a constant.
    if not isinstance(sound_speed, float):
        cs = np.load(sound_speed)
    else:
        cs = sound_speed

    # Find the shape of the input arrays, so we can setup empty arrays of the same shape for the outputs
    xlen  = len(M0[:,0])
    ylen  = len(M0[0,:])
    N_pix = pix2beam*N_bpk # find the FWHM of the kernel in pixels 
    R     = int(diameter*N_pix)//2 # Find the radius of the truncated kernel in pixels.

    # Make output arrays the same size as the input, but fill with NaNs for easy masking later
    brunt_arr   = np.empty([xlen,ylen])*np.nan
    sigr_3d_arr = np.empty([xlen,ylen])*np.nan
    mach_arr    = np.empty([xlen,ylen])*np.nan
    tdp_arr     = np.empty([xlen,ylen])*np.nan
    sigr_2d_arr = np.empty([xlen,ylen])*np.nan
    sig_cv_arr  = np.empty([xlen,ylen])*np.nan
    total_its   = (xlen-2*R)*(ylen-2*R)
    counter     = 1

    # loop over each pixel in the map, but the edges get cropped by the radius of the truncated kernel.
    for i in range(R, xlen-R): 
        for j in range(R, ylen-R):
            print()
            print('--- Iteration: '+str(counter)+'/'+str(total_its))
            print('--- Pixel Coords: '+str(i)+' , '+str(j))
            print(flush=True)

            # This masks out everything that isn't in this particular instance of the kernel, then trims down the data to make it quicker to process.
            centre      = np.array([i, j])       
            mask        = get_kernel_mask(centre, R, xlen, ylen)
            trim_M0     = M0[j-R:j+R+1, i-R:i+R+1]
            trim_M1     = M1[j-R:j+R+1, i-R:i+R+1]
            trim_mask   = mask[j-R:j+R+1, i-R:i+R+1]
            trim_grid   = np.mgrid[j-R:j+R+1, i-R:i+R+1]
            trim_kernel = kernel_weights(trim_grid, np.array([i, j]), N_pix)

            # add in the sensitivity map to the weighting of the kernel, if it is provided.
            if sens is not None: trim_kernel= trim_kernel*sensitivity[j-R:j+R+1, i-R:i+R+1]
            if not isinstance(cs, float): # Trim down the sound speed map if provided
                trim_cs = cs[j-R:j+R, i-R:i+R]
            else:
                trim_cs = cs
            
            # We will only perform the calculations on kernel instances that do not contain erroneous pixels.
            if np.isnan(trim_M0) is True: 
                print('No viable data in this kernel')
                R_sqrt, sigr_3d, mach, tdp, sig_cv, sigr_2d = [np.nan]*6
            else:
                R_sqrt, sigr_3d, mach, tdp, sig_cv, sigr_2d = calc_tdp(trim_mask, trim_kernel, trim_M0, trim_M1, trim_cs, correct_v)

            # fill in the output arrays with the values for this particular instance
            brunt_arr[i,j]   = R_sqrt
            sigr_3d_arr[i,j] = sigr_3d
            mach_arr[i,j]    = mach
            tdp_arr[i,j]     = tdp
            sig_cv_arr[i,j]  = sig_cv
            sigr_2d_arr[i,j] = sigr_2d

            counter += 1
    
    # save the numpy arrays to file.
    np.save(name+'_brunt_FWHM'+str(N_bpk)+'.npy',  brunt_arr)
    np.save(name+'_sigr_3d_FWHM'+str(N_bpk)+'.npy',sigr_3d_arr)
    np.save(name+'_mach_FWHM'+str(N_bpk)+'.npy',   mach_arr)
    np.save(name+'_tdp_FWHM'+str(N_bpk)+'.npy',    tdp_arr)
    np.save(name+'_sig_cv_FWHM'+str(N_bpk)+'.npy', sig_cv_arr)
    np.save(name+'_sigr_2d_FWHM'+str(N_bpk)+'.npy',sigr_2d_arr)
    print('Saved patch arrays')

def parse_args(process_args_locally=False):
    parser = argparse.ArgumentParser(description='Calculate the turbulence driving parameter (tdp)')
    parser.add_argument("-m0",          "--m0",          type=None,  default=None, help="name of M0 map numpy file, or numpy array")
    parser.add_argument("-m1",          "--m1",          type=None,  default=None, help="name of M1 map numpy file, or numpy array")
    parser.add_argument("-sens",        "--sens",        type=str,   default=None, help="name of sensitivity map numpy file, or numpy array")
    parser.add_argument("-N_bpk",       "--N_bpk",       type=float, default=10,   help="FWHM of kernel in beams")
    parser.add_argument("-diameter",    "--diameter",    type=float, default=3.,   help="truncation diameter of gaussian kernel, i.e. 3 times the FWHM")
    parser.add_argument("-pix2beam",    "--pix2beam",    type=float, default=30/7, help="pixel to beam conversion")
    parser.add_argument("-correct_v",   "--correct_v",   type=float, default=3.3,  help="Velocity correction factor to convert sigv1D to sigv3D, via Stewart method (Stewart & Federrath 2022)")
    parser.add_argument("-sound_speed", "--sound_speed", type=None,  default=8.3,  help="Sound speed (km/s)")

    if process_args_locally:
        args_for_parser = None # normal parse_args mode
    else:
        args_for_parser = [] # this is when the module is included from another python script
    args = parser.parse_args(args=args_for_parser)
    return args

def main():
    args = parse_args(process_args_locally=True)
    process_file(args.m0, args.m1, sens=args.sens, N_bpk=args.N_bpk, name=args.m0[:-13], example=args.example, correct_v=args.correct_v, pix2beam=args.pix2beam, diameter=args.diameter, sound_speed=args.sound_speed)

if __name__ == "__main__":
    main()