import numpy as np
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import astropy.units as u
import astropy.constants as cst
from astropy.io import fits

from pathlib import Path

#%%
root = Path('/Users/avigan/Cloud/OSU/Work/PCS/Simulations/WP2_profiles/CLC/')

wave     = 0.5
star_mag = 0

wave_grid        = np.arange(0.5, 2.51, 0.1)
sep_loD          = np.loadtxt(root / f'separation_lamD_MAG{star_mag*10:03.0f}.txt')
sep_mas          = np.loadtxt(root / f'separation_mas_MAG{star_mag*10:03.0f}.txt')
stellar_flux     = np.loadtxt(root / f'stellar_flux_MAG{star_mag*10:03.0f}.txt')
contrast         = np.loadtxt(root / f'contrast_MAG{star_mag*10:03.0f}.txt')
core_throughput  = np.loadtxt(root / f'core_throughput_MAG{star_mag*10:03.0f}.txt')

psf_sep_loD      = np.loadtxt(root / f'../IMG/separation_lamD_MAG{star_mag*10:03.0f}.txt')
psf_stellar_flux = np.loadtxt(root / f'../IMG/stellar_flux_MAG{star_mag*10:03.0f}.txt')
psf_contrast     = np.loadtxt(root / f'../IMG/contrast_MAG{star_mag*10:03.0f}.txt')

#%% old data used for extrapoliation up to 150 λ/D
old_coro, hdr = fits.getdata(root / '../../psfsim_data/profiles_coro_Q2.fits', header=True)
old_wave_grid = np.arange(hdr['WAVEMIN'], hdr['WAVEMAX']+0.1e-7, hdr['WAVESTEP'])*1e6
old_mag_grid  = np.arange(hdr['MAGMIN'], hdr['MAGMAX']+0.1, hdr['MAGSTEP'])
old_sep_loD   = np.arange(old_coro.shape[2]) / hdr['SAMPLING']

#%%
iwave = np.argmin(np.abs(wave_grid - wave))
old_iwave = np.argmin(np.abs(old_wave_grid - wave))
old_imag  = np.argmin(np.abs(old_mag_grid - star_mag))

psf_sep_loD_p      = psf_sep_loD[iwave]
psf_stellar_flux_p = psf_stellar_flux[iwave]
psf_contrast_p     = psf_contrast[iwave]
sep_loD_p          = sep_loD[iwave]
stellar_flux_p     = stellar_flux[iwave]
contrast_p         = contrast[iwave]

old_coro_p = old_coro[old_iwave, old_imag]

#%% plot
fig = plt.figure(0, figsize=(20, 10))
fig.clf()
ax = fig.add_subplot(121)
ax.semilogy(psf_sep_loD_p, psf_stellar_flux_p, label='Stellar flux', color='C0')
ax.semilogy(sep_loD_p, stellar_flux_p, color='C0', linestyle='--')

ax.semilogy(psf_sep_loD_p, psf_contrast_p, label='Contrast', color='C1')
ax.semilogy(sep_loD_p, contrast_p, color='C1', linestyle='--')

ax.semilogy(old_sep_loD, old_coro_p, label='psfsim (old)', color='C2')

ax.set_xlabel('Separation [$\\lambda/D$]')
ax.set_xlim(0, 150)
ax.set_ylabel('Contrast')
ax.set_title(f'$\\lambda$ = {wave_grid[iwave]:.1f} µm')
ax.set_ylim(1e-9, 1e-2)
ax.legend()

ax = fig.add_subplot(122, sharex=ax)
ax.plot(sep_loD[iwave], core_throughput[iwave], label='Core throughput')
ax.set_xlabel('Separation [$\\lambda/D$]')
ax.set_ylabel('Throughput')
ax.set_ylim(0, 1)
ax.legend()

fig.tight_layout()

fig.savefig(root / f'pcs_wp2_profiles_starmag={star_mag:.1f}.pdf')
