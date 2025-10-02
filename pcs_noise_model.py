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

#%% reference cases
ref_cases_mags = [5.20, 8.50, 11.40]
ref_cases = pd.read_csv('ref_cases_Goulas2024.csv', index_col=0)

def xao_psf(star_mag):
    sep = ref_cases.index.values

    if star_mag <= ref_cases_mags[0]:
        dmag = -2.5*np.log10(ref_cases['bright'].values)
    elif (ref_cases_mags[0] < star_mag) and (star_mag <= ref_cases_mags[1]):
        mag0  = ref_cases_mags[0]
        dmag0 = -2.5*np.log10(ref_cases['bright'].values)
        mag1  = ref_cases_mags[1]
        dmag1 = -2.5*np.log10(ref_cases['red-1'].values)

        a = (dmag1 - dmag0) / (mag1 - mag0)
        b = -a * mag0 + dmag0

        dmag = a * star_mag + b
    elif (ref_cases_mags[1] < star_mag) and (star_mag <= ref_cases_mags[2]):
        mag1  = ref_cases_mags[1]
        dmag1 = -2.5*np.log10(ref_cases['red-1'].values)
        mag2  = ref_cases_mags[2]
        dmag2 = -2.5*np.log10(ref_cases['red-4'].values)

        a = (dmag2 - dmag1) / (mag2 - mag1)
        b = -a * mag1 + dmag1

        dmag = a * star_mag + b
    else:
        raise ValueError('Star is too faint!')

    return sep, dmag

#%% input parameters

# main instrumental parameters
Dtel         = 39.0   # [m] telescope diameter
COtel        = 0.3    # telescope central obscuration
wave_min     = 0.5    # [µm] minimum wavelength
wave_max     = 1.8    # [µm] maximum wavelength
coro_iwa     = 3.0    # [λ/D] inner-working angle of the coronagraph
R            = 1_000  # spectral resolution
transmission = 0.1    # average end-to-end transmission (atmosphere + telescope + instrument) (similar to SPHERE)
temperature  = -50    # [°C] instrument temperature (SPHERE/IFS)
pix_res_spatial  = 3  # [pix] number of spatial pixels across a resolution element
pix_res_spectral = 1  # [pix] number of spectral pixels along a resolution element
fwhm_spaxel  = 2.5    # [spaxel] ?
ron          = 0.4    # [e-/readout] detector readout noise (assumpin Saphira-like detector)

# main target parameters
star_mag     = 5.0   # [mag] stellar magnitude at reference wavelength
planet_dmag  = 18.0   # [mag] companion contrast at reference wavelength
planet_polar = 0.24   # polarization fraction of the companion

# observation parameters
Tint         = 3600   # [s] total integration time
DIT          = 600    # [s] detector integration time of individual exposures
FoV_rotation = 45     # [deg] field-of-view rotation during the observation

#%% apply units
Dtel         *= u.m
wave_min     *= u.micron
wave_max     *= u.micron
temperature  *= u.deg_C

pix_res_spatial  *= u.pix
pix_res_spectral *= u.pix

star_mag    *= u.mag
planet_dmag *= u.mag

Tint         *= u.s
DIT          *= u.s
FoV_rotation *= u.deg

#%% derived quantities
telescope_area = np.pi * (Dtel / 2)**2 - np.pi * (COtel*Dtel / 2)**2
wave_ref   = (wave_min + wave_max) / 2

# based on fit by Raffaele
zero_point = 10**(-2.1028*np.log10(wave_ref.value*1000)+8.7633)
zero_point *= u.ph / u.s / u.AA / u.cm**2

channel_width = (wave_max - wave_min) / (2 * R)
nchannels     = (wave_max - wave_min) / channel_width

loD_ref = wave_ref / Dtel * 180 / np.pi * 3600

psf_peak_area = fwhm_spaxel**2

det_pixel_per_channel = pix_res_spatial * pix_res_spectral * psf_peak_area

# [ph/s for J=0]
detected_photons = transmission * telescope_area * zero_point * channel_width

# empirical, from SPHERE/IFS
thermal_background = 2.468*10**((temperature.value-6) / 23)
thermal_background *= u.electron / u.s / u.pix
thermal_noise_channel = thermal_background*np.sqrt(det_pixel_per_channel)

planet_mag = star_mag + planet_dmag

NDIT = Tint / DIT

#%% signal
sep_loD = np.linspace(2, 100, 1000)
sep_psf, dmag_psf = xao_psf(star_mag=star_mag.value)

psf_interp = interpolate.interp1d(sep_psf, dmag_psf)
coro_profile = psf_interp(sep_loD)

sep_mas = sep_loD * loD_ref

fig = plt.figure('PSF profile', figsize=(12, 10))
fig.clf()
ax = fig.add_subplot(111)
ax.plot(sep_psf, dmag_psf)
ax.set_xlim(0, 100)
ax.set_xlabel('Angular separation [$\lambda/D$]')
ax.set_ylim(20, 0)
ax.set_ylabel('Contrast [mag]')
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.95)
