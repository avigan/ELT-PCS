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
# ref_cases = pd.read_csv('ref_cases_Raffaele.csv', index_col=0)

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

stellar_params = pd.read_csv('main_sequence_stars.csv')

def stellar_parameters(SpT, parameter='Teff'):
    try:
        index = stellar_params[stellar_params['SpT'] == SpT].index[0]
    except IndexError:
        raise ValueError('Spectral type not found!')
        return None

    indices = stellar_params.index.values
    values  = stellar_params[parameter].values

    good = np.isfinite(values)
    indices = indices[good]
    values  = values[good]

    interp_fun = interpolate.interp1d(indices, values, kind='linear')

    return interp_fun(index)


#%% input parameters

# main instrumental parameters
Dtel         = 39.0   # [m] telescope diameter
COtel        = 0.3    # telescope central obscuration
wave_min     = 0.5    # [µm] minimum wavelength
wave_max     = 1.8    # [µm] maximum wavelength
coro_iwa     = 3.0    # [λ/D] inner-working angle of the coronagraph
R            = 1_000  # spectral resolution
transmission = 0.1    # average end-to-end transmission (atmosphere + telescope + instrument + qe) (similar to SPHERE)
temperature  = -50    # [°C] instrument temperature (SPHERE/IFS)
pix_res_spatial  = 3  # [pix] number of spatial pixels across a resolution element
pix_res_spectral = 1  # [pix] number of spectral pixels along a resolution element
fwhm_spaxel  = 2.5    # [spaxel] ?
ron          = 0.4    # [e-/readout] detector readout noise (assumpin Saphira-like detector)

# main target parameters
star_SpT     = 'G2'   # stellar spectral type, e.g. 'G2'
star_mag     = 5.0    # [mag] stellar magnitude at reference wavelength
planet_dmag  = 20.0   # [mag] companion contrast at reference wavelength
planet_polar = 0.24   # polarization fraction of the companion

# observation parameters
Tint         = 3600   # [s] total integration time
DIT          = 600    # [s] detector integration time of individual exposures
FoV_rotation = 45     # [deg] field-of-view rotation during the observation

# data analysis
asdi_gain = 10
mm_gain   = 100

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

#%% read model
Teff = stellar_parameters(star_SpT, parameter='Teff')
Teff = np.round(Teff/100)*100
logg = 4.0

path = Path('/Users/avigan/data/Models/Spectra/')

# Vega
vega = fits.getdata(path / 'vega_k93.fits')

vega_star_wave = (vega['wavelength'] * u.angstrom).to(u.nm)
vega_star_flux = vega['flux'] * u.erg / u.s / u.cm**2 / u.angstrom
vega_star_phot = vega_star_flux.to(u.ph / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(vega_star_wave))

# model
file = path / f'Husser-2013/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{Teff:05.0f}-{logg:4.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits.gz'

data = fits.getdata(file.parent / '../../WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
star_wave = data / 10     # Angstrom --> nm

data = fits.getdata(file)
star_flux = data * 1e-7   # erg/s/cm2/cm --> W/m^2/µm

star_wave = star_wave * u.nm
star_flux = star_flux * u.W / u.m**2 / u.micron

star_phot = star_flux.to(u.ph / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(star_wave))

ref_vega = vega_star_phot[np.min(np.where(vega_star_wave > wave_ref)[0])]
ref_star = star_phot[np.min(np.where(star_wave > wave_ref)[0])]

star_phot = star_phot / ref_star * ref_vega

stop
#%%
channel_width = (wave_max - wave_min) / (2 * R)
nchannels     = (wave_max - wave_min) / channel_width

loD_ref = wave_ref / Dtel * 180 / np.pi * 3600

psf_peak_area = fwhm_spaxel**2

det_pixel_per_channel = pix_res_spatial * pix_res_spectral * psf_peak_area

# [ph/s for J=0]
detected_photons = transmission * telescope_area * zero_point * channel_width
detected_photons = detected_photons.decompose()

# empirical, from SPHERE/IFS
thermal_background = 2.468*10**((temperature.value-6) / 23)
thermal_background *= u.electron / u.s / u.pix
thermal_noise_channel = thermal_background*np.sqrt(det_pixel_per_channel)

planet_mag = star_mag + planet_dmag

NDIT = Tint / DIT

#%% signal
sep_loD = np.linspace(2, 100, 1000)
sep_psf, dmag_psf = xao_psf(star_mag=star_mag.value)
sep_loD = sep_psf  # !!!!!!!!!!!!!

psf_interp = interpolate.interp1d(sep_psf, dmag_psf)
coro_profile = 10**(-psf_interp(sep_loD)/2.5)

detected_photon_rate_star  = coro_profile * detected_photons * 10**(-star_mag.value / 2.5)
detected_signal_total_star = detected_photon_rate_star * nchannels * Tint
detected_signal_DIT_star   = detected_photon_rate_star * nchannels * DIT

detected_photon_rate_planet  = detected_photons * 10**(-planet_mag.value / 2.5)
detected_signal_total_planet = detected_photon_rate_planet * nchannels * Tint
detected_signal_pol_planet   = detected_signal_total_planet * planet_polar

cancelation = FoV_rotation * np.pi / 180 * (sep_loD - 2) / sep_loD

calibration_noise = coro_profile * np.nanmax(detected_signal_total_star)
residuals_asdi = calibration_noise / asdi_gain

#%% noise
photon_noise_star   = np.sqrt(detected_signal_total_star)
photon_noise_planet = np.sqrt(detected_signal_total_planet)
thermal_noise = thermal_noise_channel * np.sqrt(nchannels) * np.sqrt(Tint)
readout_noise = ron * np.sqrt(nchannels) * np.sqrt(det_pixel_per_channel) * np.sqrt(NDIT)

#%% final SNR

snr_asdi = (detected_signal_total_planet.value * cancelation) / np.sqrt(residuals_asdi.value**2 + photon_noise_star.value**2 + photon_noise_planet.value**2 + thermal_noise.value**2 + readout_noise.value**2)
snr_asdi[snr_asdi < 0] = np.nan
snr_mm   = (detected_signal_total_planet.value) /  np.sqrt(photon_noise_star.value**2 + photon_noise_planet.value**2 + thermal_noise.value**2 + readout_noise.value**2)
snr_asdi[snr_asdi < 0] = np.nan

#%% plots

def loD2mas(x):
    return x * loD_ref.decompose().value * 1000


def mas2loD(x):
    return x / (loD_ref.decompose().value * 1000)

fig = plt.figure('Signals', figsize=(12, 10))
fig.clf()
ax = fig.add_subplot(111)
ax.semilogy(sep_loD, detected_signal_total_star, color='b', label='Signal (star)')
ax.semilogy(sep_loD, photon_noise_star, color='b', linestyle=':', label='Photon noise (star)')
ax.semilogy(sep_loD, residuals_asdi, color='g', linestyle=':', label='ASDI residuals (star)')

ax.axhline(detected_signal_total_planet.value, color='r', label='Signal (planet)')
ax.axhline(photon_noise_planet.value, color='r', linestyle=':', label='Photon noise (planet)')

ax.axhline(thermal_noise.value, linestyle='--', color='0.4', label='Thermal noise')
ax.axhline(readout_noise.value, linestyle='-.', color='0.4', label='Readout noise')

ax.set_xlim(0, 50)
ax.set_xlabel(r'Angular separation [$\lambda/D$]')
# ax.set_ylim(20, 0)
ax.set_ylabel('Signal [ph]')

ax.tick_params(axis='x', which='both', top=False)
secax = ax.secondary_xaxis('top', functions=(loD2mas, mas2loD))
secax.set_xlabel('Angular separation [mas]')

ax.legend()

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92)


fig = plt.figure('SNR', figsize=(12, 10))
fig.clf()
ax = fig.add_subplot(111)
ax.semilogy(sep_loD, snr_asdi, color='b', label='ASDI')
ax.semilogy(sep_loD, snr_mm, color='r', label='MM')

ax.set_xlim(0, 50)
ax.set_xlabel(r'Angular separation [$\lambda/D$]')
# ax.set_ylim(20, 0)
ax.set_ylabel('SNR')

ax.tick_params(axis='x', which='both', top=False)
secax = ax.secondary_xaxis('top', functions=(loD2mas, mas2loD))
secax.set_xlabel('Angular separation [mas]')

ax.legend()

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92)
