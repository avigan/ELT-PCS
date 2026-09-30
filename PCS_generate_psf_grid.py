import numpy as np
import sys
sys.path.append('/Users/avigan/Work/PCS/Code/psfsim/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import psfsim as ps
import tqdm
import pandas as pd

from hcipy import *
from vigan.utils import imutils
from vigan.optics import aperture
from pathlib import Path
from astropy.io import fits

import matplotlib.colors as colors

#%%
def wave2erg(wavelength):
	return 1.986e-12 * 1e-6 / wavelength

def jansky2photons(flux, wavelength, bandwidth):
	#erg
	energy_per_photon = wave2erg(wavelength)

	# bandwidth conversion from delta_m to delta_Hz
	c = 299792.0e3 # m/s
	nu_0 = c / (wavelength - bandwidth/2)
	nu_1 = c / (wavelength + bandwidth/2)
	frequency_bandwidth = abs(nu_0 - nu_1)

	return flux * 1e-23 * frequency_bandwidth / energy_per_photon

def generate_psf(wave, star_mag, FoV=150, sampling_factor=5, quantile='Q2'):
    # Aperture
    telescope_diameter = 39.4
    aperture_function = make_elt_aperture()

    dgrid = 2. * telescope_diameter
    num_otf_pixels = 1024

    # Star parameters
    wavelength = wave
    magnitude  = star_mag

    # AO system parameters
    nact = 128
    nact_ncpc = 60
    integration_time = 1 / 2000.
    delay = 250e-6
    nsubap = 4 * (1.2 * nact)**2 * np.pi/4
    readnoise = 0.1
    photon_sensitivity = 1
    read_noise_sensitivity = 1
    AO_throughput = 0.3

    # Derived parameters
    lamD = wavelength / telescope_diameter

    # Science camera sampling
    focal_plane_q = sampling_factor
    field_of_view = FoV

    # The pupil grid and the apertures
    grid = make_pupil_grid(num_otf_pixels, dgrid)
    aperture = evaluate_supersampled(aperture_function, grid, 4)
    aperture /= np.sqrt(np.sum(aperture**2) * grid.weights)

    # zeropoint flux
    telescope_area = np.sum(aperture / aperture.max() * grid.weights)
    zeropoint_flux = jansky2photons(2432.84, 790.0e-9, 149e-9) * telescope_area * 1e4

    # AO system
    # atmosphere = ps.make_single_atmospheric_layers(0.65, 18.7)
    atmosphere = ps.make_armazones_atmospheric_layers(quantile=quantile)

    pitch = telescope_diameter / nact
    dm = ps.DeformableMirror(nact, pitch)
    wfs = ps.WavefrontSensor(integration_time, nsubap, readnoise, photon_sensitivity, read_noise_sensitivity)
    controller = ps.IntegralController(integration_time, delay)
    ao = ps.AdaptiveOptics(grid, delay, dm, wfs, controller)

    # System
    photon_flux = zeropoint_flux * integration_time * 10**(-magnitude / 2.5) * AO_throughput
    ao.optimize(atmosphere, wavelength, photon_flux)

    # The coronagraph
    cor_pupil_grid = make_pupil_grid(256, 1.1 * telescope_diameter)
    focal_grid = make_focal_grid(q=focal_plane_q, num_airy=field_of_view / 2, spatial_resolution=lamD)
    sci_cor_prop = FraunhoferPropagator(cor_pupil_grid, focal_grid)

    # Simulate the coronagraph residuals
    cor_aperture = evaluate_supersampled(aperture_function, cor_pupil_grid, 4)
    sa = SurfaceAberration(cor_pupil_grid, 0.05 * wavelength, telescope_diameter)
    ff = FourierFilter(cor_pupil_grid, lambda t: make_circular_aperture(2 * np.pi * nact_ncpc / telescope_diameter)(t))

    forward_fft = make_fourier_transform(cor_pupil_grid, ff.internal_grid)
    backward_fft = make_fourier_transform(ff.internal_grid, grid)

    low_order_ncpa = np.real(ff.forward(sa.surface_sag + 0j))
    high_order_ncpa = sa.surface_sag - low_order_ncpa
    high_order_ncpa *= 1/2 * 25e-9 / np.std(high_order_ncpa[cor_aperture>0])
    low_order_ncpa *= 1/2 * 1e-9 / np.std(low_order_ncpa[cor_aperture>0])
    sa.surface_sag = high_order_ncpa + low_order_ncpa

    upscaled_surface_sag = (aperture / aperture.max()) * np.real( backward_fft.forward( forward_fft.forward( cor_aperture * sa.surface_sag + 0j) ) )
    upscaled_surface_sag = upscaled_surface_sag.shaped[::-1, ::-1].ravel()
    upscaled_surface_sag *= np.std(sa.surface_sag[cor_aperture>0]) / np.std(upscaled_surface_sag[aperture>0])

    # Coronagraph
    switch_radius = 4.0 * lamD
    perfect_coronagraph = OpticalSystem([sa, PerfectCoronagraph(cor_aperture, order=4), sci_cor_prop])
    imager = OpticalSystem([sa, sci_cor_prop])

    ####
    telescope = ps.Telescope(aperture * np.exp(1j * 4 * np.pi / wavelength * upscaled_surface_sag), telescope_diameter)
    cor_imager = ps.CoronagraphicImager(focal_grid, grid, cor_pupil_grid, switch_radius, cor_aperture, perfect_coronagraph, sci_cor_prop)
    sci_imager = ps.CoronagraphicImager(focal_grid, grid, cor_pupil_grid, switch_radius, cor_aperture, imager, sci_cor_prop)

    hci_cor = ps.HighContrastImager(telescope, atmosphere, ao, cor_imager)
    hci_sci = ps.HighContrastImager(telescope, atmosphere, ao, sci_imager)

    low_pass_res, high_pass_res, total_residual = hci_cor.psf(wavelength, jitter_rms=None)
    low_pass_res, high_pass_res, psf = hci_sci.psf(wavelength, jitter_rms=None)

    return psf, total_residual, lamD

#%%
path = Path('/Users/avigan/Cloud/OSU/Work/PCS/Simulations/psfsim_data')

FoV = 150*2          # [lambda/D]
sampling_factor = 5  # oversampling factor of the PSF [px/(lambda/D)] (number of pixel per lambda/D)
quantile = 'Q2'      # atmospheric quantile
wave_min  = 500e-9   # [nm] - minimum wavelength
wave_max  = 2500e-9  # [nm] - maximum wavelength
wave_step = 100e-9   # [nm] - wavelength step
mag_min   = 0        # [mag] - minimum stellar magnitude
mag_max   = 10       # [mag] - maximum stellar magnitude
mag_step  = 1        # [mag] - stellar magnitude step

# wave_min = 1300e-9
# wave_max = 1400e-9
# mag_min  = 1
# mag_max  = 2

# generate input grid
waves = np.arange(wave_min, wave_max, wave_step)
mags  = np.arange(mag_min, mag_max, mag_step)

# waves = [wave_min, ]
# mags = [mag_min, ]

# encircled energy aperture
aper = aperture.disc(FoV * sampling_factor, sampling_factor, diameter=True, cpix=True)

# spatial sampling
loD = np.arange(FoV // 2 * sampling_factor) / sampling_factor

# final arrays
profiles_psf     = np.zeros((len(waves), len(mags), FoV//2 * sampling_factor))
profiles_coro    = np.zeros((len(waves), len(mags), FoV//2 * sampling_factor))
encircled_energy = np.zeros((len(waves), len(mags)))
trans_coro       = np.zeros((len(waves), len(mags)))
with tqdm.tqdm(total=len(waves)*len(mags)) as pbar:
    for iwave, wave in enumerate(waves):
        for imag, mag in enumerate(mags):
            psf_f, coro_f, lamD = generate_psf(wave, mag, FoV=FoV, sampling_factor=sampling_factor, quantile=quantile)

            psf_raw  = psf_f.shaped
            coro_raw = coro_f.shaped

            # transmission of the coronagraph
            trans_coro[iwave, imag] = np.sum(coro_raw) / np.sum(psf_raw)

            # normalise by total energy
            coro = coro_raw / np.sum(coro_raw)
            psf  = psf_raw / np.sum(psf_raw)

            # normalise by pixel scale (converting in density)
            pixel = lamD/sampling_factor*180/np.pi*3600*1000  # [px/mas] should be ~ 0.64 mas at 0.6µm
            coro /= pixel**2
            psf  /= pixel**2

            # profiles
            p_psf, _  = imutils.profile(psf, ptype='mean')
            p_coro, _ = imutils.profile(coro, ptype='mean')

            profiles_psf[iwave, imag]  = p_psf
            profiles_coro[iwave, imag] = p_coro

            # encircled energy
            encircled_energy[iwave, imag] = np.sum(psf * aper) / np.sum(psf)

            #%% plots

            plt.figure('Images', figsize=(12,4.7))
            plt.clf()
            ax = plt.subplot(1,2,1)
            imshow_psf(psf_f, vmax=1, vmin=1e-6, spatial_resolution=lamD)
            plt.xlabel(r'x ($\lambda / D$)')
            plt.ylabel(r'y ($\lambda / D$)')

            plt.subplot(1,2,2, sharex=ax, sharey=ax)
            imshow_psf(coro_f, vmax=1e-3, vmin=1e-6, spatial_resolution=lamD)
            plt.xlabel(r'x ($\lambda / D$)')
            plt.ylabel(r'y ($\lambda / D$)')

            plt.figure('FastYield profiles')
            plt.clf()
            plt.semilogy(loD, p_psf, label=f'PSF {wave*1e9:.0f}nm', color='C0', linestyle='-')
            plt.semilogy(loD, p_coro*trans_coro[iwave, imag], label=f'Corono {wave*1e9:.0f}nm', color='C1', linestyle='-')
            plt.xlim(left=0)

            #%% temporary
            kv2010 = pd.read_csv('Korkiakoski_Verinaud_2010.csv', index_col=0)
            sep = kv2010.columns.astype(float).values
            idx = kv2010.index[np.argmin(kv2010.index - mag)]
            p_kv2010 = kv2010.loc[idx]

            p_psf_tmp = p_psf / np.max(p_psf)
            p_coro_tmp = p_coro / np.max(p_psf)

            plt.figure('Normalised profiles', figsize=(8, 6))
            plt.clf()
            plt.semilogy(loD, p_psf_tmp, label=f'PSF', color='C0')
            plt.semilogy(loD, p_coro_tmp*trans_coro[iwave, imag], label=f'Corono (perfect)', color='C1')
            plt.semilogy(p_kv2010.index.values.astype(float)/(pixel*sampling_factor), p_kv2010.values, label='Korkiakoski & Verinaud (2010)', color='k')

            plt.xlim(left=0, right=100)
            plt.xlabel('Angular separation [$\lambda/D$]')
            plt.ylabel('Contrast')
            plt.title(f'mag I={mag}, wave={wave*1e9:.0f}nm')
            plt.legend(fontsize='small')
            plt.subplots_adjust(left=0.14, right=0.96, bottom=0.12, top=0.94)

            plt.savefig('pcs_psf_comparison.pdf')

            pbar.update()
            stop

hdr = fits.Header()
hdr['WAVEMIN']  = (wave_min, '[m] - wavelength start')
hdr['WAVEMAX']  = (wave_max, '[m] - wavelength end [m]')
hdr['WAVESTEP'] = (wave_step, '[m] - wavelength step')
hdr['MAGMIN']   = (mag_min, '[mag] - stellar magnitude start')
hdr['MAGMAX']   = (mag_max, '[mag] - stellar magnitude end')
hdr['MAGSTEP']  = (mag_step, '[mag] - stellar magnitude step')
hdr['SAMPLING'] = (sampling_factor, 'spatial oversampling factor')
hdr['QUANTILE'] = (quantile, 'Observing conditions quantile')

fits.writeto(path / f'profiles_psf_{quantile}.fits', profiles_psf, hdr, overwrite=True)
fits.writeto(path / f'profiles_coro_{quantile}.fits', profiles_coro, hdr, overwrite=True)
fits.writeto(path / f'encircled_energy_{quantile}.fits', encircled_energy, hdr, overwrite=True)
fits.writeto(path / f'corono_transmission_{quantile}.fits', trans_coro, hdr, overwrite=True)