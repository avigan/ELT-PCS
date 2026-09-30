# import FastYield modules
from fastyield.config import rad2arcsec
from fastyield.utils import interp_extrap_sep, build_PSF_grid, plot_profiles, add_if_necessary

# import astropy modules
from astropy.io import fits

# import matplotlib modules
import matplotlib.pyplot as plt

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logit, expit
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter1d

# import other modules
from tqdm import tqdm
from pathlib import Path



def radial_centers_to_edges(r):
    """
    Convert radial bin centers to radial bin edges.
    Convention consistent with your script:
      - first center at r=0
      - first edge at 0
      - intermediate edges at midpoints
      - last edge extrapolated by half-bin
    """
    r = np.asarray(r, dtype=float)

    if r.ndim != 1:
        raise ValueError("r must be 1D.")
    if len(r) == 0:
        raise ValueError("r must not be empty.")
    if np.any(r < 0):
        raise ValueError("r must be >= 0.")
    if np.any(np.diff(r) < 0):
        raise ValueError("r must be sorted in increasing order.")

    edges = np.empty(len(r) + 1, dtype=float)

    if len(r) == 1:
        # Fallback case, rarely useful here
        dr = r[0] if r[0] > 0 else 1.0
        edges[0] = 0.0
        edges[1] = r[0] + 0.5 * dr
        return edges

    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0] = 0.0
    edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])

    return edges



def _overlap_annulus_area_matrix(edges_in, edges_out):
    """
    Matrix A such that:
        density_out = (A @ density_in) / area_out
    where A[i, j] is the exact overlap area between input annulus j
    and output annulus i.

    All radii must be expressed in the SAME unit.
    """
    a_in = edges_in[:-1]
    b_in = edges_in[1:]
    a_out = edges_out[:-1]
    b_out = edges_out[1:]

    n_in = len(a_in)
    n_out = len(a_out)

    A = np.zeros((n_out, n_in), dtype=float)

    j0 = 0
    for i in range(n_out):
        a, b = a_out[i], b_out[i]

        while j0 < n_in and b_in[j0] <= a:
            j0 += 1

        j = j0
        while j < n_in and a_in[j] < b:
            aa = max(a, a_in[j])
            bb = min(b, b_in[j])

            if bb > aa:
                A[i, j] = np.pi * (bb**2 - aa**2)

            j += 1

    return A



def rebin_radial_profile_density(profile_density, separation_in, pxscale_out, separation_out=None):
    """
    Flux-conserving radial rebinning for annular profile densities.

    Parameters
    ----------
    profile_density : ndarray
        Array of shape (..., nsep_in), containing densities in [fraction / mas^2]
        (or any density per radial-annulus area unit).
    separation_in : 1D ndarray
        Input radial centers [mas].
    pxscale_out : float
        Desired output radial sampling [mas].
        Only used if separation_out is None.
    separation_out : 1D ndarray or None
        Output radial centers [mas]. If None, builds a grid
        0, pxscale_out, 2*pxscale_out, ...

    Returns
    -------
    separation_out : 1D ndarray
        Output radial centers [mas].
    profile_density_out : ndarray
        Rebinned density array of shape (..., nsep_out).
    """
    profile_density = np.asarray(profile_density, dtype=float)
    separation_in = np.asarray(separation_in, dtype=float)

    if profile_density.shape[-1] != len(separation_in):
        raise ValueError("Last axis of profile_density must match len(separation_in).")
    if pxscale_out <= 0:
        raise ValueError("pxscale_out must be > 0.")

    edges_in = radial_centers_to_edges(separation_in)

    if separation_out is None:
        rmax_in = edges_in[-1]
        # same logic as "same or larger FoV": keep all flux, possibly with a partially empty last bin
        n_out = int(np.ceil(rmax_in / pxscale_out + 0.5))
        separation_out = np.arange(n_out, dtype=float) * pxscale_out
    else:
        separation_out = np.asarray(separation_out, dtype=float)

    edges_out = radial_centers_to_edges(separation_out)
    area_out = np.pi * (edges_out[1:]**2 - edges_out[:-1]**2)

    overlap_area = _overlap_annulus_area_matrix(edges_in, edges_out)

    # Exact annular-overlap rebinning
    profile_density_out = np.einsum(
        "ij,...j->...i",
        overlap_area,
        profile_density,
        optimize=True,
    ) / area_out

    return separation_out, profile_density_out



def get_total_flux_from_PSF_profile_density(separation, PSF_profile_density_1D):
    """
    separation : 1D array [mas], radial coordinates of the profile samples
    PSF_profile_density_1D : 1D array [flux fraction / mas^2]

    Returns
    -------
    total_flux : float
        Integrated flux fraction.
    """
    r = np.asarray(separation, dtype=float)
    I = np.asarray(PSF_profile_density_1D, dtype=float)

    if r.ndim != 1 or I.ndim != 1:
        raise ValueError("separation and PSF_profile_density_1D must both be 1D arrays.")
    if len(r) != len(I):
        raise ValueError("separation and PSF_profile_density_1D must have the same length.")
    if np.any(np.diff(r) < 0):
        raise ValueError("separation must be sorted in increasing order.")

    edges = radial_centers_to_edges(r)
    
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    total_flux = np.nansum(I * area)
    return total_flux



#
# Parameters
#
instru        = "PCS"
strehl        = "Q2"
apodizer      = "NO_SP"
coronagraphs  = [None, "LYOT"] 
order_PSF     = 1 # 1 is the best choice
model_PSF     = "gaussian"
sigfactor     = None
debug         = False

# Specs
size_core = 2      # [px]
sep_unit  = "mas"  # [mas]
D         = 38.54  # [m]
WFE_ref   = 50.026 # [nm RMS]
IWA_ref   = 33.4/2 # [mas] ANDES Lyot FPM radius (cf. plot de Mamadou N'Diaye)



# --- Retrieving PCS data ---
PSF_profile_density_no_coro_3D = fits.getdata("data/PCS/PSF_simulations/profiles_psf_Q2.fits")        # (wave, mag_star, sep) [fraction/mas**2]      (fraction = flux without coronagraph per bin     / total flux without coronagraph) (off-axis)
PSF_profile_density_coro_3D    = fits.getdata("data/PCS/PSF_simulations/profiles_coro_Q2.fits")       # (wave, mag_star, sep) [fraction/mas**2]      (fraction = flux with    coronagraph per bin     / total flux with    coronagraph) (on-axis)
fraction_core_no_coro_2D       = fits.getdata("data/PCS/PSF_simulations/encircled_energy_Q2.fits")    # (wave, mag_star)      [fraction/FWHM/mas**2] (fraction = flux without coronagraph inside core / total flux without coronagraph) (off-axis)
star_transmission_2D           = fits.getdata("data/PCS/PSF_simulations/corono_transmission_Q2.fits") # (wave, mag_star)      star coronagraphic transmission (=  total flux with coronagraph / total flux without coronagraph)         (on-axis)



# --- Retrieving PCS data axis ---
PSF_header   = fits.getheader("data/PCS/PSF_simulations/profiles_coro_Q2.fits")

# wave axis
N_wave       = PSF_header["NAXIS3"]
wave_min     = PSF_header["WAVEMIN"]                        # [m]
wave_max     = PSF_header["WAVEMAX"]                        # [m]
dwave        = PSF_header["WAVESTEP"]                       # [m]
wave         = (wave_min + np.arange(N_wave) * dwave) * 1e6 # µm]
cmap_wave    = plt.get_cmap("Spectral_r", N_wave)

# mag_star axis
N_mag_star   = PSF_header["NAXIS2"]
mag_star_min = PSF_header["MAGMIN"]                             # [no unit]
mag_star_max = PSF_header["MAGMAX"]                             # [no unit]
dmag_star    = PSF_header["MAGSTEP"]                            # [no unit]
mag_star     = mag_star_min + np.arange(N_mag_star) * dmag_star # [no unit]

# separation_2D axis (originally in lambda/D units)
N_sep_loD     = PSF_header["NAXIS1"]
sep_loD       = np.arange(N_sep_loD) / PSF_header['SAMPLING'] # (sep)       [l/D]
loD           = wave*1e-6 / D * rad2arcsec * 1000             # (wave)      [mas]
separation_2D = sep_loD[None, :] * loD[:, None]               # (wave, sep) [mas]
pxscales      = 1/PSF_header['SAMPLING'] * loD                # (wave)      [mas]



# Rebinning the PSF profiles
pxscales_new                    = loD / size_core # (wave) [mas] Effective detector spatial sampling
PSF_profile_density_no_coro_new = []
PSF_profile_density_coro_new    = []
separation_new_list             = []

for iw in range(N_wave):
    sep_in = separation_2D[iw] # [mas]
    dr_out = pxscales_new[iw]  # [mas]

    sep_out, prof_no_coro = rebin_radial_profile_density(profile_density=PSF_profile_density_no_coro_3D[iw], separation_in=sep_in, pxscale_out=dr_out)
    _,       prof_coro    = rebin_radial_profile_density(profile_density=PSF_profile_density_coro_3D[iw],    separation_in=sep_in, pxscale_out=dr_out)

    separation_new_list.append(sep_out)
    PSF_profile_density_no_coro_new.append(prof_no_coro)
    PSF_profile_density_coro_new.append(prof_coro)

PSF_profile_density_no_coro_3D = np.stack(PSF_profile_density_no_coro_new, axis=0)
PSF_profile_density_coro_3D    = np.stack(PSF_profile_density_coro_new, axis=0)
separation_2D                  = np.stack(separation_new_list, axis=0)
pxscales                       = pxscales_new



# WFE, IWA, wavelength and separation axis
lmin       = wave[0]      # [µm]
lmax       = wave[-1]     # [µm]
sep_max    = 1_000        # [mas]
WFE_min    = 10           # [nm] RMS
WFE_max    = 200          # [nm] RMS
IWA_min    = 1            # [mas]
IWA_max    = 100          # [mas]

WFE        = np.linspace(WFE_min,           WFE_max,           num=20)
IWA        = np.logspace(np.log10(IWA_min), np.log10(IWA_max), num=20)
separation = np.logspace(np.log10(1),       np.log10(sep_max), num=1_000)

WFE        = add_if_necessary(array=WFE,        value=WFE_ref)
IWA        = add_if_necessary(array=IWA,        value=IWA_ref)
separation = add_if_necessary(array=separation, value=0.0)

r_WFE = WFE / WFE_ref
r_IWA = IWA / IWA_ref

# r_WFE = np.array([1.0])
# r_IWA = np.array([1.0])

N_coro = len(coronagraphs)
N_WFE  = len(r_WFE)
N_IWA  = len(r_IWA)
N_sep  = len(separation)



# --- Estimating the DL component ---
# Perfect CORONO           => I_c = I_speck
# I_nc = SR*I_DL + I_speck => SR*I_DL = I_nc - I_speck = I_nc - I_c
PSF_profile_density_speck_3D = PSF_profile_density_coro_3D*star_transmission_2D[:, :, None]
SR_PSF_profile_density_DL_3D = PSF_profile_density_no_coro_3D - PSF_profile_density_speck_3D
# Renormalizing I_DL (in order to have the same total flux)
PSF_profile_density_DL_3D = np.zeros_like(SR_PSF_profile_density_DL_3D)
for iw in range(N_wave):
    for im in range(N_mag_star):
        # assuming total flux I_DL ~ total flux I_nc
        flux_DL    = get_total_flux_from_PSF_profile_density(separation=separation_2D[iw], PSF_profile_density_1D=PSF_profile_density_no_coro_3D[iw, im])
        flux_SR_DL = get_total_flux_from_PSF_profile_density(separation=separation_2D[iw], PSF_profile_density_1D=SR_PSF_profile_density_DL_3D[iw, im])
        PSF_profile_density_DL_3D[iw, im] = SR_PSF_profile_density_DL_3D[iw, im] * flux_DL / flux_SR_DL
# The DL profile does not depend on the star magnitude
PSF_profile_density_DL_2D = np.nanmean(PSF_profile_density_DL_3D, axis=1)



# --- Estimating the SR and the WFE (with the DL profiles) ---

r_core    = 1.029 * loD / 2         # (wave) DL FWHM radius [mas]

SR_2D     = np.zeros((N_wave, N_mag_star))
SR_raw_2D = np.zeros((N_wave, N_mag_star))
WFE_nm_1D = np.zeros((N_mag_star))

for iw in range(N_wave):
    mask_core = (separation_2D[iw] <= r_core[iw]) # FWHM mask
    for im in range(N_mag_star):
        x                 = PSF_profile_density_DL_2D[iw][mask_core].ravel()
        y                 = PSF_profile_density_no_coro_3D[iw, im][mask_core].ravel()
        A                 = np.column_stack([x, np.ones_like(x)])
        w                 = x / np.nanmax(x)
        W                 = np.sqrt(w)
        A_w               = A * W[:, None]
        y_w               = y * W
        res               = lsq_linear(A_w, y_w, bounds=([0.0, 0.0], [1.0, np.inf]))
        alpha, beta       = res.x
        alpha             = max(alpha, 1e-8)
        SR_raw_2D[iw, im] = min(alpha, 1.0)

for im in range(N_mag_star):
    WFE_rad_raw   = np.sqrt(-np.log(SR_raw_2D[:, im]))
    WFE_nm_1D[im] = np.nanmedian(1e3*wave / (2*np.pi) * WFE_rad_raw)
    WFE_rad       = WFE_nm_1D[im] / (1e3*wave / (2*np.pi))
    SR_2D[:, im]  = np.exp(-WFE_rad**2)
WFE_nm = np.nanmedian(WFE_nm_1D)            

cmap_mag = plt.get_cmap("inferno_r", N_mag_star)
plt.figure(figsize=(10, 6), dpi=300)
for im in range(N_mag_star):
    plt.plot(wave, 100*SR_2D[:, im],     c=cmap_mag(im), label=f"K = {mag_star[im]:.1f} (WFE = {WFE_nm_1D[im]:.1f} nm RMS)")
    plt.plot(wave, 100*SR_raw_2D[:, im], c=cmap_mag(im), ls="--")
plt.legend(loc="lower right", fontsize=14, fancybox=True, shadow=True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.title(f"Strehl Ratio for {instru} with {apodizer.replace('_', ' ')}-apodizer in {strehl}-strehl", fontsize=14)
plt.xlabel("Wavelength [µm]", fontsize=12)
plt.ylabel("SR [%]",          fontsize=12)
plt.axvspan(wave[0], wave[-1], color="black", alpha=0.1, lw=0, label="Simulated data range")
plt.xlim(wave[0], wave[-1])
plt.ylim(0, 100)
plt.show()



# --- Estimating the DL core fractions ---
fraction_core_DL_2D = fraction_core_no_coro_2D / SR_2D
# The DL profile does not depend on the star magnitude
fraction_core_DL_1D = np.nanmean(fraction_core_DL_2D, axis=1)



# --- Adding the separation axis for core fractions ---
fraction_core_no_coro_3D = fraction_core_no_coro_2D[:, :, None] * np.ones((1, 1, len(separation)))
fraction_core_DL_2D      = fraction_core_DL_1D[:, None]         * np.ones((1, len(separation)))



# --- Retrieving the coronagraphic profiles (from the ANDES coronagraphic profiles shapes) ---

# ANDES coronagraphic PSF data ranges (HARD CODED)
l0_min_PSF     = 0.6          # [µm]
l0_max_PSF     = 3.0          # [µm]
sep_max_PSF    = 1_000        # [mas]
WFE_min_PSF    = 10           # [nm]
WFE_max_PSF    = 500          # [nm]
IWA_min_PSF    = 0.1          # [mas]
IWA_max_PSF    = 100          # [mas]
instru_PSF     = "ANDES"
suffix_PSF     = f"NO_SP_LYOT_MED_5.0_{l0_min_PSF}_{l0_max_PSF}_{WFE_min_PSF}_{WFE_max_PSF}_{IWA_min_PSF}_{IWA_max_PSF}_{sep_max_PSF}"
ROOT           = Path(__file__).resolve().parent
instru_dir_PSF = ROOT / "data" / f"{instru_PSF}"
psf_dir        = instru_dir_PSF / "PSF_simulations"

# Opening data
wave0                         = fits.getdata(psf_dir / f"{instru_PSF}_wave_{suffix_PSF}.fits")                   # (wave0)       [µm]
WFE0                          = fits.getdata(psf_dir / f"{instru_PSF}_WFE_{suffix_PSF}.fits")                    # (WFE0)        [nm]
IWA0                          = fits.getdata(psf_dir / f"{instru_PSF}_IWA_{suffix_PSF}.fits")                    # (IWA0)        [mas]
separation0                   = fits.getdata(psf_dir / f"{instru_PSF}_separation_{suffix_PSF}.fits")             # (separation0) [mas]
fraction_core_coro_ANDES0_4D  = fits.getdata(psf_dir / f"{instru_PSF}_fraction_core_4D_{suffix_PSF}.fits")       # (wave0, WFE0, IWA0, separation0) [planet flux fraction/FWHM]
radial_transmission_ANDES0_4D = fits.getdata(psf_dir / f"{instru_PSF}_radial_transmission_4D_{suffix_PSF}.fits") # (wave0, WFE0, IWA0, separation0) coronagraphic transmission

# Interpolating over current grid (wave, separation) t WFE_ref and IWA_ref
pts                          = np.stack((np.meshgrid(wave, WFE_ref, IWA_ref, separation, indexing="ij")), axis=-1)
fraction_core_coro_ANDES_2D  = RegularGridInterpolator((wave0, WFE0, IWA0, separation0), fraction_core_coro_ANDES0_4D,  bounds_error=True, fill_value=np.nan)(pts)[:, 0, 0, :] # (l0, sep) [planet flux fraction/FWHM]
radial_transmission_ANDES_2D = RegularGridInterpolator((wave0, WFE0, IWA0, separation0), radial_transmission_ANDES0_4D, bounds_error=True, fill_value=np.nan)(pts)[:, 0, 0, :] # (l0, sep) coronagraphic transmission

# Getting coronagraphic fraction core
fraction_core_coro_ANDES_1D_off_axis = fraction_core_coro_ANDES_2D[:, -1]                                                   # (wave) With coronagraph, the off-axis fraction core is assumed to be given at the largest offset, i.e. sep = -1
shape_fraction_core_coro_2D          = fraction_core_coro_ANDES_2D / fraction_core_coro_ANDES_1D_off_axis[:, None]          # (wave, separation)
fraction_core_coro_2D_off_axis       = fraction_core_no_coro_3D[:, :, -1]                                                   # (wave, mag_star)
fraction_core_coro_3D                = shape_fraction_core_coro_2D[:, None, :] * fraction_core_coro_2D_off_axis[:, :, None] # (wave, mag_star, separation)

# Getting coronagraphic radial transmission (we assume the same ANDES pupil Lyot stop mask => the off axis transmission will be the same)
radial_transmission_ANDES_1D_on_axis  = radial_transmission_ANDES_2D[:, 0]  # (wave) With coronagraph, the on-axis coronagraphic is assumed to be given at sep = 0
radial_transmission_ANDES_1D_off_axis = radial_transmission_ANDES_2D[:, -1] # (wave) With coronagraph, the off-axis fraction core is assumed to be given at the largest offset, i.e. sep = -1
radial_transmission_2D_on_axis        = star_transmission_2D # (wave, mag_star)
radial_transmission_3D                = np.zeros((N_wave, N_mag_star, N_sep))
for im in range(N_mag_star):
    # Anchors 
    # y0     = reference anchor after IWA stretch
    # y1     = off-axis asymptote
    # y0_new = target on-axis transmission
    y0     = radial_transmission_ANDES_1D_on_axis  # (wave) reference anchor
    y1     = radial_transmission_ANDES_1D_off_axis # (wave) off-axis asymptote
    y0_new = radial_transmission_2D_on_axis[:, im] # (wave)
    # Affine transform in logit space:
    #   logit(T_new) = a * logit(T_ref) + b
    # constrained by:
    #   T_new(small sep) = y0_new
    #   T_new(large sep) = y1
    L0     = logit(y0)                   # (wave)
    L1     = logit(y1)                   # (wave)
    L0new  = logit(y0_new)               # (wave)
    L1new  = L1                          # (wave) the radial transmission does not change far from the coronagraph
    a      = (L1new - L0new) / (L1 - L0) # (wave)
    b      = L0new - a * L0              # (wave)
    radial_transmission_3D[:, im] = expit(a[:, None] * logit(radial_transmission_ANDES_2D) + b[:, None])
    


# --- INTERPOLATION AND EXTRAPOLATION ALONG SEPARATION ---
PSF_profile_density_DL_2D_new      = np.zeros((N_wave, N_sep))
PSF_profile_density_no_coro_3D_new = np.zeros((N_wave, N_mag_star, N_sep))
PSF_profile_density_coro_3D_new    = np.zeros((N_wave, N_mag_star, N_sep))
PSF_profile_density_speck_3D_new   = np.zeros((N_wave, N_mag_star, N_sep))
for iw in range(N_wave):
    PSF_profile_density_DL_2D_new[iw] = interp_extrap_sep(separation_ref=separation_2D[iw], separation_new=separation, y_ref=PSF_profile_density_DL_2D[iw], mode="log", tail_model="powerlaw")
    for im in range(N_mag_star):
        PSF_profile_density_no_coro_3D_new[iw, im] = interp_extrap_sep(separation_ref=separation_2D[iw], separation_new=separation, y_ref=PSF_profile_density_no_coro_3D[iw, im], mode="log", tail_model="powerlaw")
        PSF_profile_density_coro_3D_new[iw, im]    = interp_extrap_sep(separation_ref=separation_2D[iw], separation_new=separation, y_ref=PSF_profile_density_coro_3D[iw, im],    mode="log", tail_model="powerlaw")
        PSF_profile_density_speck_3D_new[iw, im]   = interp_extrap_sep(separation_ref=separation_2D[iw], separation_new=separation, y_ref=PSF_profile_density_speck_3D[iw, im],   mode="log", tail_model="powerlaw")
PSF_profile_density_DL_2D      = PSF_profile_density_DL_2D_new
PSF_profile_density_no_coro_3D = PSF_profile_density_no_coro_3D_new
PSF_profile_density_coro_3D    = PSF_profile_density_coro_3D_new
PSF_profile_density_speck_3D   = PSF_profile_density_speck_3D_new





plot_profiles(instru=instru, coronagraph="LYOT", apodizer=apodizer, strehl=strehl, pxscale=None, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=None, separation=separation, PSF_profile_density_2D=PSF_profile_density_coro_3D[:, 0], fraction_core_2D=fraction_core_coro_3D[:, 0], radial_transmission_2D=radial_transmission_3D[:, 0], type_PSF="post-AO", title_suffix="\n")





# %%
# --- VARYING AO PERFORMANCE (r_WFE) AND CORONAGRAPH IWA (r_IWA) ---
# Output convention:
#   4D temporary arrays: (wave, WFE, IWA, separation)
#   5D final arrays:     (wave, WFE, IWA, mag_star, separation)

PSF_profile_density_no_coro_5D = np.zeros((N_wave, N_WFE, N_IWA, N_mag_star, N_sep))
fraction_core_no_coro_5D       = np.zeros((N_wave, N_WFE, N_IWA, N_mag_star, N_sep))
PSF_profile_density_coro_5D    = np.zeros((N_wave, N_WFE, N_IWA, N_mag_star, N_sep))
fraction_core_coro_5D          = np.zeros((N_wave, N_WFE, N_IWA, N_mag_star, N_sep))
radial_transmission_5D         = np.zeros((N_wave, N_WFE, N_IWA, N_mag_star, N_sep))

eps        = 1e-8
safe01     = lambda x: np.clip(x, eps, 1 - eps)
points_ref = np.stack(np.meshgrid(wave, WFE_ref, IWA_ref, separation, indexing="ij"), axis=-1)
r          = separation
dr         = np.gradient(r)
FWHM       = 1.029 * wave * 1e-6 / D * rad2arcsec * 1000  # (wave) [mas]
sigma      = FWHM / (2 * np.sqrt(2 * np.log(2)))          # (wave) [mas]
sigma_bins = np.nanmedian(sigma[:, None] / dr[None, :], axis=1)

for im in tqdm(range(N_mag_star), desc="VARYING AO PERFORMANCE (r_WFE) AND CORONAGRAPH IWA (r_IWA)"):

    # -------------------------------------------------------------------------
    # 1) Build 4D PSF grid for the current stellar magnitude
    # -------------------------------------------------------------------------
    PSF_profile_density_no_coro_4D, fraction_core_no_coro_4D, PSF_profile_density_coro_4D, fraction_core_coro_4D, radial_transmission_4D = build_PSF_grid(SR=SR_2D[:, im], D=D, wave=wave, r_WFE=r_WFE, r_IWA=r_IWA, separation=separation, PSF_profile_density_no_coro_2D=PSF_profile_density_no_coro_3D[:, im], fraction_core_no_coro_2D=fraction_core_no_coro_3D[:, im], PSF_profile_density_DL_2D=None, fraction_core_DL_2D=fraction_core_DL_2D, coronagraph="LYOT", IWA_ref=IWA_ref, PSF_profile_density_coro_2D=PSF_profile_density_coro_3D[:, im], fraction_core_coro_2D=fraction_core_coro_3D[:, im], radial_transmission_2D=radial_transmission_3D[:, im], PSF_profile_density_speck_2D=PSF_profile_density_speck_3D[:, im])

    # -------------------------------------------------------------------------
    # 2) Forcing nominal equality
    # -------------------------------------------------------------------------

    # On PSF_profile_density (no_coro)
    PSF_profile_density_no_coro_OG_2D = np.copy(PSF_profile_density_no_coro_3D[:, im])
    PSF_profile_density_no_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), PSF_profile_density_no_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_PSF_no_coro           = PSF_profile_density_no_coro_OG_2D / PSF_profile_density_no_coro_EX_2D
    PSF_profile_density_no_coro_4D   *= corr_factor_PSF_no_coro[:, None, None, :]

    # On fraction_core (no_coro)
    fraction_core_no_coro_OG_2D = np.copy(fraction_core_no_coro_3D[:, im])
    fraction_core_no_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), fraction_core_no_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_FC_no_coro      = fraction_core_no_coro_OG_2D / fraction_core_no_coro_EX_2D
    fraction_core_no_coro_4D   *= corr_factor_FC_no_coro[:, None, None, :]

    # On PSF_profile_density (coro)
    PSF_profile_density_coro_OG_2D = np.copy(PSF_profile_density_coro_3D[:, im])
    PSF_profile_density_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), PSF_profile_density_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_PSF_coro           = PSF_profile_density_coro_OG_2D / PSF_profile_density_coro_EX_2D
    PSF_profile_density_coro_4D   *= corr_factor_PSF_coro[:, None, None, :]

    # On fraction_core (coro)
    fraction_core_coro_OG_2D = np.copy(fraction_core_coro_3D[:, im])
    fraction_core_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), fraction_core_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_FC_coro      = fraction_core_coro_OG_2D / fraction_core_coro_EX_2D
    fraction_core_coro_4D   *= corr_factor_FC_coro[:, None, None, :]

    # On radial_transmission (coro)
    radial_transmission_OG_2D = np.copy(radial_transmission_3D[:, im])
    radial_transmission_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), radial_transmission_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_RT_coro       = radial_transmission_OG_2D / radial_transmission_EX_2D
    radial_transmission_4D   *= corr_factor_RT_coro[:, None, None, :]

    # -------------------------------------------------------------------------
    # 3) Empirical consistency checks and monotonicity constraints
    # -------------------------------------------------------------------------

    # 3.1) The non-coronagraphic core fraction must not exceed the empirical
    #      estimate derived from the PSF profile.
    fraction_core_no_coro_3D_from_PSF = PSF_profile_density_no_coro_4D[:, :, :, 0] * pxscales[:, None, None]**2 * size_core**2
    fraction_core_no_coro_4D          = safe01(np.minimum(fraction_core_no_coro_4D, fraction_core_no_coro_3D_from_PSF[:, :, :, None]))

    # 3.2) The coronagraphic core fraction must not exceed the non-coronagraphic one.
    fraction_core_coro_4D = safe01(np.minimum(fraction_core_coro_4D, fraction_core_no_coro_4D))

    # 3.3) Monotonicity with IWA:
    #      for a larger IWA, both the coronagraphic core fraction and the radial
    #      transmission are assumed to be no better than for a smaller IWA.
    fraction_core_coro_4D  = safe01(np.minimum.accumulate(fraction_core_coro_4D, axis=2))
    radial_transmission_4D = safe01(np.minimum.accumulate(radial_transmission_4D, axis=2))

    # 3.4) Monotonicity with separation:
    #      both the coronagraphic core fraction and the radial transmission are
    #      assumed to increase monotonically with angular separation.
    fraction_core_coro_4D  = safe01(np.maximum.accumulate(fraction_core_coro_4D, axis=3))
    radial_transmission_4D = safe01(np.maximum.accumulate(radial_transmission_4D, axis=3))

    # -------------------------------------------------------------------------
    # 4) On-axis coronagraphic core-fraction consistency check
    # -------------------------------------------------------------------------

    fraction_core_coro_3D_from_PSF = safe01(PSF_profile_density_coro_4D[:, :, :, 0] * pxscales[:, None, None]**2 * size_core**2)
    y0_3D                          = safe01(fraction_core_coro_4D[:, :, :, 0])
    y0_new_3D                      = safe01(np.minimum(y0_3D, fraction_core_coro_3D_from_PSF))
    y1_3D                          = safe01(fraction_core_coro_4D[:, :, :, -1])

    for idx_WFE in range(N_WFE):
        for idx_IWA in range(N_IWA):
            y0     = y0_3D[:, idx_WFE, idx_IWA]
            y0_new = y0_new_3D[:, idx_WFE, idx_IWA]
            y1     = y1_3D[:, idx_WFE, idx_IWA]
            L0     = logit(y0)
            L1     = logit(y1)
            L0new  = logit(y0_new)
            den_a  = L1 - L0
            valid  = np.isfinite(L0) & np.isfinite(L1) & np.isfinite(L0new) & (np.abs(den_a) > eps)
            if not np.any(valid):
                continue
            prof = safe01(fraction_core_coro_4D[:, idx_WFE, idx_IWA, :])
            a    = (L1[valid] - L0new[valid]) / den_a[valid]
            b    = L0new[valid] - a * L0[valid]
            fraction_core_coro_4D[valid, idx_WFE, idx_IWA, :] = expit(a[:, None] * logit(prof[valid]) + b[:, None])

    fraction_core_coro_4D = safe01(np.minimum(fraction_core_coro_4D, fraction_core_no_coro_4D))
    fraction_core_coro_4D = safe01(np.minimum.accumulate(fraction_core_coro_4D, axis=2))
    fraction_core_coro_4D = safe01(np.maximum.accumulate(fraction_core_coro_4D, axis=3))

    # -------------------------------------------------------------------------
    # 5) Smoothing the coronagraphic core fraction
    # -------------------------------------------------------------------------
    for iw in range(N_wave):
        fraction_core_coro_4D[iw] = gaussian_filter1d(fraction_core_coro_4D[iw], sigma=sigma_bins[iw], axis=2)
    
    fraction_core_coro_4D = safe01(np.minimum(fraction_core_coro_4D, fraction_core_no_coro_4D))
    fraction_core_coro_4D = safe01(np.minimum.accumulate(fraction_core_coro_4D, axis=2))
    fraction_core_coro_4D = safe01(np.maximum.accumulate(fraction_core_coro_4D, axis=3))

    # -------------------------------------------------------------------------
    # 6) Store corrected 4D products into the final 5D arrays
    # -------------------------------------------------------------------------
    PSF_profile_density_no_coro_5D[:, :, :, im, :] = PSF_profile_density_no_coro_4D
    fraction_core_no_coro_5D[:, :, :, im, :]       = fraction_core_no_coro_4D
    PSF_profile_density_coro_5D[:, :, :, im, :]    = PSF_profile_density_coro_4D
    fraction_core_coro_5D[:, :, :, im, :]          = fraction_core_coro_4D
    radial_transmission_5D[:, :, :, im, :]         = radial_transmission_4D




# --- Plotting + saving ---
for coronagraph in coronagraphs:

    # --- Assigning the profiles ---
    if coronagraph is None:
        PSF_profile_density_5D = PSF_profile_density_no_coro_5D
        fraction_core_5D       = fraction_core_no_coro_5D
    else:
        PSF_profile_density_5D = PSF_profile_density_coro_5D
        fraction_core_5D       = fraction_core_coro_5D    
    
    # --- PLOT ---
    idx_MAG = np.abs(mag_star - 0).argmin() # at mag_star = 0.0
    idx_IWA = np.abs(r_IWA - 1).argmin()    # at r_IWA = 1.0
    #for idx_WFE in range(N_WFE):
    for idx_WFE in [0, np.abs(r_WFE - 1).argmin(), -1]:
        psf    = PSF_profile_density_5D[:, idx_WFE, idx_IWA, idx_MAG]
        fc     = fraction_core_5D[:, idx_WFE, idx_IWA, idx_MAG]
        if coronagraph is not None:
            rt = radial_transmission_5D[:, idx_WFE, idx_IWA, idx_MAG]
        else:
            rt = None
        plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=None, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=None, separation=separation, PSF_profile_density_2D=psf, fraction_core_2D=fc, radial_transmission_2D=rt, type_PSF="post-AO", title_suffix=f" \n WFE = {WFE_ref*r_WFE[idx_WFE]:.1f}nm | IWA = {IWA_ref*r_IWA[idx_IWA]:.1f}mas | $m_\star$ = {mag_star[idx_MAG]:.1f}")
    
    if coronagraph is not None:
        idx_MAG = np.abs(mag_star - 0).argmin() # at mag_star = 0.0
        idx_WFE = np.abs(r_WFE - 1).argmin()    # at r_WFE = 1.0
        #for idx_IWA in range(N_IWA):
        for idx_IWA in [0, np.abs(r_IWA - 1).argmin(), -1]:
            psf    = PSF_profile_density_5D[:, idx_WFE, idx_IWA, idx_MAG]
            fc     = fraction_core_5D[:, idx_WFE, idx_IWA, idx_MAG]
            if coronagraph is not None:
                rt = radial_transmission_5D[:, idx_WFE, idx_IWA, idx_MAG]
            else:
                rt = None
            plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=None, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=None, separation=separation, PSF_profile_density_2D=psf, fraction_core_2D=fc, radial_transmission_2D=rt, type_PSF="post-AO", title_suffix=f" \n WFE = {WFE_ref*r_WFE[idx_WFE]:.1f}nm | IWA = {IWA_ref*r_IWA[idx_IWA]:.1f}mas | $m_\star$ = {mag_star[idx_MAG]:.1f}")

    

    # --- SAVING ---
    suffix_PSF = f"{apodizer}_{coronagraph}_{strehl}_{lmin}_{lmax}_{WFE_min}_{WFE_max}_{IWA_min}_{IWA_max}_{mag_star_min}_{mag_star_max}_{sep_max}"
    fits.writeto(f"data/{instru}/PSF_simulations/{instru}_wave_{suffix_PSF}.fits",                       wave,                   overwrite=True)
    fits.writeto(f"data/{instru}/PSF_simulations/{instru}_WFE_{suffix_PSF}.fits",                        WFE,                    overwrite=True)
    fits.writeto(f"data/{instru}/PSF_simulations/{instru}_IWA_{suffix_PSF}.fits",                        IWA,                    overwrite=True)
    fits.writeto(f"data/{instru}/PSF_simulations/{instru}_mag_star_{suffix_PSF}.fits",                   mag_star,               overwrite=True)
    fits.writeto(f"data/{instru}/PSF_simulations/{instru}_separation_{suffix_PSF}.fits",                 separation,             overwrite=True)
    fits.writeto(f"data/{instru}/PSF_simulations/{instru}_PSF_profile_density_5D_{suffix_PSF}.fits",     PSF_profile_density_5D, overwrite=True)
    fits.writeto(f"data/{instru}/PSF_simulations/{instru}_fraction_core_5D_{suffix_PSF}.fits",           fraction_core_5D,       overwrite=True)
    if coronagraph is not None:
        fits.writeto(f"data/{instru}/PSF_simulations/{instru}_radial_transmission_5D_{suffix_PSF}.fits", radial_transmission_5D, overwrite=True)






















