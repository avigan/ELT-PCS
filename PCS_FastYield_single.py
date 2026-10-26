import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["NUMBA_NUM_THREADS"]    = "1"

# import FastYield modules
# import fastyield.config as config
# config.dpi_fig = 72

from fastyield.config import rad2arcsec, h, c, R0_min, R0_max, lmin_bands, lmax_bands, archive_path, simulated_path, get_sim_data_path, dpi_fig
from fastyield.get_specs import load_tell_trans, get_detector_specs
from fastyield.FastYield import load_planet_table, get_mask_planet_type, yield_population_plot, yield_heatmap_ELT
from fastyield.FastYield_helpers import print_simulation_summary, make_suffix, write_meta, memmap_nbytes, format_nbytes, create_memmap_with_log
from fastyield.spectrum import get_wavelength_axis_constant_R, Spectrum, load_vega_spectrum, get_spectrum_contribution_name_model, get_wave_K, get_counts_from_density

# import astropy modules
from astropy.io import fits

# import numpy modules
import numpy as np

# import matplotlib modules
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator
from matplotlib.lines import Line2D
import matplotlib as mpl

# import scipy modules
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logit, expit

# import other modules
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import cpu_count
from pathlib import Path
import sys
import gc

import PCS_FastYield_helpers as fyh


# Parameters for multiprocessing
nproc     = max(cpu_count()-3, 1)
chunksize = 8


# %%
def main():

    # ---------------------------
    # Parameters
    # ---------------------------
    PCS_CODE_VERSION = "PCS_FastYield_v1.0"
    dtype            = np.float32

    # --- Instrument specs (fixed) ---
    instru     = "PCS"     # Instrument's name
    D          = 38.54     # [m]  Telescope diameter
    S          = 980.      # [m2] Telescope collective surface
    N_mirror   = 5         # Number of mirror at the ELT (in order to compute the telescope transmission)
    trans_dust = 0.90      # effect of dust, from common ICD, section 4.11, p37 (Document Number: ESO-253082)

    # --- Detector mode ---
    # "PCS_visNIR_conservative"
    #     Conservative detector setting:
    #         visible/red optical : VIS_CCD (λ<0.8 µm)
    #         NIR                 : H4RG
    #
    # "PCS_low_noise_visNIR"
    #     More optimistic, low-noise detector setting:
    #         visible/red optical : EMCCD (λ<0.8 µm)
    #         NIR                 : H4RG
    #
    # "PCS_NIR_APD_optimistic"
    #     Technology-study mode:
    #         visible/red optical : EMCCD-like detector
    #         NIR                 : SAPHIRA / LmAPD-like detector
    #     Not a baseline science-detector prescription because current SAPHIRA-like
    #     arrays are small-format and mainly used for fast NIR applications.
    detector = "PCS_visNIR_conservative"

    # --- Paths ---
    instru_dir = get_sim_data_path() / instru
    sim_dir    = instru_dir / "FastYield_simulations"
    psf_dir    = instru_dir / "PSF_simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Definition of the standard photometric bands
    # band | l0   | dl
    # --------------------------
    # V    | 0.55 | 0.09
    # R    | 0.66 | 0.14
    # I    | 0.81 | 0.15
    # z    | 0.90 | 0.15
    # Y    | 1.02 | 0.12
    # J    | 1.22 | 0.21
    # H    | 1.63 | 0.31
    # K    | 2.19 | 0.39

    # --- Instrument concept ---
    instru_type     = "IFU"    # Type of instrument ("IFU" or "imager")
    post_processing = "MM"     # Post-processing method ("MM" or "DI")
    R               = 80_000   # [dimensionless]
    l0              = 0.74     # [µm]
    Dl              = 0.48     # [µm]
    Nl_max          = None     #
    r_WFE           = 1.0      # [dimensionless]
    IWA             = 2.4      # [lambda0_band/D]
    trans_instru    = 0.25     # [dimensionless]
    sigma_m_p       = 0.01     # [%]
    FoV             = 300      # [mas]

    # --- General parameters for the simulation ---
    coronagraph        = "APLC"                                # "CLC", "APLC", or None; matches PCS SUMMARY FITS labels
    apodizer           = "NO_SP"                               # Shaped pupil mask (NO_SP => no mask)
    strehl             = "Q2"                                  # Sky atmospheric condition (1st quartile, 2nd, etc.)
    band_WFS           = "z"                                   # Photometric band of the stellar magnitude axis (WFS) in the PCS PSF simulations
    SNR_thr            = 5                                     # Detection threshold
    exposure_time      = 10*60                                 # Total exposure time per planet [mn]
    force_new_calc     = False                                  # Forcing new simulations calculations
    thermal_model      = "auto"                                # Model for the thermal spectrum of the planet ("auto", "None", "BT-Settl", "Exo-REM", "SONORA", "PICASO", "Saumon", etc.)
    reflected_model    = "auto"                                # Model for the albedo of the planet ("auto", "tellurics", "flat", "PICASO")
    spatial_sampling   = 2                                     # [px/FWHM] Number of pixel per spatial FWHM along 1 direction (spatial_sampling >= 2 => Nyquist spatial sampling)
    size_core          = 2                                     # Aperture size on which the signal is integrated [px]
    A_FWHM             = size_core**2                          # Number of pixel per FWHM box area
    Rc                 = 1_00                                  # MM cut-off resolution (Rc~100 is enough to reach ~1e-8 with speckles only, Rc~1000 would allows to go further (more conservative))
    filter_type        = "gaussian_fast"                       # MM filter type
    table_type         = "Archive"                             # "Archive": for all known exoplanets | "Simulated": TODO

    # --- Separation range ---
    sep_min = 0     # [mas]
    sep_max = 1_000 # [mas] max separation of the raw PSF data

    # --- Plots configuration ---

    # Band used for the contrast shown on the y-axis and for the light regime split (thermal/reflected)
    band_plot = "I"

    # Choose in wich space the planet populations are plotted
    # "obs"  : observationnal space (contrast vs. angular separation)
    # "phys" : physical space (mass vs. projected semi-major axis)
    # "both" : combine the plots in two panels
    planet_population_space = 'phys'

    # additional infos on plots
    show_golden_sample = False
    show_det_regions   = False
    show_ptype_regions = False

    # --- IFU parameters space to explore ---
    if instru_type == "IFU":
        # Arrays
        R                = np.array([R])
        l0               = np.array([l0])
        Dl               = np.array([Dl])
        r_WFE            = np.array([r_WFE])
        IWA              = np.array([IWA])
        trans_instru     = np.array([trans_instru])
        sigma_m          = np.array([sigma_m_p/100])
        FoV              = np.array([FoV])
        # Dimensions
        N_R              = len(R)
        N_l0             = len(l0)
        N_Dl             = len(Dl)
        N_r_WFE          = len(r_WFE)
        N_IWA            = len(IWA)
        N_TI             = len(trans_instru)
        N_SM             = len(sigma_m)
        N_FoV            = len(FoV)
        # Resolution for the raw models
        R_model          = R0_max

    # --- Imager parameters space to explore ---
    elif instru_type == "imager":
        # safety check
        if post_processing == "MM":
            raise ValueError("Molecular mapping not available for imager")
        # Arrays
        l0               = np.array(l0)
        Dl               = np.array(Dl)
        r_WFE            = np.array(r_WFE)
        IWA              = np.array(IWA)
        trans_instru     = np.array(trans_instru)
        sigma_m          = np.array(sigma_m_p/100)
        FoV              = np.array(FoV)
        # Dimensions
        N_l0             = len(l0)
        N_Dl             = len(Dl)
        N_r_WFE          = len(r_WFE)
        N_IWA            = len(IWA)
        N_TI             = len(trans_instru)
        N_SM             = len(sigma_m)
        N_FoV            = len(FoV)
        # Resolution for the raw models
        R_model          = R0_min

    # --- Detector assignment as a function of wavelength ---
    if detector.startswith("constant_"):
        detector_l0 = np.full(N_l0, detector.replace("constant_", "", 1))
    elif detector == "PCS_visNIR_conservative":
        detector_l0 = np.where(l0 < 0.80, "VIS_CCD", "H4RG")
    elif detector == "PCS_low_noise_visNIR":
        detector_l0 = np.where(l0 < 0.80, "EMCCD", "H4RG")
    elif detector == "PCS_NIR_APD_optimistic":
        detector_l0 = np.where(l0 < 0.80, "EMCCD", "SAPHIRA")
    else:
        raise ValueError("detector must be 'constant_H4RG', 'PCS_visNIR_conservative', 'PCS_low_noise_visNIR', or 'PCS_NIR_APD_optimistic'.")
    spec_dict = {name: get_detector_specs(name) for name in np.unique(detector_l0)}
    RON0         = np.array([spec_dict[d]["RON0"]         for d in detector_l0], dtype=dtype)
    RON_lim      = np.array([spec_dict[d]["RON_lim"]      for d in detector_l0], dtype=dtype)
    DC0          = np.array([spec_dict[d]["DC0"]          for d in detector_l0], dtype=dtype)
    saturation_e = np.array([spec_dict[d]["saturation_e"] for d in detector_l0], dtype=dtype)
    min_DIT      = np.array([spec_dict[d]["min_DIT"]      for d in detector_l0], dtype=dtype)
    max_DIT      = np.array([spec_dict[d]["max_DIT"]      for d in detector_l0], dtype=dtype)

    # --- Pixel scales (assuming Nyquist sampling) ---
    l0D          = l0*1e-6 / D * 1000*rad2arcsec  # lambda/D [mas/(lambda/D)]
    FWHM0        = 1.029 * l0D                    # [mas/FWHM]
    pxscales     = FWHM0 / spatial_sampling       # [mas/px]
    pxscales_rad = pxscales / (rad2arcsec * 1000) # [rad/px]

    # Residuals modulations considered for the post_processing
    residuals = "Systematics" if post_processing == "MM" else "Speckles"

    # Getting labels
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Valid spectral format windows
    if instru_type == "IFU":
        valid_windows_3D = fyh.get_valid_IFU_windows(R=R, l0=l0, Dl=Dl, Nl_max=Nl_max)
        if not np.any(valid_windows_3D):
            raise ValueError("No valid IFU configurations with the requested Nl_max.")
    else:
        valid_windows_3D = None

    # ---------------------------
    # Getting data
    # ---------------------------

    # Retrieving the planet table
    planet_table = load_planet_table(f"{table_type}_Pull_For_FastYield.ecsv")
    N_PT_raw     = len(planet_table)

    # Filtering separation range
    planet_table       = planet_table[(planet_table["AngSep"].value >= sep_min) & (planet_table["AngSep"].value <= sep_max)]
    separation_planets = planet_table["AngSep"].value # [mas]
    N_PT               = len(planet_table)
    print(f"\nKeeping {N_PT}/{N_PT_raw} planets between {sep_min:.0f} and {sep_max:.0f} mas in thermal and reflected light")

    if N_PT == 0:
        raise ValueError("No planets remain after the separation selection")

    # Prints simulation parameters
    print_simulation_summary(instru=instru, instru_type=instru_type, detector=detector, post_processing=post_processing, D=D, S=S, N_mirror=N_mirror, trans_dust=trans_dust, RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, thermal_model=thermal_model, reflected_model=reflected_model, SNR_thr=SNR_thr, exposure_time=exposure_time, size_core=size_core, A_FWHM=A_FWHM, Rc=Rc if instru_type == "IFU" else None, filter_type=filter_type if instru_type == "IFU" else None, table_type=table_type, sep_min=sep_min, sep_max=sep_max, N_PT_raw=N_PT_raw, N_PT=N_PT, force_new_calc=force_new_calc, l0=l0, R=R if instru_type == "IFU" else None, Dl=Dl, WFE=r_WFE, IWA=IWA, trans_instru=trans_instru, sigma_m=sigma_m, FoV=FoV)
    # The shared legacy summary helper calls its argument 'WFE'; it receives r_WFE
    # for API compatibility. Its WFE/IWA unit annotations are NOT authoritative here.
    print(f"PCS PSF axes (authoritative): r_WFE={r_WFE} [dimensionless]; "
          f"IWA={IWA} [lambda0_band/D]; mag_star=stellar V-band magnitude")

    # Build meta for YOUR case (IFU or imager)
    meta = dict(code_version=PCS_CODE_VERSION, dtype=str(dtype),

                # instrument / telescope
                instru=instru, D=D, S=S, N_mirror=N_mirror, trans_dust=trans_dust, detector=detector, detector_l0=detector_l0,

                # detector numeric specs that affect DIT/noise/products
                RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT,

                # science / pipeline
                coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, band_WFS=band_WFS, thermal_model=thermal_model, reflected_model=reflected_model, instru_type=instru_type, post_processing=post_processing, size_core=size_core, R_model=R_model, table_type=table_type,

                # exploration space
                sep_min=sep_min, sep_max=sep_max, N_PT=N_PT,

                # axis used
                l0=l0, Dl=Dl, r_WFE=r_WFE, IWA=IWA, trans_instru=trans_instru,
                )

    if instru_type == "IFU":
        meta.update(dict(Rc=Rc, filter_type=filter_type, R=R, Nl_max=Nl_max))
    suffix, meta_clean, _payload = make_suffix(meta, n=16)
    concept_name     = f'{instru_type}_{post_processing}'
    sim_dir_concept  = sim_dir / concept_name
    sim_dir_concept.mkdir(parents=True, exist_ok=True)
    write_meta(sim_dir_concept, suffix, meta_clean, suffix_as_prefix=True)
    print("\nFastYield data files suffix:", suffix)

    # File paths derived from suffix
    signal_path                 = sim_dir_concept / f"{suffix}_signal.npy"
    sigma_halo_2_path           = sim_dir_concept / f"{suffix}_sigma_halo_2.npy"
    sigma_bkg_2_path            = sim_dir_concept / f"{suffix}_sigma_bkg_2.npy"
    DIT_path                    = sim_dir_concept / f"{suffix}_DIT.npy"
    if instru_type == "IFU":
        sigma_syst_base_2_path  = sim_dir_concept / f"{suffix}_sigma_syst_base_2.npy"
    else:
        sigma_syst_base_2_path  = None
    PSF_profile_5D_tmp_path     = sim_dir_concept / f"{suffix}_tmp_PSF_profile_5D.npy"
    fraction_core_5D_tmp_path   = sim_dir_concept / f"{suffix}_tmp_fraction_core_5D.npy"
    PSF_profile_max_4D_tmp_path = sim_dir_concept / f"{suffix}_tmp_PSF_profile_max_4D.npy"

    for p in [PSF_profile_5D_tmp_path, fraction_core_5D_tmp_path, PSF_profile_max_4D_tmp_path]:
        if p.exists():
            p.unlink()

    do_new = force_new_calc or (not signal_path.exists()) or (not sigma_halo_2_path.exists()) or (not sigma_bkg_2_path.exists()) or (not DIT_path.exists())
    if instru_type == "IFU":
        do_new = do_new or (not sigma_syst_base_2_path.exists())



    ####################
    # Opening existing #
    ####################
    if not do_new:
        # For IFU (7D):    (Planets, R, l0, Dl, r_WFE, IWA, trans_instru)
        # For Imager (6D): (Planets,    l0, Dl, r_WFE, IWA, trans_instru)
        signal_planets       = np.load(signal_path,       mmap_mode="r") # [e-/FWHM/DIT]
        sigma_halo_2_planets = np.load(sigma_halo_2_path, mmap_mode="r") # [e-/FWHM/DIT]**2
        sigma_bkg_2_planets  = np.load(sigma_bkg_2_path,  mmap_mode="r") # [e-/FWHM/DIT]**2
        DIT_planets          = np.load(DIT_path,          mmap_mode="r") # [mn/DIT]
        if instru_type == "IFU":
            sigma_syst_base_2_planets = np.load(sigma_syst_base_2_path, mmap_mode="r") # [e-/FWHM/DIT]**2 (at sigma_m = 1)
        elif instru_type == "imager":
            sigma_syst_base_2_planets = None
        print("\nOpening existing FastYield data...")



    ####################################
    # Otherwise doing the calculations #
    ####################################
    else:
        if force_new_calc:
            print(f"\nNew FastYield ELT/{instru} calculations (forcing_new_calc=True).")
        else:
            print(f"\nNew FastYield ELT/{instru} calculations (missing files).")

        #
        # PSF data (PSF profile, fraction core and coronagraph transmission) preparations (l0, r_WFE, IWA, mag_star, sep)
        #

        # PSF_profile_density (no coronagraph) = mean surface-brightness density in each annular bin divided by the total flux (as function of angular separation)
        # fraction_core       (no coronagraph) = total flux enclosed in the core divided by the total flux                     (constant with separation)

        # PSF_profile_density (coronagraph) = mean surface-brightness density in each annular bin divided by the total coronagraphic flux (as function of angular separation)
        # fraction_core       (coronagraph) = total flux enclosed in the core divided by the total coronagraphic flux                     (for different offset angular separation from the coronagraph, at 0 => on-axis)
        # radial_transmission (coronagraph) = total coronagraphic flux divided by total flux                                              (for different offset angular separation from the coronagraph, at 0 => on-axis)

        # Retrieving PSF data and axis (HARD CODED RANGES...)
        l0_min_PSF              = 0.5   # [µm]; exact PCS SUMMARY FITS suffix
        l0_max_PSF              = 2.5   # [µm]
        r_WFE_min_PSF           = 0.2   # [dimensionless]
        r_WFE_max_PSF           = 2.0   # [dimensionless]
        IWA_min_PSF             = 1.0   # [lambda0_band/D]
        IWA_max_PSF             = 5.0   # [lambda0_band/D]
        mag_star_min_PSF        = 0     # [dimensionless]
        mag_star_max_PSF        = 10    # [dimensionless]
        sep_max_PSF             = 1_000 # [mas]
        suffix_PSF              = f"{apodizer}_{coronagraph}_{strehl}_{l0_min_PSF}_{l0_max_PSF}_{r_WFE_min_PSF}_{r_WFE_max_PSF}_{IWA_min_PSF}_{IWA_max_PSF}_{mag_star_min_PSF}_{mag_star_max_PSF}_{sep_max_PSF}"
        wave0                   = fits.getdata(psf_dir / f"{instru}_wave_{suffix_PSF}.fits")
        r_WFE0                  = fits.getdata(psf_dir / f"{instru}_r_WFE_{suffix_PSF}.fits")
        IWA0                    = fits.getdata(psf_dir / f"{instru}_IWA_{suffix_PSF}.fits")
        mag_star0               = fits.getdata(psf_dir / f"{instru}_mag_star_{suffix_PSF}.fits")
        separation0             = fits.getdata(psf_dir / f"{instru}_separation_{suffix_PSF}.fits")
        PSF_profile_density0_5D = fits.getdata(psf_dir / f"{instru}_PSF_profile_density_5D_{suffix_PSF}.fits")
        fraction_core0_5D       = fits.getdata(psf_dir / f"{instru}_fraction_core_5D_{suffix_PSF}.fits")
        if coronagraph is not None:
            radial_transmission0_5D = fits.getdata(psf_dir / f"{instru}_radial_transmission_5D_{suffix_PSF}.fits")
        else:
            radial_transmission0_5D = None

        # --- Checking PSF input axes and dimensions ---
        axes_psf = (wave0, r_WFE0, IWA0, mag_star0, separation0)
        for axis in axes_psf:
            if axis.ndim != 1 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                raise ValueError("Invalid PCS PSF axis.")
        expected_shape = tuple(map(len, axes_psf))
        for array in (PSF_profile_density0_5D, fraction_core0_5D, radial_transmission0_5D):
            if array is not None and array.shape != expected_shape:
                raise ValueError(f"Invalid PCS PSF shape: {array.shape}, expected {expected_shape}.")
        for requested, available in ((l0, wave0), (r_WFE, r_WFE0), (IWA, IWA0)):
            if np.min(requested) < available[0] - 1e-6 or np.max(requested) > available[-1] + 1e-6:
                raise ValueError("Requested parameters outside PCS PSF grid.")

        # Keeping the raw mag_star and separation axis
        mag_star   = mag_star0.astype(dtype,   copy=False)
        separation = separation0.astype(dtype, copy=False)
        N_mag_star = len(mag_star)
        N_sep      = len(separation)

        # Temporary PSF memmaps
        PSF_profile_5D     = create_memmap_with_log(PSF_profile_5D_tmp_path,     shape=(N_l0, N_r_WFE, N_IWA, N_mag_star, N_sep), dtype=dtype, mode="w+")
        fraction_core_5D   = create_memmap_with_log(fraction_core_5D_tmp_path,   shape=(N_l0, N_r_WFE, N_IWA, N_mag_star, N_sep), dtype=dtype, mode="w+")
        PSF_profile_max_4D = create_memmap_with_log(PSF_profile_max_4D_tmp_path, shape=(N_l0, N_r_WFE, N_IWA, N_mag_star),        dtype=dtype, mode="w+",)

        # Build interpolators once
        eps_pos  = np.finfo(dtype).tiny
        eps01    = 1e-32
        safe_pos = lambda x: np.clip(x, eps_pos, None)
        safe01   = lambda x: np.clip(x, eps01, 1 - eps01)
        log_interp_psf      = RegularGridInterpolator((wave0, r_WFE0, IWA0, mag_star0, separation0), np.log(safe_pos(PSF_profile_density0_5D)), bounds_error=True)
        logit_interp_core   = RegularGridInterpolator((wave0, r_WFE0, IWA0, mag_star0, separation0), logit(safe01(fraction_core0_5D)),          bounds_error=True)
        if coronagraph is not None:
            logit_interp_rt = RegularGridInterpolator((wave0, r_WFE0, IWA0, mag_star0, separation0), logit(safe01(radial_transmission0_5D)),    bounds_error=True)
        else:
            logit_interp_rt = None

        # Build the temporary PSF memmaps by chunks along l0
        l0_chunk = max(1, min(chunksize, N_l0))
        for j0 in tqdm(range(0, N_l0, l0_chunk), desc="Building temporary PSF memmaps"):
            j1           = min(j0 + l0_chunk, N_l0)
            l0_chunk_arr = l0[j0:j1]
            pts          = np.stack(np.meshgrid(l0_chunk_arr, r_WFE, IWA, mag_star, separation, indexing="ij"), axis=-1)

            # Interpolating over current grid
            PSF_profile_density_blk     = np.exp(log_interp_psf(pts))
            fraction_core_blk           = expit(logit_interp_core(pts))
            if coronagraph is not None:
                radial_transmission_blk = expit(logit_interp_rt(pts))
            else:
                radial_transmission_blk = None

            # Sanity check of the values
            if (np.any(~np.isfinite(PSF_profile_density_blk)) or np.any(~np.isfinite(fraction_core_blk)) or (coronagraph is not None and np.any(~np.isfinite(radial_transmission_blk)))):
                raise KeyError("The PSF data should only have finite values!")

            # Converting flux density in fraction of flux inside the FOV per spaxel for a bin at l0, r_WFE, IWA, mag_star as function of separation
            PSF_profile_blk = PSF_profile_density_blk * pxscales[j0:j1, None, None, None, None]**2 # [star flux fraction/px]

            # Star transmission throught coronagraph, given at sep = 0, on-axis, assuming that the coronagraph is perfectly align with the star
            # star_transmission = total star flux through coronagraph / total star flux
            if coronagraph is not None:
                i_sep0 = 0
                if separation[i_sep0] != 0.0:
                    raise KeyError("The first value of 'separation' should be 0.")
                star_transmission_blk = radial_transmission_blk[..., i_sep0] # (l0c, r_WFE, IWA, mag_star)
                # Star PSF profile throught the coronagraph
                PSF_profile_blk      *= star_transmission_blk[..., None]     # (l0c, r_WFE, IWA, mag_star, sep)
                # Fraction core signal (for the planet, inside the FWHM) through the coronagraph
                fraction_core_blk    *= radial_transmission_blk              # (l0c, r_WFE, IWA, mag_star, sep)

            PSF_profile_5D[j0:j1]   = PSF_profile_blk.astype(dtype,   copy=False)
            fraction_core_5D[j0:j1] = fraction_core_blk.astype(dtype, copy=False)

            # PSF_max (brightest pixels in the PSF) [max star flux fraction/px] in order to compute the DIT length to reach saturation
            PSF_profile_max_4D[j0:j1] = np.max(PSF_profile_blk, axis=-1).astype(dtype, copy=False)

            # Free block temporaries immediately
            del pts
            del PSF_profile_density_blk
            del fraction_core_blk
            del PSF_profile_blk
            if radial_transmission_blk is not None:
                del radial_transmission_blk
                del star_transmission_blk
            gc.collect()

        # PSF_profile_5D(l0, r_WFE, IWA, mag_star, separation)   = [star flux fraction/px]     = star flux per px divided by the total star flux          (as function of angular separation)
        # fraction_core_5D(l0, r_WFE, IWA, mag_star, separation) = [planet flux fraction/FWHM] = planet flux inside FWHM divided by the total planet flux (constant with separation if no coronagraph, or for different angular separation from the coronagraph)
        # PSF_profile_max_4D(l0, r_WFE, IWA, mag_star)           = [max star flux fraction/px] = max star flux per px divided by the total star flux      (given at the brightest position)

        PSF_profile_5D.flush()
        fraction_core_5D.flush()
        PSF_profile_max_4D.flush()

        # Free raw PSF arrays as soon as temp memmaps are built
        del wave0, r_WFE0, IWA0, mag_star0, separation0
        del PSF_profile_density0_5D, fraction_core0_5D
        if radial_transmission0_5D is not None:
            del radial_transmission0_5D
        del log_interp_psf, logit_interp_core, logit_interp_rt
        del PSF_profile_5D, fraction_core_5D, PSF_profile_max_4D
        gc.collect()



        #
        # Spectra preparations
        #

        # K-band for photometry (for star spectra normalizations using K-band magnitudes)
        wave_K = get_wave_K() # [µm]

        # Vega spectrum on K-band in [J/s/m2/µm]
        vega_spectrum   = load_vega_spectrum()                                       # [J/s/m2/µm]
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False) # [J/s/m2/µm]
        counts_vega_K   = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)

        # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotationnal broadening with Vsini)
        # Cover every requested IFU spectral window, including the widest one
        # at the boundaries of the l0 axis; retain the original imager handling.
        if l0[0] - Dl[0] / 2 <= 0:
            raise ValueError("The spectral bandwidth extends to nonphysical wavelengths; reduce Dl_max.")
        lmin_model = 0.98 * (l0[0] - Dl[0] / 2)                # [µm]
        lmax_model = 1.02 * (l0[0] + Dl[0] / 2)                # [µm]
        dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
        wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)

        # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
        wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
        dl_instru   = np.gradient(wave_instru)                                                    # [µm/bin] Nyquist sampling of a spectrum with resolving power R_model: 2 samples per resolution element across the whole axis

        # Effective model range
        lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin
        lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax

        # Tellurics transmission spectrum (from SkyCalc)
        wave_tell, trans_tell = load_tell_trans(airmass=1.0) # at airmass = 1.0
        airmass               = 1.2                          # 1.2 to be coherent with the used background skytable
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell**airmass).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1]))

        # Telescope transmission spectrum
        data_elt_ref = np.genfromtxt(instru_dir / "ELT_reflectivity.csv", delimiter=",", names=True, dtype=float, encoding=None)
        wave_ref     = data_elt_ref["wave"] # [µm]
        elt_ref      = data_elt_ref["M1M5"] # ELT mirror train reflectivity, from common ICD, section 4.11, p37 (Document Number: ESO-253082)
        trans_tel    = trans_dust * elt_ref
        trans_tel    = Spectrum(wavelength=wave_ref, flux=trans_tel).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tel[0], trans_tel[-1]))

        # Tellurics x telescope transmission spectrum
        trans_tell_tel = Spectrum(wavelength=wave_instru, flux=trans_tell.flux*trans_tel.flux, R=np.fmax(trans_tell.R, trans_tel.R), T=None, lg=None, model=None, rv=None, vsini=None, sigma=None)

        # Plotting transmissions
        # from fastyield.utils import plot_trans_tell_tel, plot_bkg_skycalc
        # plot_trans_tell_tel(trans_tell=trans_tell, trans_tel=trans_tel)

        # Background spectrum [ph/mas2/mn/µm] (from SkyCalc)
        # plot_bkg_skycalc(filename=instru_dir / "skytable_background.fits")
        data_bkg         = fits.getdata(instru_dir / "skytable_background.fits")
        wave_bkg         = data_bkg["lam"]  * 1e-3             # [nm] => [µm]
        background       = data_bkg["flux"] * 60 * S / 1000**2 # [ph/s/m2/µm/arcsec2] => [ph/mas2/mn/µm]
        background       = Spectrum(wavelength=wave_bkg, flux=background).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(background[0], background[-1])) # [ph/mas2/mn/µm]
        background.flux *= trans_tel.flux # [ph/mas2/mn/µm] through the telescope



        #
        # IFU path
        #
        if instru_type == "IFU":

            # --- Pre-computing the scale factor to convert from [J/s/m²/µm] to [ph/mn/µm] ---
            scale_spectrum  = wave_instru*1e-6 / (h*c) # [J/s/m²/µm]  => [ph/s/m2/µm]
            scale_spectrum *= S * 60                   # [ph/s/m2/µm] => [ph/mn/µm]

            # --- Preparing for wave axis, range, trans_tell_tel, background for each point ---
            trans_tell_tel_1D_list = [None] * len(R) # (R)
            background_1D_list     = [None] * len(R) # (R)     [ph/px/mn/bin]
            wave_1D_list           = [None] * len(R) # (R)     [µm]
            dwave_1D_list          = [None] * len(R) # (R)     [µm/bin]
            range_1D_list          = [None] * len(R) # (R)
            l0_2D                  = l0[:, None]     # (l0, 1) [µm]
            Dl_2D                  = Dl[None, :]     # (1, Dl) [um]
            # Symmetric wavelength intervals, exactly as in the imager case.
            l0_2D_min = l0_2D - Dl_2D / 2
            l0_2D_max = l0_2D + Dl_2D / 2
            for i, res in enumerate(R):

                # --- Build a wavelength grid with *constant resolving power* R = res ---
                # For a spectrograph with resolving power R, one resolution element is:
                #   Δλ_res = λ / R
                # Nyquist sampling requires 2 samples per resolution element, hence a pixel step:
                #   Δλ_pix ≈ (λ / R) / 2  =>  Δλ_pix / λ ≈ 1 / (2R)
                # This means the natural uniform grid is in ln(λ), with constant step: dln(λ)/dλ = 1/λ => dln(λ) = dλ/λ = 1 / (2R)
                #   Δln(λ) = ln(λ_{i+1}) - ln(λ_i) ≈ Δλ_pix / λ ≈ 1/(2R) = constant
                dln = 1.0 / (2.0 * res)

                # Number of samples needed to cover [lmin_model, lmax_model] with that constant Δln(λ)
                # ln(λ_i) = ln(λ_min) + k*Δln(λ) => λ_i = λ_min * exp(k*Δln(λ))
                # If we have n points, it means that we have n - 1 between l0_min and l0_max
                # because: ln(l0_max/l0_min) = (n-1) * dln  =>  n ≈ ln(l0_max/l0_min)/dln + 1
                n = int(np.floor(np.log(lmax_model / lmin_model) / dln)) + 1

                # Log-uniform wavelength grid: λ_i = λ_min * exp(i * Δln(λ))
                # This keeps λ/Δλ (i.e., R) approximately constant across the whole band.
                wave_res         = lmin_model * np.exp(np.arange(n) * dln) # [µm]
                dwave_res        = np.gradient(wave_res)                   # [µm/bin]
                wave_1D_list[i]  = wave_res                                # [µm]
                dwave_1D_list[i] = dwave_res                               # [µm/bin]

                # Find spectral channels whose centres lie in
                # [l0 - Dl/2, l0 + Dl/2]; the effective channel count depends
                # on R, l0 and Dl and is no longer an independent axis.
                # Using side="left"/"right" ensures we include the full requested wavelength span.
                idx_lo           = np.searchsorted(wave_res, l0_2D_min, side="left").astype(np.int32)
                idx_hi           = np.searchsorted(wave_res, l0_2D_max, side="right").astype(np.int32)
                idx_lo           = np.clip(idx_lo, 0, len(wave_res)-1)
                idx_hi           = np.clip(idx_hi, 1, len(wave_res))
                idx_hi           = np.maximum(idx_hi, idx_lo + 1)
                range_1D_list[i] = (idx_lo, idx_hi)

                # Tellurics x Telescope spectrum on the grid at this resolution
                trans_tell_tel_1D_list[i] = trans_tell_tel.degrade_resolution(wave_res, renorm=False, R_output=res, filter_type=filter_type).flux

                # Background spectrum on this grid in [ph/px/mn/bin]
                background_res        = background.degrade_resolution(wave_res, renorm=False, R_output=res, filter_type=filter_type).flux # (len(wave_res))     [ph/mn/µm/mas2]
                background_res       *= dwave_res                                                                                         # (len(wave_res))     [ph/mn/µm/mas2]  => [ph/mn/bin/mas2]
                background_1D_list[i] = background_res.astype(dtype, copy=False)                                                          # (len(wave_res))     [ph/mn/bin/mas2]


            # --- Global context worker ---
            _IFU_CTX = dict(planet_table=planet_table, A_FWHM=A_FWHM, separation=separation, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, wave_model=wave_model, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, mag_star=mag_star, band_WFS=band_WFS, scale_spectrum=scale_spectrum, N_R=N_R, N_l0=N_l0, N_Dl=N_Dl, R=R, wave_1D_list=wave_1D_list, dwave_1D_list=dwave_1D_list, trans_tell_tel_1D_list=trans_tell_tel_1D_list, trans_instru=trans_instru, background_1D_list=background_1D_list, pxscales=pxscales, Rc=Rc, filter_type=filter_type, range_1D_list=range_1D_list, valid_windows_3D=valid_windows_3D, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, PSF_profile_5D_path=str(PSF_profile_5D_tmp_path), fraction_core_5D_path=str(fraction_core_5D_tmp_path), PSF_profile_max_4D_path=str(PSF_profile_max_4D_tmp_path), signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), sigma_syst_base_2_path=str(sigma_syst_base_2_path), dtype=dtype, D=D, pxscales_rad=pxscales_rad)


            # --- Arguments for Pool ---
            init_worker  = fyh.init_worker_IFU
            _CTX         = _IFU_CTX
            func_process = fyh.process_IFU
            shape_SNR    = (N_PT, N_R, N_l0, N_Dl, N_r_WFE, N_IWA, N_TI)
            tqdm_desc    = f"ELT/{instru} ({instru_type} and {post_processing}) multiprocessing over planets of signal, stellar halo photon noise, DIT length and base for {residuals} noise"


        #
        # Imager path
        #
        elif instru_type == "imager":

            # --- Pre-computing the scale factor to convert from [J/s/m²/µm] to [ph/mn/bin] ---
            scale_spectrum  = wave_instru*1e-6 / (h*c) # [J/s/m²/µm]  => [ph/s/m2/µm]
            scale_spectrum *= S * 60 * dl_instru       # [ph/s/m2/µm] => [ph/mn/bin]

            # Preparing for bandwidth range for each point
            l0_2D  = l0[:, None]        # [µm] (l0, 1)
            Dl_2D  = Dl[None, :]        # [µm] (1, Dl)
            l0_2D_min = l0_2D - Dl_2D/2 # [µm] (l0, Dl)
            l0_2D_max = l0_2D + Dl_2D/2 # [µm] (l0, Dl)

            # Find the corresponding index interval [idx_lo, idx_hi) in the wave_res grid.
            # Using side="left"/"right" ensures we include the full requested wavelength span.
            idx_lo   = np.searchsorted(wave_instru, l0_2D_min, side="left").astype(np.int32)
            idx_hi   = np.searchsorted(wave_instru, l0_2D_max, side="right").astype(np.int32)
            idx_lo   = np.clip(idx_lo, 0, len(wave_instru)-1)
            idx_hi   = np.clip(idx_hi, 1, len(wave_instru))
            idx_hi   = np.maximum(idx_hi, idx_lo + 1)
            range_1D = (idx_lo, idx_hi)

            # Background total flux preparation (l0, Dl) in [ph/px/mn]
            background          = background.flux * dl_instru                                                   # [ph/mn/µm/mas2] => [ph/mn/bin/mas2]
            background_flux_2D  = fyh.window_sums_1d(values=background, idx_lo=idx_lo, idx_hi=idx_hi).astype(dtype) # [ph/mn/mas2]
            background_flux_2D *= pxscales[:, None]**2                                                          # [ph/mn/mas2] => [ph/px/mn]

            # --- Global context worker ---
            _IM_CTX = dict(planet_table=planet_table, A_FWHM=A_FWHM, separation=separation, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, mag_star=mag_star, scale_spectrum=scale_spectrum, N_l0=N_l0, N_Dl=N_Dl, trans_instru=trans_instru, trans_tell_tel=trans_tell_tel.flux, background_flux_2D=background_flux_2D, range_1D=range_1D, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, PSF_profile_5D_path=str(PSF_profile_5D_tmp_path), fraction_core_5D_path=str(fraction_core_5D_tmp_path), PSF_profile_max_4D_path=str(PSF_profile_max_4D_tmp_path), signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), dtype=dtype)

            # --- Arguments for Pool ---
            init_worker  = fyh.init_worker_IM
            _CTX         = _IM_CTX
            func_process = fyh.process_IM
            shape_SNR    = (N_PT, N_l0, N_Dl, N_r_WFE, N_IWA, N_TI)
            tqdm_desc    = f"ELT/{instru} ({instru_type} and {post_processing}) multiprocessing over planets of signal, stellar halo photon noise and DIT length"



        #
        # Parallelized calculations for each planet
        #

        # Creating files with size logging
        file_specs = [("signal",       signal_path,       shape_SNR, dtype),
                      ("sigma_halo_2", sigma_halo_2_path, shape_SNR, dtype),
                      ("sigma_bkg_2",  sigma_bkg_2_path,  shape_SNR, dtype),
                      ("DIT",          DIT_path,          shape_SNR, dtype),]
        if instru_type == "IFU":
            file_specs.append(("sigma_syst_base_2", sigma_syst_base_2_path, shape_SNR, dtype))
        total_nbytes = sum(memmap_nbytes(shape, dtype) for _, _, shape, dtype in file_specs)
        print(f"\nCreating memmap files in {sim_dir_concept}")
        print(f"Total expected disk usage: {format_nbytes(total_nbytes)}")
        print("---------------------------------------------------------")
        print()
        signal_planets       = create_memmap_with_log(signal_path,       shape_SNR, dtype=dtype, mode="w+")
        sigma_halo_2_planets = create_memmap_with_log(sigma_halo_2_path, shape_SNR, dtype=dtype, mode="w+")
        sigma_bkg_2_planets  = create_memmap_with_log(sigma_bkg_2_path,  shape_SNR, dtype=dtype, mode="w+")
        DIT_planets          = create_memmap_with_log(DIT_path,          shape_SNR, dtype=dtype, mode="w+")
        if instru_type == "IFU":
            sigma_syst_base_2_planets = create_memmap_with_log(sigma_syst_base_2_path, shape_SNR, dtype=dtype, mode="w+")
        elif instru_type == "imager":
            sigma_syst_base_2_planets = None

        # Creating context
        if sys.platform.startswith(("win", "darwin")): # Windows, MACOS
            ctx_mp = mp.get_context("spawn")
        else:
            ctx_mp = mp.get_context("fork") # Linux

        # Warmup for systematics in IFU
        if instru_type == "IFU":
            fyh.warmup_signal_noise_numba(dtype=dtype)

        # Computations
        print()
        with ctx_mp.Pool(processes=nproc, initializer=init_worker, initargs=(_CTX,)) as pool:
            print(f'{tqdm_desc}:')
            for _ in tqdm(pool.imap_unordered(func_process, range(N_PT), chunksize=chunksize), total=N_PT):
                pass
        signal_planets.flush()
        sigma_halo_2_planets.flush()
        sigma_bkg_2_planets.flush()
        DIT_planets.flush()
        if instru_type == "IFU":
            sigma_syst_base_2_planets.flush()

        for p in [PSF_profile_5D_tmp_path, fraction_core_5D_tmp_path, PSF_profile_max_4D_tmp_path]:
            if p.exists():
                p.unlink()



    # %%

    # Define parameters and their names
    if instru_type == "IFU":
        params         = [  R,        l0,        Dl,   r_WFE,               IWA,           trans_instru,   100*sigma_m,         FoV]  # params axis
        params_names   = ["R", "l0 [µm]", "Dl [µm]", "r_WFE", "IWA [lambda0/D]", "trans_instru [e-/ph]", "sigma_m [%]", "FoV [mas]"]  # params names
    elif instru_type == "imager":
        params         = [       l0,        Dl,   r_WFE,               IWA,   trans_instru,   100*sigma_m,   FoV]  # params axis
        params_names   = ["l0 [µm]", "Dl [µm]", "r_WFE", "IWA [lambda0/D]", "trans_instru", "sigma_m [%]", "FoV"]  # params labels


    # %%
    # COMPUTING MAIN QUANTITIES

    # Computing the SNR and detection mask (SNR > SNR_thr) for each planet
    # For IFU (8D):    (planets, R, l0, Dl, r_WFE, IWA, trans_instru, sigma_m)
    # For Imager (7D): (planets,    l0, Dl, r_WFE, IWA, trans_instru, sigma_m)
    SNR_planets     = fyh.get_SNR(instru=instru, instru_type=instru_type, post_processing=post_processing, exposure_time=exposure_time, min_DIT=min_DIT, RON0=RON0, RON_lim=RON_lim, DC0=DC0, A_FWHM=A_FWHM, Rc=Rc, filter_type=filter_type, signal_planets=signal_planets, sigma_halo_2_planets=sigma_halo_2_planets, sigma_bkg_2_planets=sigma_bkg_2_planets, sigma_syst_base_2_planets=sigma_syst_base_2_planets, DIT_planets=DIT_planets, R=R if instru_type == "IFU" else None, Dl=Dl if instru_type == "IFU" else None, sigma_m=sigma_m, sim_dir=sim_dir_concept, suffix=suffix, dtype=dtype)
    mask_detections = fyh.get_mask_detections(SNR_planets=SNR_planets, SNR_thr=SNR_thr, sim_dir=sim_dir_concept, suffix=suffix)

    # Precompute FoV gating once
    p0_FoV = np.searchsorted(FoV / 2, separation_planets, side="left").astype(np.int32)

    # Getting ndim detection probability
    # For IFU (8D):    (R, l0, Dl, r_WFE, IWA, trans_instru, sigma_m, FoV)
    # For Imager (7D): (   l0, Dl, r_WFE, IWA, trans_instru, sigma_m, FoV)
    Pdet = fyh.get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, valid_windows_3D=valid_windows_3D)

    #%%
    # POPULATION DIAGNOSTICS

    idx_FoV      = next(idx for idx, name in enumerate(params_names) if "FoV"     in name)
    idx_sigma_m  = next(idx for idx, name in enumerate(params_names) if "sigma_m" in name)
    idx_l0       = next(idx for idx, name in enumerate(params_names) if "l0"      in name)
    noise_labels = np.array(["Stellar halo", "Background", "Read noise", "Dark current", str(residuals)], dtype=object)

    # Configuration maximizing Pdet
    params_imax     = np.array(np.unravel_index(np.nanargmax(Pdet), Pdet.shape), dtype=int)
    SNR_plot        = SNR_planets.squeeze()
    SNR_plot[p0_FoV > params_imax[idx_FoV]] = 0.0

    # Raw arrays do not contain sigma_m or FoV
    sigma_halo_2_best = sigma_halo_2_planets.squeeze()
    sigma_bkg_2_best  = sigma_bkg_2_planets.squeeze()
    DIT_best          = DIT_planets.squeeze()

    il0_best     = params_imax[idx_l0]
    RON0_best    = float(np.asarray(RON0)[il0_best])
    RON_lim_best = float(np.asarray(RON_lim)[il0_best])
    DC0_best     = float(np.asarray(DC0)[il0_best])
    min_DIT_best = float(np.asarray(min_DIT)[il0_best])
    N_DIT_best   = np.clip(np.floor(exposure_time / DIT_best), 1, None)
    N_read_best  = np.floor(DIT_best / min_DIT_best).astype(np.int32)

    sigma_RON_2_best = np.full_like(DIT_best, RON0_best**2, dtype=float)
    mask_read        = N_read_best >= 2
    if np.any(mask_read):
        n_read                      = N_read_best[mask_read].astype(float)
        sigma_RON_2_best[mask_read] = RON0_best**2 * 12 * (n_read - 1) / (n_read * (n_read + 1)) + RON_lim_best**2
    sigma_RON_2_best *= A_FWHM
    sigma_DC_2_best   = DC0_best * DIT_best * A_FWHM
    sigma_m_best      = float(params[idx_sigma_m][params_imax[idx_sigma_m]]) / 100.0

    if instru_type == "IFU":
        sigma_syst_base_2_best = sigma_syst_base_2_planets.squeeze()
    elif instru_type == "imager":
        sigma_syst_base_2_best = sigma_halo_2_best**2
    else:
        raise ValueError("instru_type must be 'IFU' or 'imager'.")

    var_halo = N_DIT_best * sigma_halo_2_best
    var_bkg  = N_DIT_best * sigma_bkg_2_best
    var_RON  = N_DIT_best * sigma_RON_2_best
    var_DC   = N_DIT_best * sigma_DC_2_best
    var_syst = N_DIT_best**2 * sigma_m_best**2 * sigma_syst_base_2_best

    noise_stack                 = np.vstack([var_halo, var_bkg, var_RON, var_DC, var_syst])
    valid_noise                 = np.any(np.isfinite(noise_stack), axis=0)
    dominant_noise              = np.full(N_PT, "Unknown", dtype=object)
    dominant_noise[valid_noise] = noise_labels[np.nanargmax(noise_stack[:, valid_noise], axis=0)]

    yield_population_plot(table=table_type, instru=instru, instru_type=instru_type, post_processing=post_processing, thermal_model=thermal_model, reflected_model=reflected_model, exposure_time=exposure_time, band_contrast_plot=band_plot, band_regime_plot=band_plot, planet_table=planet_table, SNR_plot=SNR_plot, dominant_noise=dominant_noise, SNR_thr=SNR_thr, save_dir=sim_dir_concept, DL_mas=np.nan, show_golden_sample=show_golden_sample, show_det_regions=show_det_regions, show_ptype_regions=show_ptype_regions, space=planet_population_space)


    # %%
    # --- Clean temporary SNR and mask_detections memmap files ---
    snr_tmp_path  = Path(SNR_planets.filename)
    mask_tmp_path = Path(mask_detections.filename)

    del SNR_planets, mask_detections, signal_planets, sigma_halo_2_planets, sigma_bkg_2_planets, sigma_syst_base_2_planets, DIT_planets
    gc.collect()

    if snr_tmp_path.exists():
        snr_tmp_path.unlink()
    if mask_tmp_path.exists():
        mask_tmp_path.unlink()

    return locals()



# %%

if __name__ == "__main__":
    __spec__ = None

    if sys.platform.startswith("win"):
        mp.freeze_support() # for Windows
    _out = main()
    globals().update(_out)

