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

    # --- Instrument concept ---
    # One YAML file per concept in instrument_concepts/ (ifu_global, imager_global, concept_A_highres_vis_ifu, ...)
    #concept_name       = "concept_A_highres_vis_ifu_band=VRIz"
    concept_name       = "concept_Ab_highres_nir_ifu_band=zYJ.yaml"
    #concept_name       = "ifu_global"
    instrument_concept = fyh.load_instrument_concept(concept_name)

    # --- General parameters for the simulation ---
    coronagraph        = "APLC"                                # "CLC", "APLC", or None; matches PCS SUMMARY FITS labels
    apodizer           = "NO_SP"                               # Shaped pupil mask (NO_SP => no mask)
    strehl             = "Q2"                                  # Sky atmospheric condition (1st quartile, 2nd, etc.)
    SNR_thr            = 5                                     # Detection threshold
    exposure_time      = 10*60                                 # Total exposure time per planet [mn]
    force_new_calc     = False                                  # Forcing new simulations calculations
    thermal_model      = "auto"                                # Model for the thermal spectrum of the planet ("auto", "None", "BT-Settl", "Exo-REM", "SONORA", "PICASO", "Saumon", etc.)
    reflected_model    = "auto"                                # Model for the albedo of the planet ("auto", "tellurics", "flat", "PICASO")
    instru_type        = instrument_concept['instru_type']     # Type of instrument ("IFU" or "imager")
    post_processing    = instrument_concept['postproc']        # Post-processing method ("MM" or "DI")
    spatial_sampling   = 2                                     # [px/FWHM] Number of pixel per spatial FWHM along 1 direction (spatial_sampling >= 2 => Nyquist spatial sampling)
    size_core          = 2                                     # Aperture size on which the signal is integrated [px]
    A_FWHM             = size_core**2                          # Number of pixel per FWHM box area
    Rc                 = 1_00                                  # MM cut-off resolution (Rc~100 is enough to reach ~1e-8 with speckles only, Rc~1000 would allows to go further (more conservative))
    filter_type        = "gaussian_fast"                       # MM filter type
    table_type         = "Archive"                             # "Archive": for all known exoplanets | "Simulated": TODO

    # --- Separation range ---
    sep_min = 0     # [mas]
    sep_max = 1_000 # [mas] max separation of the raw PSF data

    # --- Relative post-AO WFE and coronagraphic FPM size ---
    # r_WFE=1 means the nominal AO performance for each stellar V magnitude.
    # IWA is the FPM radius in lambda0_band/D (not mas); the PCS nominal value is 2.4.
    r_WFE_ref = 1.0 # [dimensionless], by definition
    IWA_ref   = 2.4 # [lambda0_band/D], see PSF data

    # Nominal PCS configuration. For sensitivity studies, vary within the FITS ranges.
    #r_WFE_min, r_WFE_max = r_WFE_ref, r_WFE_ref
    IWA_min, IWA_max     = IWA_ref, IWA_ref

    # Exploratory (extrapolated) ranges:
    r_WFE_min, r_WFE_max = 0.5, 2.0
    # IWA_min, IWA_max     = 1.0, 5.0

    # --- Plots configuration ---
    plot_global_results = False
    plot_populations    = True

    # Band used for the contrast shown on the y-axis
    band_contrast_plot = "I"

    # Choose in wich space the planet populations are plotted
    # "instr" : instrumental space (contrast vs. angular separation)
    # "phys"  : physical space (mass vs. projected semi-major axis)
    # "both"  : combine the plots in two panels
    planet_population_space = 'phys'

    # Choose how the detection status is defined for population plots.
    # "max"          : use the SNR at the global maximum of the Pdet hypercube.
    # "marginalized" : use the default params_ranges.
    # "personalized" : use custom parameter ranges defined below.
    snr_population_mode = "personalized"

    # Standard definition of the photometric bands
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

    # Personalized ranges: scalar = fixed value, [min,max] = marginalized range, None = keep default params_ranges
    R_plot            = 80_000           # [dimensionless]
    l0_plot           = 0.90            # [µm]
    Dl_plot           = 0.15             # [µm]
    r_WFE_plot        = r_WFE_ref        # [dimensionless]
    IWA_plot          = IWA_ref          # [lambda0_band/D]
    trans_instru_plot = 0.25             # [e-/ph]
    sigma_m_plot      = 0.01  #[0.01, 0.1]              # [%]
    FoV_plot          = 300              # [mas]

    show_golden_sample = False
    show_det_regions   = False
    show_ptype_regions = False

    # --- IFU parameters space to explore ---
    if instru_type == "IFU":
        # Size of the parameters space to explore (N**Ndim)
        N                = 10             # [dims]
        # Spectral resolution (for each R, it is assumed to be constant along the whole wavelength range)
        R_min            = instrument_concept['R_min']        # [dimensionless]
        R_max            = instrument_concept['R_max']        # [dimensionless]
        # Central wavelength and physical spectral bandwidth (independent exploration axes).
        l0_min           = instrument_concept['l0_min']       # [µm]
        l0_max           = instrument_concept['l0_max']       # [µm]
        # Bandwidth width
        Dl_min           = instrument_concept['Dl_min']       # [µm]
        Dl_max           = instrument_concept['Dl_max']       # [µm]
        # Maximum number of spectral channels possible (is None for no restriction)
        Nl_max           = instrument_concept['Nl_max']       # [bins]
        # Instrumental transmission (without telescope transmission)
        trans_instru_min = instrument_concept['tr_min']       # [e-/ph]
        trans_instru_max = instrument_concept['tr_max']       # [e-/ph]
        # Effective residual halo modulation level:
        # For IFU: sigma_m is the effective fractional modulation of the final integrated stellar halo per spectral bin.
        # In practice, sigma_syst_base_2 stores the projected halo-squared term for one DIT,
        # already including throughput, PSF, FWHM integration, and DIT scaling.
        # The total systematic variance over the full exposure is then:
        # sigma_syst_tot^2 = sigma_m^2 * sigma_syst_base_2*N_DIT^2
        sigma_m_min      = instrument_concept['σ_m_min']      # [dimensionless]
        sigma_m_max      = instrument_concept['σ_m_max']      # [dimensionless]
        # Field of View
        FoV_min          = instrument_concept['FoV_min']      # [mas]
        FoV_max          = instrument_concept['FoV_max']      # [mas]
        # Axis
        R                = np.logspace(np.log10(R_min),       np.log10(R_max),       N) # i dim
        l0               = np.linspace(l0_min,                l0_max,                N) # j dim
        Dl               = np.linspace(Dl_min,                Dl_max,                N) # k dim [um]
        r_WFE            = np.linspace(r_WFE_min,             r_WFE_max,             N) # l dim
        IWA              = np.logspace(np.log10(IWA_min),     np.log10(IWA_max),     N) # m dim
        trans_instru     = np.linspace(trans_instru_min,      trans_instru_max,      N) # n dim
        sigma_m          = np.logspace(np.log10(sigma_m_min), np.log10(sigma_m_max), N) # o dim
        FoV              = np.logspace(np.log10(FoV_min),     np.log10(FoV_max),     N) # p dim
        # Axis must have unique and sorted values
        R                = np.unique(R)
        l0               = np.unique(l0)
        Dl               = np.unique(Dl)
        r_WFE            = np.unique(r_WFE)
        IWA              = np.unique(IWA)
        trans_instru     = np.unique(trans_instru)
        sigma_m          = np.unique(sigma_m)
        FoV              = np.unique(FoV)
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
        # Forcing to Differential Imaging only
        post_processing  = "DI"
        # Size of the parameters space to explore (N**Ndim)
        N                = 10             # [dims]
        # Bandwidth central wavelength (for each l0, the bandwidth is given by lmin,lmax = l0 +- Dl/2)
        l0_min           = instrument_concept['l0_min']       # [µm]
        l0_max           = instrument_concept['l0_max']       # [µm]
        # Bandwidth width
        Dl_min           = instrument_concept['Dl_min']       # [µm]
        Dl_max           = instrument_concept['Dl_max']       # [µm]
        # Instrumental transmission (without telescope transmission)
        trans_instru_min = instrument_concept['tr_min']       # [e-/ph]
        trans_instru_max = instrument_concept['tr_max']       # [e-/ph]
        # Effective residual halo modulation level:
        # For Imager: sigma_m is the effective fractional modulation of the final integrated stellar halo integrated over the full exposure time, the FWHM and the considered band.
        # The total systematic variance over the full exposure is then:
        # sigma_syst_tot^2 = sigma_m^2 * halo_2*N_DIT^2
        sigma_m_min      = instrument_concept['σ_m_min']      # [dimensionless]
        sigma_m_max      = instrument_concept['σ_m_max']      # [dimensionless]
        # Field of View
        FoV_min          = instrument_concept['FoV_min']      # [mas]
        FoV_max          = instrument_concept['FoV_max']      # [mas]
        # Axis
        l0               = np.linspace(l0_min,                l0_max,                N) # j dim
        Dl               = np.linspace(Dl_min,                Dl_max,                N) # k dim
        r_WFE            = np.linspace(r_WFE_min,             r_WFE_max,             N) # l dim
        IWA              = np.logspace(np.log10(IWA_min),     np.log10(IWA_max),     N) # m dim
        trans_instru     = np.linspace(trans_instru_min,      trans_instru_max,      N) # n dim
        sigma_m          = np.logspace(np.log10(sigma_m_min), np.log10(sigma_m_max), N) # o dim
        FoV              = np.logspace(np.log10(FoV_min),     np.log10(FoV_max),     N) # p dim
        # Axis must have unique and sorted values
        l0               = np.unique(l0)
        Dl               = np.unique(Dl)
        r_WFE            = np.unique(r_WFE)
        IWA              = np.unique(IWA)
        trans_instru     = np.unique(trans_instru)
        sigma_m          = np.unique(sigma_m)
        FoV              = np.unique(FoV)
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
    # Fail before costly PSF computations if the input table lacks stellar V photometry.
    vmag_check = np.array([fyh.get_star_vmag(planet) for planet in planet_table], dtype=float)
    outside_vmag = (vmag_check < 0.0) | (vmag_check > 10.0)
    if np.any(outside_vmag):
        print(f"WARNING: {np.count_nonzero(outside_vmag)} stars are outside the simulated V=[0,10] range; "
              "PSF magnitude interpolation will clamp to the nearest boundary")
    print(f"PCS stellar V-band photometry: [{vmag_check.min():.2f}, {vmag_check.max():.2f}] mag")

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
                coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, thermal_model=thermal_model, reflected_model=reflected_model, instru_type=instru_type, post_processing=post_processing, size_core=size_core, R_model=R_model, table_type=table_type,

                # exploration space
                l0_min=l0_min, l0_max=l0_max, r_WFE_min=r_WFE_min, r_WFE_max=r_WFE_max, IWA_min=IWA_min, IWA_max=IWA_max, trans_instru_min=trans_instru_min, trans_instru_max=trans_instru_max, sep_min=sep_min, sep_max=sep_max, N_PT=N_PT,

                # axis used
                l0=l0, Dl=Dl, r_WFE=r_WFE, IWA=IWA, trans_instru=trans_instru,
                )

    if instru_type == "IFU":
        meta.update(dict(Rc=Rc, filter_type=filter_type, R=R, Nl_max=Nl_max))
    suffix, meta_clean, _payload = make_suffix(meta, n=16)
    sim_dir_concept              = sim_dir / concept_name
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
        if l0_min - Dl_max / 2 <= 0:
            raise ValueError("The spectral bandwidth extends to nonphysical wavelengths; reduce Dl_max.")
        lmin_model = 0.98 * (l0_min - Dl_max / 2)                # [µm]
        lmax_model = 1.02 * (l0_max + Dl_max / 2)                # [µm]
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
            _IFU_CTX = dict(planet_table=planet_table, A_FWHM=A_FWHM, separation=separation, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, wave_model=wave_model, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, mag_star=mag_star, scale_spectrum=scale_spectrum, N_R=N_R, N_l0=N_l0, N_Dl=N_Dl, R=R, wave_1D_list=wave_1D_list, dwave_1D_list=dwave_1D_list, trans_tell_tel_1D_list=trans_tell_tel_1D_list, trans_instru=trans_instru, background_1D_list=background_1D_list, pxscales=pxscales, Rc=Rc, filter_type=filter_type, range_1D_list=range_1D_list, valid_windows_3D=valid_windows_3D, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, PSF_profile_5D_path=str(PSF_profile_5D_tmp_path), fraction_core_5D_path=str(fraction_core_5D_tmp_path), PSF_profile_max_4D_path=str(PSF_profile_max_4D_tmp_path), signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), sigma_syst_base_2_path=str(sigma_syst_base_2_path), dtype=dtype, D=D, pxscales_rad=pxscales_rad)


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
    # Plot general parameters
    fontsize           = 22                  # Standard fontsize
    ms                 = 15                  # Marker size
    ss                 = 400                 # Scatter size
    lw                 = 2                   # Linewidth
    alpha              = 0.9                 # Opacity
    gain               = True                # True for probability gain (normalized yield), False for absolute yields
    light_regime_plot  = "thermal+reflected" # 'thermal", "reflected" or "thermal+reflected"
    band_regime_plot   =  "I"                # Band where the thermal/reflected domination regime is splitted
    ptypes             = ["Jupiter",                 "Saturn",                "Neptune",                 "Earth"]
    marker_ptypes      = {"Jupiter": "s",            "Saturn": "v",           "Neptune": "P",            "Earth": "o"}
    label_ptypes       = {"Jupiter": "Jupiter-like", "Saturn": "Saturn-like", "Neptune": "Neptune-like", "Earth": "Earth-like"}

    # Define parameters and their names
    if instru_type == "IFU":
        params         = [R,                           l0,                                                Dl,                                           r_WFE,                                 IWA,                                                       trans_instru,                                    100*sigma_m,                                        FoV]                     # params axis
        params_ranges  = [(R.min(), R.max()),          (l0.min(), l0.max()),                              (Dl.min(), Dl.max()),                         (r_WFE_ref, r_WFE_ref),                (IWA_ref, IWA_ref),                                        (trans_instru.min(), trans_instru.max()),        ((100*sigma_m).min(), (100*sigma_m).min()),         (FoV.max(), FoV.max())]  # params ranges for marginalization
        #params_ranges  = [(R.min(), R.max()),          (l0.min(), l0.max()),                              (Dl.max(), Dl.max()),                         (r_WFE_ref, r_WFE_ref),                (IWA_ref, IWA_ref),                                        (trans_instru.max(), trans_instru.max()),        ((100*sigma_m).min(), (100*sigma_m).min()),         (FoV.max(), FoV.max())] # params ranges for marginalization
        params_priors  = ["log-uniform",               "uniform",                                         "uniform",                                    "uniform",                             "log-uniform",                                             "uniform",                                       "log-uniform",                                      "log-uniform"]           # params priors for marginalization
        params_islog   = [True,                        False,                                             False,                                        False,                                 True,                                                      False,                                           True,                                               True]                    # params space (linspace or logspace)
        params_isint   = [True,                        False,                                             False,                                        False,                                 False,                                                     False,                                           False,                                              True]                    # params rounded format
        params_names   = ["R",                         "l0 [µm]",                                         "Dl [µm]",                                    "r_WFE",                               "IWA [lambda0/D]",                                         "trans_instru [e-/ph]",                          "sigma_m [%]",                                      "FoV [mas]"]             # params names
        params_names_l = [r"$R$",                      r"$\lambda_0$  [µm]",                              r"$\Delta\lambda$ [µm]",                      r"$r_{\mathrm{WFE}}$",                 r"IWA [lambda0/D]",                                        r"$\gamma_{instru}$ [e-/ph]",                    r"$\sigma_m$ [%]",                                  r"$FoV$ [mas]"]          # params labels
        params_names_L = [r"Spectral resolution $R$",  r"Bandwidth central wavelength $\lambda_0$ [µm]",  r"Spectral bandwidth $\Delta\lambda$ [µm]",  r"Relative AO WFE $r_{\mathrm{WFE}}$", r"Coronagraph FPM radius [$\lambda_{0,\mathrm{band}}/D$]", r"Instrumental transmission $\gamma_{instru}$",  rf"{residuals} residual amplitude $\sigma_m$ [%]",  r"Field of View [mas]"]  # params detailed labels

    elif instru_type == "imager":
        params         = [l0,                                                Dl,                                         r_WFE,                                 IWA,                                                        trans_instru,                                    100*sigma_m,                                        FoV]                    # params axis
        params_ranges  = [(l0.min(), l0.max()),                              (Dl.min(), Dl.max()),                       (r_WFE_ref, r_WFE_ref),                (IWA_ref, IWA_ref),                                         (trans_instru.min(), trans_instru.max()),        ((100*sigma_m).min(), (100*sigma_m).min()),         (FoV.max(), FoV.max())] # params ranges for marginalization
        params_priors  = ["uniform",                                         "uniform",                                  "uniform",                             "log-uniform",                                              "uniform",                                       "log-uniform",                                      "log-uniform"]          # params priors for marginalization
        params_islog   = [False,                                             False,                                      False,                                 True,                                                       False,                                           True,                                               True]                   # params space (linspace or logspace)
        params_isint   = [False,                                             False,                                      False,                                 False,                                                      False,                                           False,                                              False]                  # params rounded format
        params_names   = ["l0 [µm]",                                         "Dl [µm]",                                  "r_WFE",                               "IWA [lambda0/D]",                                          "trans_instru",                                  "sigma_m [%]",                                      "FoV"]                  # params labels
        params_names_l = [r"$\lambda_0$ [µm]",                               r"$\Delta\lambda$ [µm]",                    r"$r_{\mathrm{WFE}}$",                 r"$IWA$ [$\lambda_{0,\mathrm{band}}/D$]",                   r"$\gamma_{instru}$ [e-/ph]",                    r"$\sigma_m$ [%]",                                  r"FoV [mas]"]           # params labels
        params_names_L = [r"Bandwidth central wavelength $\lambda_0$ [µm]",  r"Spectral coverage $\Delta\lambda$ [µm]",  r"Relative AO WFE $r_{\mathrm{WFE}}$", r"Coronagraph FPM radius [$\lambda_{0,\mathrm{band}}/D$]",  r"Instrumental transmission $\gamma_{instru}$",  rf"{residuals} residual amplitude $\sigma_m$ [%]",  r"Field of View [mas]"] # params detailed labels



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

    # --- Removing useless dimensions ---
    # Identify dimensions to remove (if size == 1)
    orig_params   = params.copy()
    orig_names    = params_names.copy()
    orig_sizes    = [len(param) for param in params]
    dim_to_remove = [idim for idim, size in enumerate(orig_sizes) if size == 1]
    dim_to_keep   = [idim for idim, size in enumerate(orig_sizes) if size != 1]
    print("\nPdet dimensions reduction:")
    for name, size in zip(orig_names, orig_sizes):
        flag = "remove" if size == 1 else "keep"
        print(f"  - {name:<20} : {size:>3}   [{flag}]")
    # Actual reduction
    for idim in sorted(dim_to_remove, reverse=True):
        Pdet = np.take(Pdet, 0, axis=idim)  # or np.squeeze(Pdet, axis=idim)
        params.pop(idim)
        params_ranges.pop(idim)
        params_priors.pop(idim)
        params_islog.pop(idim)
        params_isint.pop(idim)
        params_names.pop(idim)
        params_names_l.pop(idim)
        params_names_L.pop(idim)
    Ndim  = len(params)
    ncols = min(2, Ndim)
    nrows = int(np.ceil(Ndim / ncols))
    if instru_type == "IFU":
        valid_windows_reduced = valid_windows_3D.copy()
        for idim in (2, 1, 0):
            if orig_sizes[idim] == 1:
                valid_windows_reduced = np.take(valid_windows_reduced, 0, axis=idim)

    # Plot order: display r_WFE and IWA in the last panels when present
    dims_last  = [next((idim for idim, name in enumerate(params_names) if key in name), None) for key in ("r_WFE", "IWA")]
    dims_last  = [idim for idim in dims_last if idim is not None]
    plot_order = [idim for idim in range(Ndim) if idim not in dims_last] + dims_last


    if plot_global_results:
        # %%
        # Thermal/Reflected regime split of the planet table
        # Considering both thermal and reflected contribution
        if thermal_model != "None" and reflected_model != "None":
            mag_th   = np.asarray(planet_table[f"Planet{band_regime_plot}mag(thermal)"],   dtype=float)
            mag_re   = np.asarray(planet_table[f"Planet{band_regime_plot}mag(reflected)"], dtype=float)
            valid_th = np.isfinite(mag_th)
            valid_re = np.isfinite(mag_re)
            # Lower magnitude = brighter contribution.
            mask_thermal   = valid_th & valid_re & (mag_th < mag_re)
            mask_reflected = valid_th & valid_re & (mag_re < mag_th)
        # Thermal contribution only
        elif thermal_model != "None" and reflected_model == "None":
            light_regime_plot = "thermal"
            mask_thermal      = np.full(N_PT, True)
            mask_reflected    = ~mask_thermal
        # Reflected contribution only
        elif thermal_model == "None" and reflected_model != "None":
            light_regime_plot = "reflected"
            mask_reflected    = np.full(N_PT, True)
            mask_thermal      = ~mask_reflected
        else:
            raise ValueError("At least one of thermal_model or reflected_model must be different from 'None'.")
        mask_unknown   = ~(mask_thermal | mask_reflected)
        print(f"\nPlanet-light regime split in {band_regime_plot}-band:")
        print(f"                  => Thermal-dominated:   {mask_thermal.sum()}/{len(planet_table)}")
        print(f"                  => Reflected-dominated: {mask_reflected.sum()}/{len(planet_table)}")
        print(f"                  => Unknown regime:      {mask_unknown.sum()}/{len(planet_table)}")

        # DETECTION PROBABILITY PER TYPE AND REGIME
        N_PT_ptypes_thermal   = np.zeros((len(ptypes)))
        N_PT_ptypes_reflected = np.zeros((len(ptypes)))
        Pdet_ptypes_thermal   = [None] * len(ptypes)
        Pdet_ptypes_reflected = [None] * len(ptypes)
        mask_ptype_all        = np.full(len(planet_table), False)
        for ipt, ptype in enumerate(ptypes):
            mask_ptype                 = get_mask_planet_type(planet_table=planet_table, planet_type=ptype)
            mask_ptype_all            |= mask_ptype
            N_PT_ptypes_thermal[ipt]   = (mask_ptype & mask_thermal).sum()
            N_PT_ptypes_reflected[ipt] = (mask_ptype & mask_reflected).sum()
            if "thermal" in light_regime_plot:
                print(f"\n{label_ptypes[ptype]} in thermal light:")
                idx_ptype_thermal  = np.where(mask_ptype & mask_thermal)[0]
                Pdet_ptype_thermal = fyh.get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, indices=idx_ptype_thermal, jump_before_tqdm=False, valid_windows_3D=valid_windows_3D)
            else:
                Pdet_ptype_thermal = None
            if "reflected" in light_regime_plot:
                print(f"\n{label_ptypes[ptype]} in reflected light:")
                idx_ptype_reflected  = np.where(mask_ptype & mask_reflected)[0]
                Pdet_ptype_reflected = fyh.get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, indices=idx_ptype_reflected, jump_before_tqdm=False, valid_windows_3D=valid_windows_3D)
            else:
                Pdet_ptype_reflected = None
            for idim in sorted(dim_to_remove, reverse=True):
                if "thermal" in light_regime_plot:
                    Pdet_ptype_thermal = np.take(Pdet_ptype_thermal, 0, axis=idim)
                if "reflected" in light_regime_plot:
                    Pdet_ptype_reflected = np.take(Pdet_ptype_reflected, 0, axis=idim)
            Pdet_ptypes_thermal[ipt]   = Pdet_ptype_thermal
            Pdet_ptypes_reflected[ipt] = Pdet_ptype_reflected

        # Plot quantities for the considered ptypes and light_regime
        if light_regime_plot == "thermal":
            Pdet_ptypes_plot = Pdet_ptypes_thermal
            N_PT_ptypes_plot = N_PT_ptypes_thermal
            regime_label     = "thermal"
        elif light_regime_plot == "reflected":
            Pdet_ptypes_plot = Pdet_ptypes_reflected
            N_PT_ptypes_plot = N_PT_ptypes_reflected
            regime_label     = "reflected"
        elif light_regime_plot == "thermal+reflected":
            Pdet_ptypes_plot = []
            N_PT_ptypes_plot = []
            for ipt in range(len(Pdet_ptypes_thermal)):
                Pdet_th = Pdet_ptypes_thermal[ipt]
                Pdet_re = Pdet_ptypes_reflected[ipt]
                N_th    = N_PT_ptypes_thermal[ipt]
                N_re    = N_PT_ptypes_reflected[ipt]
                N_tot   = N_th + N_re
                if N_tot == 0:
                    Pdet_ptypes_plot.append(None)
                    N_PT_ptypes_plot.append(0)
                    continue
                Pdet_combined = np.zeros_like(Pdet_th, dtype=float)
                if N_th > 0:
                    Pdet_combined += N_th * Pdet_th
                if N_re > 0:
                    Pdet_combined += N_re * Pdet_re
                Pdet_combined /= N_tot
                Pdet_ptypes_plot.append(Pdet_combined)
                N_PT_ptypes_plot.append(N_tot)
            N_PT_ptypes_plot = np.asarray(N_PT_ptypes_plot)
            regime_label     = "thermal+reflected"

        # Combine all planet types into a single Pdet corresponding to light_regime_plot
        N_PT_plot = int(np.sum(N_PT_ptypes_plot))
        if N_PT_plot == 0:
            raise RuntimeError(f"No planets available for light_regime_plot='{light_regime_plot}'.")
        Pdet_plot = None
        for ipt in range(len(Pdet_ptypes_plot)):
            if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                continue
            if Pdet_plot is None:
                Pdet_plot = np.zeros_like(Pdet_ptypes_plot[ipt], dtype=float)
            Pdet_plot += N_PT_ptypes_plot[ipt] * Pdet_ptypes_plot[ipt]
        Pdet_plot /= N_PT_plot

        # Helper: normalize to the panel maximum or convert to yield
        def convert_Pdet_to_plot_quantity(Pdet_curve, N_PT, gain, ymax_panel=None):
            y = np.asarray(Pdet_curve, dtype=float)
            if gain:
                y = 100.0*y/ymax_panel if ymax_panel is not None and np.isfinite(ymax_panel) and ymax_panel > 0 else np.zeros_like(y)
            else:
                y = y*N_PT
            return y

        def get_panel_ymax(curves):
            values = np.concatenate([np.asarray(curve, dtype=float).ravel() for curve in curves]) if curves else np.array([])
            return np.nanmax(values) if np.any(np.isfinite(values)) else None



        # %%
        # 2D CORNER PLOT

        def set_log_decade_ticks(axis, vmin, vmax, max_labels=4):
            # Major ticks at every decade (the default LogLocator drops most of them on small panels spanning many decades),
            # with labels only every few decades so that they fit in the panel
            exps = np.arange(np.ceil(np.log10(vmin)), np.floor(np.log10(vmax)) + 1).astype(int)
            step = max(1, int(np.ceil(len(exps) / max_labels)))
            axis.set_major_locator(FixedLocator(10.0**exps))
            axis.set_major_formatter(FuncFormatter(lambda v, pos: rf"$\mathdefault{{10^{{{round(np.log10(v))}}}}}$" if round(np.log10(v)) % step == 0 else ""))
            axis.set_minor_locator(NullLocator())

        # Param values at max pdet_1D
        params_max  = np.zeros((Ndim))
        params_imax = np.zeros((Ndim), dtype=int)
        for idim in range(Ndim):
            pdet_1D           = fyh.reduce_hcube(hcube=Pdet_plot, dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names)
            params_imax[idim] = pdet_1D.argmax()
            params_max[idim]  = params[idim][pdet_1D.argmax()]

        xmin       = np.array([np.nanmin(param) for param in params])
        xmax       = np.array([np.nanmax(param) for param in params])
        levels     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cmap       = plt.get_cmap("plasma_r")
        fig, axes  = plt.subplots(Ndim, Ndim, figsize=(1.6 * Ndim, 1.6*Ndim), dpi=dpi_fig)
        axes       = np.atleast_2d(axes)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for ipanel, idim in enumerate(plot_order):
            for jpanel, jdim in enumerate(plot_order):
                ax = axes[ipanel, jpanel]
                ax.tick_params(axis="both", which="both", labelsize=12)
                if jpanel > ipanel:
                    ax.axis("off")
                    continue

                elif ipanel == jpanel:
                    pdet_1D = fyh.reduce_hcube(hcube=Pdet_plot, dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    ax.step(params[idim], pdet_1D, color="k", where="mid")
                    ax.axvline(params_max[idim], color="k", linestyle="--")
                    ax.set_yticks([])
                    if ipanel == Ndim - 1:
                        ax.set_xlabel(params_names_l[idim], fontsize=14)
                    if params_isint[idim]:
                        ax.set_title(f"{params_names_l[idim]} = {round(params_max[idim], -2):.0f}", fontsize=11)
                    elif params_islog[idim] and params_names[idim] != "IWA [lambda0/D]":
                        ax.set_title(f"{params_names_l[idim]} = {params_max[idim]:.1e}", fontsize=11)
                    else:
                        ax.set_title(f"{params_names_l[idim]} = {params_max[idim]:.2f}", fontsize=11)
                    ax.set_xlim(xmin[idim], xmax[idim])
                    if params_islog[idim]:
                        ax.set_xscale("log")
                    if params_names[idim] == "sigma_m [%]":
                        set_log_decade_ticks(ax.xaxis, xmin[idim], xmax[idim])

                elif jpanel < ipanel:
                    pdet_2D = fyh.reduce_hcube(hcube=Pdet_plot, dims_to_keep=[jdim, idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    pdet_2D = pdet_2D.T if jdim < idim else pdet_2D
                    pmax    = np.nanmax(pdet_2D)
                    if (not np.isfinite(pmax)) or (pmax <= 0):
                        ax.text(0.5, 0.5, "no detections", ha="center", va="center", transform=ax.transAxes)
                    else:
                        pdet_2D /= pmax
                        ax.contourf(params[jdim], params[idim], pdet_2D, levels=levels, cmap=cmap, alpha=0.8)
                        cs  = ax.contour(params[jdim], params[idim], pdet_2D, levels=levels, colors="black")
                        fmt = {lvl: f"{lvl:.0%}" for lvl in levels}
                        ax.clabel(cs, inline=True, fontsize=10, fmt=fmt)
                    ax.axvline(params_max[jdim], color="k", linestyle="--")
                    ax.axhline(params_max[idim], color="k", linestyle="--")
                    ax.plot(params_max[jdim], params_max[idim], "X", color="black")
                    if jpanel == 0:
                        ax.set_ylabel(params_names_l[idim], fontsize=14)
                    if ipanel == Ndim - 1:
                        ax.set_xlabel(params_names_l[jdim], fontsize=14)
                    ax.set_xlim(xmin[jdim], xmax[jdim])
                    ax.set_ylim(xmin[idim], xmax[idim])
                    if params_islog[jdim]:
                        ax.set_xscale("log")
                    if params_islog[idim]:
                        ax.set_yscale("log")
                    if params_names[jdim] == "sigma_m [%]":
                        set_log_decade_ticks(ax.xaxis, xmin[jdim], xmax[jdim])
                    if params_names[idim] == "sigma_m [%]":
                        set_log_decade_ticks(ax.yaxis, xmin[idim], xmax[idim])

                # Hide major and minor tick labels (log axes spanning < 1 decade label their minor ticks)
                if ipanel < Ndim - 1:
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                if jpanel > 0:
                    ax.tick_params(axis="y", which="both", labelleft=False)

        title  = f"ELT/{instru} detection probability"
        title += f"\n{exposure_time/60:.0f}hr - {instru_type} with {post_processing}"
        title += f"\n{regime_label} light regime"
        title += f"\n{N_PT_plot} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} planets"
        fig.suptitle(title, fontsize=18, weight="bold", x=0.63, y=0.89, linespacing=1.6)
        fig.savefig(sim_dir_concept / f"ELT_{instru}_{instru_type}_{post_processing}_corner_plot_{table_type}_{light_regime_plot}_Pdet.png", bbox_inches="tight", dpi=dpi_fig)
        plt.show(block=False)



        # %%
        # PLOT : 1D MARGINALIZED DETECTION YIELD/PROBABILITY GAIN PER PARAM, TYPE AND REGIME

        # Layout
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), dpi=dpi_fig, sharey=True)
        axes      = np.atleast_2d(axes)
        for ipanel, idim in enumerate(plot_order):
            r  = ipanel // ncols
            t  = ipanel % ncols
            ax = axes[r, t]
            ax.tick_params(axis="both", which="major", labelsize=fontsize)

            ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
            ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)

            # Compute all curves before normalizing them to the panel maximum
            curves = []
            for ipt, ptype in enumerate(ptypes):
                if "thermal" in light_regime_plot and N_PT_ptypes_thermal[ipt] > 0:
                    Pdet_thermal = fyh.reduce_hcube(hcube=Pdet_ptypes_thermal[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    curves.append((Pdet_thermal, N_PT_ptypes_thermal[ipt], ptype, "C3"))
                if "reflected" in light_regime_plot and N_PT_ptypes_reflected[ipt] > 0:
                    Pdet_reflected = fyh.reduce_hcube(hcube=Pdet_ptypes_reflected[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    curves.append((Pdet_reflected, N_PT_ptypes_reflected[ipt], ptype, "C0"))
            ymax_panel = get_panel_ymax([curve[0] for curve in curves]) if gain else None
            for Pdet_1D, N_PT_curve, ptype, color in curves:
                y = convert_Pdet_to_plot_quantity(Pdet_curve=Pdet_1D, N_PT=N_PT_curve, gain=gain, ymax_panel=ymax_panel)
                ax.plot(params[idim], y, ls="-", lw=lw, c=color, marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, alpha=alpha, zorder=3)

            # Axis formatting
            ax.set_xlim(np.nanmin(params[idim]), np.nanmax(params[idim]))
            if params_islog[idim]:
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_xlabel(params_names_L[idim], fontsize=fontsize + 4, labelpad=10)
            if t == 0:
                if gain:
                    ax.set_ylabel("Detection probability\ngain [%]", fontsize=fontsize + 4, labelpad=10)
                else:
                    ax.set_ylabel("Yield (number of planets detected)", fontsize=fontsize + 4, labelpad=10)
            else:
                ax.tick_params(labelleft=False)

            # Planet-type legend
            if ipanel ==2:
                handles_ptype = [Line2D([], [], ls="", marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, color="k", label=f"{label_ptypes[ptype]}") for ipt, ptype in enumerate(ptypes)]
                leg_ptype = ax.legend(handles=handles_ptype, fontsize='small', loc="upper left", frameon=True, edgecolor="gray", facecolor="white")#, title="Planet type", title_fontsize=fontsize + 4)
                ax.add_artist(leg_ptype)

            # Planet-regime legend
            if ipanel == 3 and light_regime_plot == "thermal+reflected":
                handles_regime = [Line2D([], [], ls="-", lw=lw + 2, color="C3", label="Thermal"), Line2D([], [], ls="-", lw=lw + 2, color="C0", label="Reflected")]
                leg_regime = ax.legend(handles=handles_regime, fontsize='small', loc="lower right", frameon=True, edgecolor="gray", facecolor="white")#, title="Planet-light regime", title_fontsize=fontsize + 4)
                ax.add_artist(leg_regime)

            # Systematics budget with JWST/MIRI/MRS ~ 1%.
            if params_names[idim] == "sigma_m [%]" and post_processing == "MM":
                x0 = 1.0 # [%]
                ax.axvline(x0, c="k", ls="--", lw=lw)
                ax.annotate("JWST/MIRI/MRS", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
            if params_names[idim] == "sigma_m [%]" and post_processing == "DI":
                x0 = 0.1  # [%] = 1e-3 in fractional units with VLT/SPHERE PACO
                ax.axvline(x0, c="k", ls="--", lw=lw)
                ax.annotate("Optimistic on-sky DI", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)

            # r_WFE and IWA ref values
            if params_names[idim] == "r_WFE":
                x0 = r_WFE_ref # [dimensionless]
                ax.axvline(x0, c="k", ls="--", lw=lw)
                ax.annotate("Nominal AO perf.", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
            if params_names[idim] == "IWA [lambda0/D]":
                x0 = IWA_ref  # [lambda0_band/D]
                ax.axvline(x0, c="k", ls="--", lw=lw)
                ax.annotate("PCS nominal FPM", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)

        # Turn off unused panels
        for k in range(Ndim, nrows * ncols):
            r = k // ncols
            t = k % ncols
            axes[r, t].axis("off")

        # Common y-scale
        axes[0, 0].set_yscale("log")
        if gain:
            axes[0, 0].set_ylim(1, 102)
        else:
            axes[0, 0].set_ylim(0.5, None)

        # Title
        quantity_name = "detection probability gain" if gain else "detection yield"
        title  = f"ELT/{instru} {quantity_name} in {exposure_time/60:.0f}hr - {instru_type} with {post_processing}"
        title += f"\n{N_PT} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} planets"
        fig.suptitle(title, fontsize=fontsize + 6, weight="bold", y=1.00, linespacing=1.6)
        fig.tight_layout(h_pad=3.0, w_pad=3.0)
        fig.savefig(sim_dir_concept / f"ELT_{instru}_{instru_type}_{post_processing}_detection_{table_type}_{light_regime_plot}_Pdet.png", bbox_inches="tight", dpi=dpi_fig)
        plt.show(block=False)



        # %%
        # Plot : 1D MARGINALIZED DETECTION YIELD/PROBABILITY GAIN PER PARAM, TYPE, REGIME AND BANDS

        ptypes_plot = ["Jupiter", "Saturn", "Neptune", "Earth"]
        bands_plot  = ["V", "R", "I", "z", "Y", "J", "H", "K"] # Choose the spectral bands to show

        # Identify lambda0 axis (required for this plot) and identify a l0 for each considered band
        idx_l0 = [idx for idx, param_name in enumerate(params_names) if "l0" in param_name]
        try:
            idx_l0 = idx_l0[0]
        except:
            raise RuntimeError("Could not identify the lambda0 axis in params_names / params_names_L.")
        l0_axis_values = params[idx_l0]
        band_l0_values = np.array([(lmin_bands[band]+lmax_bands[band])/2 for band in bands_plot], dtype=float)

        # Keep only the bands whose central wavelength is within the sampled lambda0 range
        in_range = (band_l0_values >= l0_axis_values[0]) & (band_l0_values <= l0_axis_values[-1])
        if not np.all(in_range):
            dropped = [band for band, ok in zip(bands_plot, in_range) if not ok]
            print(f"Warning: removing band(s) {dropped} from the plot (central wavelength outside the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm).")
            bands_plot     = [band for band, ok in zip(bands_plot, in_range) if ok]
            band_l0_values = band_l0_values[in_range]
        if len(bands_plot) == 0:
            raise ValueError(f"None of the selected bands has a central wavelength within the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm.")
        NbBand      = len(bands_plot)
        band_labels = [f"{band}\n$\\lambda_0$={l0_band:.2f} µm" for band, l0_band in zip(bands_plot, band_l0_values)]
        cmap        = plt.get_cmap("rainbow", NbBand)

        # loop over the different types of planets to make one plot for each
        for ptype in ptypes_plot:

            # Layout
            fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), dpi=dpi_fig, sharey=True)
            axes      = np.atleast_2d(axes)
            for ipanel, idim in enumerate(plot_order):
                r  = ipanel // ncols
                t  = ipanel % ncols
                ax = axes[r, t]
                ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
                ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)
                ax.tick_params(axis="both", which="major", labelsize=fontsize)

                # Compute all curves (before normalizing them to the panel maximum if gain)
                curves = []
                if idim == idx_l0:
                    ipt = ptypes.index(ptype)
                    if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                        continue
                    Pdet_1D = fyh.reduce_hcube(hcube=Pdet_ptypes_plot[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    curves.append((Pdet_1D, N_PT_ptypes_plot[ipt], ptype, "k"))
                else:
                    ipt = ptypes.index(ptype)
                    if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                        continue
                    for iband, l0_band in enumerate(band_l0_values):
                        params_ranges_band         = [tuple(rng) for rng in params_ranges]
                        params_ranges_band[idx_l0] = (l0_band, l0_band)
                        Pdet_1D                    = fyh.reduce_hcube(hcube=Pdet_ptypes_plot[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges_band, params_priors=params_priors, params_names=params_names, verbose=False)
                        curves.append((Pdet_1D, N_PT_ptypes_plot[ipt], ptype, cmap(iband)))
                ymax_panel = get_panel_ymax([curve[0] for curve in curves]) if gain else None
                for Pdet_1D, N_PT_curve, ptype, color in curves:
                    y = convert_Pdet_to_plot_quantity(Pdet_curve=Pdet_1D, N_PT=N_PT_curve, gain=gain, ymax_panel=ymax_panel)
                    ax.plot(params[idim], y, ls="-", lw=lw, c=color, marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, alpha=alpha, zorder=3)
                if idim == idx_l0:
                    for iband, l0_band in enumerate(band_l0_values):
                        ax.axvline(l0_band, c=cmap(iband), ls="-", lw=3*lw, alpha=0.3, zorder=2)

                # Axis formatting
                ax.set_xlim(np.nanmin(params[idim]), np.nanmax(params[idim]))
                if params_islog[idim]:
                    ax.set_xscale("log")
                    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                else:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.set_xlabel(params_names_L[idim], fontsize=fontsize + 2, labelpad=10)
                if t == 0:
                    if gain:
                        ax.set_ylabel("Detection probability gain [%]", fontsize=fontsize + 2, labelpad=10)
                    else:
                        ax.set_ylabel("Yield (number of planets detected)", fontsize=fontsize + 2, labelpad=10)
                else:
                    ax.tick_params(labelleft=False)

                # Planet-type legend
                # if ipanel == 0:
                #     handles_ptype = [Line2D([], [], ls="", marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, color="k", label=label_ptypes[ptype]) for ptype in ptypes_plot]
                #     leg_ptype     = ax.legend(handles=handles_ptype, fontsize=fontsize + 2, loc="upper right", frameon=True, edgecolor="gray", facecolor="white", title="Planet type", title_fontsize=fontsize + 4)
                #     ax.add_artist(leg_ptype)

                # Systematics budget with JWST/MIRI/MRS ~ 1%.
                if params_names[idim] == "sigma_m [%]" and post_processing == "MM":
                    x0 = 1.0 # [%]
                    ax.axvline(x0, c="k", ls="--", lw=lw)
                    ax.annotate("JWST/MIRI/MRS", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
                if params_names[idim] == "sigma_m [%]" and post_processing == "DI":
                    x0 = 0.1  # [%] = 1e-3 in fractional units with VLT/SPHERE PACO
                    ax.axvline(x0, c="k", ls="--", lw=lw)
                    ax.annotate("Optimistic on-sky DI", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)

                # r_WFE and IWA ref values
                if params_names[idim] == "r_WFE":
                    x0 = r_WFE_ref # [dimensionless]
                    ax.axvline(x0, c="k", ls="--", lw=lw)
                    ax.annotate("Nominal AO perf.", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
                if params_names[idim] == "IWA [lambda0/D]":
                    x0 = IWA_ref  # [lambda0_band/D]
                    ax.axvline(x0, c="k", ls="--", lw=lw)
                    ax.annotate("PCS nominal FPM", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize='small', color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)

            # Turn off unused panels
            for k in range(Ndim, nrows * ncols):
                r = k // ncols
                t = k % ncols
                axes[r, t].axis("off")

            # Common y-scale
            axes[0, 0].set_yscale("log")
            if gain:
                axes[0, 0].set_ylim(1, 102)
            else:
                axes[0, 0].set_ylim(0.5, None)

            # Common discrete colorbar for selected bands
            norm = mpl.colors.BoundaryNorm(boundaries=np.arange(NbBand + 1) - 0.5, ncolors=NbBand)
            sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", fraction=0.025, pad=0.05, ticks=np.arange(NbBand))
            cbar.ax.set_yticklabels(band_labels)
            cbar.set_label(r"Selected spectral band / fixed central wavelength $\lambda_0$", fontsize=fontsize + 2, rotation=270, labelpad=35)
            cbar.ax.tick_params(labelsize=fontsize)

            # Title
            if gain:
                quantity_name = "detection probability gain"
            else:
                quantity_name = "detection yield"
            N_PT_ptype_plot_1D = int(N_PT_ptypes_plot[ptypes.index(ptype)])
            title  = f"ELT/{instru} {quantity_name} in {exposure_time/60:.0f}hr - {instru_type} with {post_processing}"
            title += f"\n{light_regime_plot} planet-light regime"
            title += f"\n{N_PT_ptype_plot_1D} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} {ptype}-like planets"
            fig.suptitle(title, fontsize=fontsize + 6, weight="bold", y=1.00, linespacing=1.6)
            fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.88, wspace=0.15, hspace=0.3)
            fig.savefig(sim_dir_concept / f"ELT_{instru}_{instru_type}_{post_processing}_detection_band_{table_type}_{light_regime_plot}_{ptype}_Pdet.png", bbox_inches="tight", dpi=dpi_fig)
            plt.show(block=False)



        #%%
        # HEATMAP: BAND x RESOLUTION / BANDWIDTH HEATMAPS PER PLANET TYPE AND LIGHT REGIME

        ptypes_heatmap = ["Jupiter", "Saturn", "Neptune", "Earth"]
        bands_plot     = ["V", "R", "I", "z", "Y", "J", "H", "K"]
        heatmap_mode   = "marginalized" # "marginalized" or "max"
        regimes_plot   = [regime for regime in ["thermal", "reflected"] if regime in light_regime_plot]
        if not regimes_plot:
            raise RuntimeError("No valid regime to plot. light_regime_plot must contain 'thermal' and/or 'reflected'.")

        # Choose the second heatmap axis depending on the instrument type
        if post_processing == "MM": #instru_type == "IFU":
            spec_plot      = [1_000, 3_000, 10_000, 30_000, 100_000]
            spec_plot      = [50_000, 75_000, 100_000, 150_000, 200_000]
            idx_spec       = next((idx for idx, name in enumerate(params_names) if "R" in name), None)
            spec_axis_name = "Resolution"
        else:
            spec_plot      = [0.1, 0.3, 1.0, 3.0, 10.0]
            idx_spec       = next((idx for idx, name in enumerate(params_names) if "sigma_m" in name), None)
            spec_axis_name = r"Speckle residuals $\sigma_m$ [%]"

        idx_l0 = next((idx for idx, name in enumerate(params_names) if "l0" in name), None)
        if idx_l0 is None:
            raise RuntimeError("Could not identify the lambda0 axis in params_names.")
        if idx_spec is None:
            raise RuntimeError("Could not identify the secondary spectral axis in params_names.")
        if heatmap_mode not in {"marginalized", "max"}:
            raise ValueError("heatmap_mode must be 'marginalized' or 'max'.")

        l0_axis_values   = np.asarray(params[idx_l0],   dtype=float)
        spec_axis_values = np.asarray(params[idx_spec], dtype=float)
        band_l0_values   = np.array([(lmin_bands[band] + lmax_bands[band]) / 2 for band in bands_plot], dtype=float)
        in_range         = (band_l0_values >= l0_axis_values[0]) & (band_l0_values <= l0_axis_values[-1])
        if not np.all(in_range):
            dropped = [band for band, ok in zip(bands_plot, in_range) if not ok]
            print(f"Warning: removing band(s) {dropped} from the heatmap (central wavelength outside the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm).")
            bands_plot     = [band for band, ok in zip(bands_plot, in_range) if ok]
            band_l0_values = band_l0_values[in_range]
        if len(bands_plot) == 0:
            raise ValueError(f"None of the selected bands has a central wavelength within the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm.")
        spec_values      = np.asarray(spec_plot, dtype=float)
        spec_values      = np.clip(spec_values, spec_axis_values[0], spec_axis_values[-1])
        spec_plot        = spec_values.tolist()
        spec_plot_labels = [f"{spec:g}" for spec in spec_plot] if instru_type == "IFU" else [f"{spec:.3f}" for spec in spec_plot]

        regime_to_Pdet = {"thermal": Pdet_ptypes_thermal, "reflected": Pdet_ptypes_reflected}
        regime_to_NPT  = {"thermal": N_PT_ptypes_thermal, "reflected": N_PT_ptypes_reflected}
        dims_keep_max  = [idim for idim in range(Ndim) if idim not in (idx_spec, idx_l0)]
        heatmaps       = {}

        for ptype in ptypes_heatmap:
            ipt = ptypes.index(ptype)
            for regime in regimes_plot:
                Pdet_cube = regime_to_Pdet[regime][ipt]
                N_PT_reg  = regime_to_NPT[regime][ipt]
                heatmap   = np.full((len(spec_plot), len(bands_plot)), np.nan, dtype=float)
                if Pdet_cube is not None and N_PT_reg > 0:
                    for ispec, spec_fixed in enumerate(spec_plot):
                        for iband, l0_band in enumerate(band_l0_values):
                            ranges           = [tuple(rng) for rng in params_ranges]
                            ranges[idx_l0]   = (l0_band, l0_band)
                            ranges[idx_spec] = (spec_fixed, spec_fixed)
                            if heatmap_mode == "marginalized":
                                Pdet_value            = fyh.reduce_hcube(hcube=Pdet_cube, dims_to_keep=[], params=params, params_ranges=ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                                heatmap[ispec, iband] = float(np.asarray(Pdet_value)) * N_PT_reg
                            else:
                                Pdet_subcube          = np.asarray(fyh.reduce_hcube(hcube=Pdet_cube, dims_to_keep=dims_keep_max, params=params, params_ranges=ranges, params_priors=params_priors, params_names=params_names, verbose=False), dtype=float)
                                heatmap[ispec, iband] = float(np.nanmax(Pdet_subcube)) * N_PT_reg if np.any(np.isfinite(Pdet_subcube)) else np.nan
                heatmaps[(ptype, regime)] = heatmap

        panel_keys   = [(ptype, regime) for ptype in ptypes_heatmap for regime in regimes_plot]
        panel_labels = {(ptype, regime): f"{label_ptypes[ptype]} — {regime.capitalize()}" for ptype, regime in panel_keys}
        title        = f"ELT/{instru} band × {'resolution' if instru_type == 'IFU' else 'speckle residuals'} yield - {instru_type} with {post_processing}"#"\nheatmap mode: {heatmap_mode}"

        yield_heatmap_ELT(instru=instru, exposure_time=exposure_time, heatmaps=heatmaps, ptypes_heatmap=panel_keys, bands_plot=bands_plot, config_labels=spec_plot_labels, config_axis_name=spec_axis_name, x_axis_name="Spectral band", panel_labels=panel_labels, title=title, save_dir=sim_dir_concept, filename=f"ELT_{instru}_{instru_type}_{post_processing}_band_spectral_heatmap_grid_{table_type}_{light_regime_plot}_{heatmap_mode}.png")


    if plot_populations:
        #%%
        # POPULATION DIAGNOSTICS

        idx_FoV      = next(idx for idx, name in enumerate(params_names) if "FoV"     in name)
        idx_sigma_m  = next(idx for idx, name in enumerate(params_names) if "sigma_m" in name)
        idx_l0       = next(idx for idx, name in enumerate(params_names) if "l0"      in name)
        noise_labels = np.array(["Stellar halo", "Background", "Read noise", "Dark current", str(residuals)], dtype=object)

        # Parameter ranges used specifically for population diagnostics
        params_ranges_pop = [tuple(rng) for rng in params_ranges]
        if snr_population_mode == "personalized":
            personalized_ranges = {"R": R_plot, "l0": l0_plot, "Dl": Dl_plot, "r_WFE": r_WFE_plot, "IWA": IWA_plot, "trans_instru": trans_instru_plot, "sigma_m": sigma_m_plot, "FoV": FoV_plot}
            allowed_keys        = {"IFU": {"R", "l0", "Dl", "r_WFE", "IWA", "trans_instru", "sigma_m", "FoV"}, "imager": {"l0", "Dl", "r_WFE", "IWA", "trans_instru", "sigma_m", "FoV"}}[instru_type]
            def format_range(r):
                return f"{r[0]:g}" if np.isclose(r[0], r[1]) else f"[{r[0]:g}, {r[1]:g}]"
            def format_value(value):
                if value is None:
                    return "None"
                value = np.asarray(value).ravel()
                return f"{value[0]:g}" if value.size == 1 else f"[{value[0]:g}, {value[1]:g}]"
            def print_range(key, value, used=None, warning=""):
                used_str = f"-> {used}" if used is not None else ""
                print(f"  {key+'_plot':<18} = {format_value(value):<14} {used_str:<14} {warning}")
            print("\nPopulation plot personalized ranges:")
            for key, value in personalized_ranges.items():
                if key not in allowed_keys:
                    print_range(key, value, warning=f"(WARNING: not used for {instru_type})")
                    continue
                idx = next((idx for idx, name in enumerate(params_names) if name == key or key in name), None)
                # Dimension existed originally but was removed because it was fixed
                if idx is None:
                    idx_orig = next((idx for idx, name in enumerate(orig_names) if name == key or key in name), None)
                    if idx_orig is not None:
                        fixed_value = float(np.asarray(orig_params[idx_orig])[0])
                        requested   = np.asarray(value, dtype=float).ravel() if value is not None else np.array([])
                        changed     = value is not None and (requested.size != 1 or not np.isclose(requested[0], fixed_value))
                        print_range(key, value, f"{fixed_value:g}" if changed else None, f"(WARNING: parameter fixed at {fixed_value:g})")
                    else:
                        print_range(key, value, warning="(WARNING: parameter unavailable)")
                    continue
                # None -> keep default range
                if value is None:
                    print_range(key, value, format_range(params_ranges_pop[idx]))
                    continue
                axis      = np.asarray(params[idx], dtype=float)
                requested = np.asarray(value, dtype=float).ravel()
                if requested.size == 1:
                    requested = np.repeat(requested, 2)
                elif requested.size != 2:
                    print_range(key, value, format_range(params_ranges_pop[idx]), "(WARNING: expected scalar or [min, max]; default used)")
                    continue
                warning = ""
                if requested[0] > requested[1]:
                    requested = requested[::-1]
                    warning   = "(WARNING: reversed range reordered)"
                used                   = np.clip(requested, axis[0], axis[-1])
                params_ranges_pop[idx] = (float(used[0]), float(used[1]))
                changed = not np.allclose(requested, used)
                if changed:
                    warning = f"(WARNING: cropped to [{axis[0]:g}, {axis[-1]:g}])"
                print_range(key, value, format_range(params_ranges_pop[idx]) if changed else None, warning)
            print()

        # Best-Pdet configuration
        if snr_population_mode == "max":
            params_imax                             = np.array(np.unravel_index(np.nanargmax(Pdet), Pdet.shape), dtype=int)
            params_imax_snr                         = [params_imax[idim] for idim in range(Ndim) if idim != idx_FoV]
            SNR_plot                                = np.asarray(SNR_planets.squeeze()[(slice(None),) + tuple(params_imax_snr)], dtype=float).copy()
            SNR_plot[p0_FoV > params_imax[idx_FoV]] = 0.0

            # Raw arrays do not contain sigma_m or FoV
            params_imax_raw   = [params_imax[idim] for idim in range(Ndim) if idim not in (idx_sigma_m, idx_FoV)]
            raw_selection     = (slice(None),) + tuple(params_imax_raw)
            sigma_halo_2_best = np.asarray(sigma_halo_2_planets.squeeze()[raw_selection], dtype=float)
            sigma_bkg_2_best  = np.asarray(sigma_bkg_2_planets.squeeze()[raw_selection], dtype=float)
            DIT_best          = np.asarray(DIT_planets.squeeze()[raw_selection], dtype=float)

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
                sigma_syst_base_2_best = np.asarray(sigma_syst_base_2_planets.squeeze()[raw_selection], dtype=float)
            elif instru_type == "imager":
                sigma_syst_base_2_best = sigma_halo_2_best**2
            else:
                raise ValueError("instru_type must be 'IFU' or 'imager'.")

            var_halo = N_DIT_best * sigma_halo_2_best
            var_bkg  = N_DIT_best * sigma_bkg_2_best
            var_RON  = N_DIT_best * sigma_RON_2_best
            var_DC   = N_DIT_best * sigma_DC_2_best
            var_syst = N_DIT_best**2 * sigma_m_best**2 * sigma_syst_base_2_best

        # Marginalized parameter space
        elif snr_population_mode in ("marginalized", "personalized"):
            non_FoV_dims      = [idim                    for idim in range(Ndim) if idim != idx_FoV]
            params_snr        = [params[idim]            for idim in non_FoV_dims]
            params_ranges_snr = [params_ranges_pop[idim] for idim in non_FoV_dims]
            params_priors_snr = [params_priors[idim]     for idim in non_FoV_dims]
            params_names_snr  = [params_names[idim]      for idim in non_FoV_dims]

            SNR_plot = np.zeros(N_PT, dtype=float)
            for ip in tqdm(range(N_PT), desc="Marginalizing SNR over parameter grid"):
                SNR_i        = np.asarray(SNR_planets[ip], dtype=float).squeeze()
                if instru_type == "IFU":
                    mask_shape = (valid_windows_reduced.shape + (1,) * (SNR_i.ndim - valid_windows_reduced.ndim))
                    SNR_i      = np.where(valid_windows_reduced.reshape(mask_shape), SNR_i, np.nan)
                SNR_plot[ip] = float(fyh.reduce_hcube(hcube=SNR_i, dims_to_keep=[], params=params_snr, params_ranges=params_ranges_snr, params_priors=params_priors_snr, params_names=params_names_snr, verbose=False))

            # FoV marginalization
            idx_FoV_range, w_FoV = fyh.get_axis_weights_in_range(axis=np.asarray(params[idx_FoV], dtype=float), pmin=params_ranges_pop[idx_FoV][0], pmax=params_ranges_pop[idx_FoV][1], prior=params_priors[idx_FoV])
            w_FoV               /= np.sum(w_FoV)
            FoV_weight_planets   = np.zeros(N_PT, dtype=float)
            for iFoV, weight in zip(idx_FoV_range, w_FoV):
                FoV_weight_planets[p0_FoV <= iFoV] += weight
            SNR_plot *= FoV_weight_planets

            # Raw quantities contain neither sigma_m nor FoV
            raw_dims          = [idim for idim in range(Ndim) if idim not in (idx_sigma_m, idx_FoV)]
            params_raw        = [params[idim]            for idim in raw_dims]
            params_ranges_raw = [params_ranges_pop[idim] for idim in raw_dims]
            params_priors_raw = [params_priors[idim]     for idim in raw_dims]
            params_names_raw  = [params_names[idim]      for idim in raw_dims]
            idx_l0_raw        = raw_dims.index(idx_l0)

            idx_sigma_m_range, w_sigma_m = fyh.get_axis_weights_in_range(axis=np.asarray(params[idx_sigma_m], dtype=float), pmin=params_ranges_pop[idx_sigma_m][0], pmax=params_ranges_pop[idx_sigma_m][1], prior=params_priors[idx_sigma_m])
            w_sigma_m                   /= np.sum(w_sigma_m)
            sigma_m_axis                 = np.asarray(params[idx_sigma_m], dtype=float)
            mean_sigma_m_2               = np.sum(w_sigma_m * (sigma_m_axis[idx_sigma_m_range] / 100.0)**2)

            shape_l0             = [1] * len(raw_dims)
            shape_l0[idx_l0_raw] = len(params[idx_l0])
            shape_l0             = tuple(shape_l0)
            RON0_grid            = np.asarray(RON0, dtype=float).reshape(shape_l0)
            RON_lim_grid         = np.asarray(RON_lim, dtype=float).reshape(shape_l0)
            DC0_grid             = np.asarray(DC0, dtype=float).reshape(shape_l0)
            min_DIT_grid         = np.asarray(min_DIT, dtype=float).reshape(shape_l0)

            def marginalize_raw(quantity):
                quantity = np.asarray(quantity, dtype=float)
                if instru_type == "IFU":
                    mask_shape = (valid_windows_reduced.shape + (1,) * (quantity.ndim - valid_windows_reduced.ndim))
                    quantity   = np.where(valid_windows_reduced.reshape(mask_shape), quantity, np.nan)
                return float(fyh.reduce_hcube(hcube=quantity, dims_to_keep=[], params=params_raw, params_ranges=params_ranges_raw, params_priors=params_priors_raw, params_names=params_names_raw, verbose=False))

            var_halo = np.zeros(N_PT, dtype=float)
            var_bkg  = np.zeros(N_PT, dtype=float)
            var_RON  = np.zeros(N_PT, dtype=float)
            var_DC   = np.zeros(N_PT, dtype=float)
            var_syst = np.zeros(N_PT, dtype=float)
            N_DIT_marg        = np.zeros(N_PT, dtype=float)
            signal_total_marg = np.zeros(N_PT, dtype=float)

            for ip in tqdm(range(N_PT), desc="Marginalizing dominant-noise terms"):
                signal_i        = np.nan_to_num(np.asarray(signal_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
                sigma_halo_2_i = np.nan_to_num(np.asarray(sigma_halo_2_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
                sigma_bkg_2_i  = np.nan_to_num(np.asarray(sigma_bkg_2_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
                DIT_i          = np.asarray(DIT_planets[ip], dtype=float).squeeze()
                valid_DIT      = np.isfinite(DIT_i) & (DIT_i > 0)
                DIT_safe       = np.where(valid_DIT, DIT_i, 1.0)
                RON0_i         = np.broadcast_to(RON0_grid, DIT_safe.shape)
                RON_lim_i      = np.broadcast_to(RON_lim_grid, DIT_safe.shape)
                DC0_i          = np.broadcast_to(DC0_grid, DIT_safe.shape)
                min_DIT_i      = np.broadcast_to(min_DIT_grid, DIT_safe.shape)
                N_DIT_i        = np.where(valid_DIT, np.clip(np.floor(exposure_time / DIT_safe), 1, None), 0.0)
                N_read_i       = np.where(valid_DIT, np.floor(DIT_safe / min_DIT_i), 0).astype(np.int32)
                sigma_RON_2_i  = RON0_i**2
                mask_read      = N_read_i >= 2

                if np.any(mask_read):
                    n_read                   = N_read_i[mask_read].astype(float)
                    sigma_RON_2_i[mask_read] = RON0_i[mask_read]**2 * 12 * (n_read - 1) / (n_read * (n_read + 1)) + RON_lim_i[mask_read]**2

                sigma_RON_2_i *= A_FWHM
                sigma_RON_2_i  = np.where(valid_DIT, sigma_RON_2_i, 0.0)
                sigma_DC_2_i   = np.where(valid_DIT, DC0_i * DIT_safe * A_FWHM, 0.0)

                if instru_type == "IFU":
                    sigma_syst_base_2_i = np.nan_to_num(np.asarray(sigma_syst_base_2_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
                elif instru_type == "imager":
                    sigma_syst_base_2_i = sigma_halo_2_i**2
                else:
                    raise ValueError("instru_type must be 'IFU' or 'imager'.")

                var_halo[ip]         = marginalize_raw(N_DIT_i * sigma_halo_2_i)
                var_bkg[ip]          = marginalize_raw(N_DIT_i * sigma_bkg_2_i)
                var_RON[ip]          = marginalize_raw(N_DIT_i * sigma_RON_2_i)
                var_DC[ip]           = marginalize_raw(N_DIT_i * sigma_DC_2_i)
                var_syst[ip]         = marginalize_raw(N_DIT_i**2 * mean_sigma_m_2 * sigma_syst_base_2_i)
                N_DIT_marg[ip]       = marginalize_raw(N_DIT_i)
                signal_total_marg[ip] = marginalize_raw(N_DIT_i * signal_i)

            signal_INSTRU     = np.full(N_PT, np.nan, dtype=float)
            sigma_fund_INSTRU = np.full(N_PT, np.nan, dtype=float)
            sigma_syst_INSTRU = np.full(N_PT, np.nan, dtype=float)
            DIT_INSTRU        = np.full(N_PT, np.nan, dtype=float)

            var_fund = var_halo + var_bkg + var_RON + var_DC
            valid    = np.isfinite(N_DIT_marg) & np.isfinite(signal_total_marg) & np.isfinite(var_fund) & np.isfinite(var_syst) & (N_DIT_marg > 0) & (signal_total_marg > 0) & (var_fund >= 0) & (var_syst >= 0)

            DIT_INSTRU[valid]        = exposure_time / N_DIT_marg[valid]
            signal_INSTRU[valid]     = signal_total_marg[valid] / N_DIT_marg[valid]
            sigma_fund_INSTRU[valid] = np.sqrt(var_fund[valid] / N_DIT_marg[valid])
            sigma_syst_INSTRU[valid] = np.sqrt(var_syst[valid]) / N_DIT_marg[valid]

            if instru_type == "IFU" and post_processing == "MM":
                path = archive_path if table_type == "Archive" else simulated_path
                fyh.save_marginalized_planet_table(
                    planet_table=planet_table,
                    signal=signal_INSTRU,
                    sigma_fund=sigma_fund_INSTRU,
                    sigma_syst=sigma_syst_INSTRU,
                    DIT=DIT_INSTRU,
                    SNR=SNR_plot,
                    exposure_time=exposure_time,
                    table=table_type,
                    instru=instru,
                    apodizer=apodizer,
                    strehl=strehl,
                    coronagraph=coronagraph,
                    name_model=name_model,
                    path=path,
                )

        else:
            raise ValueError("snr_population_mode must be 'max', 'marginalized' or 'personalized'.")

        noise_stack                 = np.vstack([var_halo, var_bkg, var_RON, var_DC, var_syst])
        valid_noise                 = np.any(np.isfinite(noise_stack), axis=0)
        dominant_noise              = np.full(N_PT, "Unknown", dtype=object)
        dominant_noise[valid_noise] = noise_labels[np.nanargmax(noise_stack[:, valid_noise], axis=0)]

        yield_population_plot(table=table_type, instru=instru, instru_type=instru_type, post_processing=post_processing, thermal_model=thermal_model, reflected_model=reflected_model, exposure_time=exposure_time, band_contrast_plot=band_contrast_plot, band_regime_plot=band_regime_plot, planet_table=planet_table, SNR_plot=SNR_plot, dominant_noise=dominant_noise, SNR_thr=SNR_thr, save_dir=sim_dir_concept, DL_mas=np.nan, show_golden_sample=show_golden_sample, show_det_regions=show_det_regions, show_ptype_regions=show_ptype_regions, space=planet_population_space)


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

