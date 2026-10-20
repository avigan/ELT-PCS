# Convert the new PCS SUMMARY radial products into the same FITS interface
# expected by PCS_FastYield.py.
#
# Expected FastYield output axes:
#   PSF_profile_density_5D : (wave, r_WFE, IWA, mag_star, separation) [fraction / mas^2]
#   fraction_core_5D       : (wave, r_WFE, IWA, mag_star, separation) [fraction]
#   radial_transmission_5D : (wave, r_WFE, IWA, mag_star, separation) [fraction] (coronagraph only)
# Here IWA is dimensionless and expressed in lambda0_band/D.
#
# New SUMMARY inputs:
#   IMG  : direct-imaging/non-coronagraphic reference
#   CLC  : classical Lyot coronagraph
#   APLC : apodized-pupil Lyot coronagraph
#
# Source-data conventions / remaining limitations:
#   1) MAGxxx is a V-band stellar magnitude axis.
#   2) the coronagraphic FPM radius used by the simulations is 2.4*lambda0_band/D,
#      where lambda0_band is the centre of the broad V/R/I/Y/J/H/K band containing
#      the simulated wavelength. The band centres are derived from FastYield
#      lmin_bands/lmax_bands.
#   3) WFE is estimated from a spatial Strehl fit, as in the old PCS script.
#      The inferred diffraction-limited PSF is model dependent.
#   4) SUMMARY core_throughput comes from the upstream photometric aperture
#      (photometric_mask_size=2.4). Here it is deliberately treated as the
#      absolute planet-core throughput used by FastYield.

from pathlib import Path
import re
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from astropy.io import fits
from tqdm import tqdm

from fastyield.config import rad2arcsec, get_sim_data_path, lmin_bands, lmax_bands
from fastyield.utils import interp_extrap_sep, build_PSF_grid, add_if_necessary, plot_profiles


# %%
# =============================================================================
# USER PARAMETERS
# =============================================================================

instru       = "PCS"
strehl_label = "Q2"     # filename/interface label used by the current PCS_FastYield.py
size_core    = 2        # [px/FWHM], same convention as PCS_FastYield.py
D            = 38.54    # [m], keep consistent with PCS_FastYield.py
sep_unit     = "mas"

ROOT = Path(__file__).resolve().parent

# Put SUMMARY/ or SUMMARY.zip next to this script, or set SUMMARY_INPUT explicitly.
SUMMARY_INPUT = "/home/martoss/Documents/PhD/work/MAIN/data/PCS/SUMMARY"

def resolve_summary_dir(path):
    path = Path(path)
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_root = ROOT / "_SUMMARY_extracted"
        summary_root = extract_root / "SUMMARY"
        extract_root.mkdir(parents=True, exist_ok=True)
        with ZipFile(path, "r") as zf:
            zf.extractall(extract_root)
        if not summary_root.is_dir():
            raise FileNotFoundError(f"Could not find SUMMARY/ inside {path}")
        return summary_root
    raise FileNotFoundError(f"SUMMARY input not found: {path}")

summary_dir = resolve_summary_dir(SUMMARY_INPUT)
output_dir  = get_sim_data_path() / instru / "PSF_simulations"
output_dir.mkdir(parents=True, exist_ok=True)

# Output ranges: intentionally kept identical to the ranges currently hard-coded
# in PCS_FastYield.py.
lmin, lmax                 = 0.5, 2.5       # [um]
r_WFE_min, r_WFE_max       = 0.2, 2.0       # [dimensionless]
IWA_min, IWA_max           = 1.0, 5.0       # [lambda0_band/D]
mag_star_min, mag_star_max = 0, 10
sep_max                    = 1_000          # [mas]

N_r_WFE_base = 20
N_IWA_base = 20
N_sep_log  = 1_000

# SUMMARY wavelength axis is implicit in README.md.
raw_wave_min  = 0.5 # [um]
raw_wave_step = 0.1 # [um]

# Absolute WFE is not contained in SUMMARY. Keep the previous PCS bright-star
# anchor, then infer the relative degradation with stellar magnitude from IMG.
# Used only as an external comparison diagnostic (not as an AO calibration).
STREHL_REFERENCE_MODE = "CLC" # CLC is less aggressively apodized than APLC

# AO simulation magnitude convention.
PSF_MAG_BAND = "V"

# PCS coronagraphic mask prescription from the updated simulation information.
# Each wavelength uses the FPM associated with its broad photometric band:
#     IWA_nominal(lambda) = FPM_LAMD * lambda0_band / D.
FPM_LAMD  = 2.4 # [lambda0_band/D]
IWA_ref   = FPM_LAMD
FPM_BANDS = ("V", "R", "I", "Y", "J", "H", "K")

# Output naming. CLC preserves the current PCS configuration exactly.
mode_to_fastyield = {
    "CLC":  {"apodizer": "NO_SP", "coronagraph":  "CLC"},
    "APLC": {"apodizer": "NO_SP",  "coronagraph": "APLC"},
}
coronagraph_modes = ["CLC", "APLC"]

# Diagnostics
show_plots          = True
profile_outer_slope = -3.0 # imposed r^-3 extrapolation beyond the available radial profile
EPS                 = 1e-8


# %%
# =============================================================================
# HELPERS
# =============================================================================

def radial_centers_to_edges(r):
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size < 2 or np.any(np.diff(r) <= 0) or r[0] < 0:
        raise ValueError("Radial centres must be a strictly increasing 1D array with at least two points.")
    edges = np.empty(r.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0]    = 0.0
    edges[-1]   = r[-1] + 0.5 * (r[-1] - r[-2])
    return edges


def annulus_areas(r):
    edges = radial_centers_to_edges(r)
    return np.pi * (edges[1:]**2 - edges[:-1]**2)


def safe01(x, eps=EPS):
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def safe_pos(x):
    return np.clip(np.asarray(x, dtype=float), np.finfo(float).tiny, None)


def normalize_radial_profile(profile, separation):
    """Normalize (..., separation) radial densities to unit annular integral."""
    profile = safe_pos(profile)
    area    = annulus_areas(separation)
    norm    = np.sum(profile * area, axis=-1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("Cannot normalize a radial profile with a non-finite/non-positive integral.")
    return profile / norm[..., None]


def parse_magnitude_labels(mode_dir):
    labels = []
    for path in sorted(mode_dir.glob("stellar_flux_MAG*.txt")):
        m = re.search(r"MAG(\d+)$", path.stem)
        if m:
            labels.append(m.group(1))
    if not labels:
        raise FileNotFoundError(f"No stellar_flux_MAG*.txt files found in {mode_dir}")
    labels = sorted(set(labels), key=int)
    mags   = np.array([int(label) / 10.0 for label in labels], dtype=float)
    return labels, mags


def load_summary_mode(mode, labels):
    """Load one SUMMARY subdirectory into arrays shaped (wave, mag, separation)."""
    mode_dir = summary_dir / mode
    if not mode_dir.is_dir():
        raise FileNotFoundError(f"Missing SUMMARY directory: {mode_dir}")

    quantities = {name: [] for name in ("contrast", "stellar_flux", "core_throughput", "separation_lamD", "separation_mas")}
    for label in labels:
        for name in quantities:
            path = mode_dir / f"{name}_MAG{label}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            quantities[name].append(np.loadtxt(path, dtype=float))

    data = {name: np.stack(arrays, axis=1) for name, arrays in quantities.items()} # (wave, mag, sep)
    shape = data["stellar_flux"].shape
    for name, arr in data.items():
        if arr.shape != shape:
            raise ValueError(f"{mode}/{name} has shape {arr.shape}, expected {shape}.")
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{mode}/{name} contains non-finite values.")

    # The separation arrays should not depend on magnitude.
    for name in ("separation_lamD", "separation_mas"):
        if not np.allclose(data[name], data[name][:, :1, :], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{mode}/{name} changes with stellar magnitude.")

    return data


def validate_common_grids(reference, other, mode):
    for name in ("separation_lamD", "separation_mas"):
        if not np.allclose(reference[name], other[name], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{mode} and IMG do not use the same {name} grid.")


def validate_contrast_relation(data, mode):
    """README relation: contrast = stellar_flux / max(reference_PSF)."""
    peak = np.divide(data["stellar_flux"], data["contrast"], out=np.full_like(data["stellar_flux"], np.nan), where=data["contrast"] > 0)
    rel_spread = np.nanstd(peak, axis=-1) / np.clip(np.nanmedian(peak, axis=-1), np.finfo(float).tiny, None)
    worst = float(np.nanmax(rel_spread))
    print(f"  {mode}: max radial inconsistency in stellar_flux/contrast = {worst:.3e}")


def resample_stellar_profiles(stellar_flux, separation_native_2D, separation_out):
    """Resample raw radial intensities and return profiles normalized in [fraction/mas^2]."""
    N_wave, N_mag, _ = stellar_flux.shape
    out = np.empty((N_wave, N_mag, separation_out.size), dtype=float)
    for iw in range(N_wave):
        sep_ref = separation_native_2D[iw]
        for im in range(N_mag):
            out[iw, im] = interp_extrap_sep(separation_ref=sep_ref, separation_new=separation_out, y_ref=safe_pos(stellar_flux[iw, im]), mode="log", tail_model="powerlaw", slope=profile_outer_slope)
    return normalize_radial_profile(out, separation_out)


def resample_bounded(quantity, separation_native_2D, separation_out):
    """Resample a quantity constrained to (0,1), with a flat outer continuation."""
    N_wave, N_mag, _ = quantity.shape
    out = np.empty((N_wave, N_mag, separation_out.size), dtype=float)
    for iw in range(N_wave):
        sep_ref = separation_native_2D[iw]
        for im in range(N_mag):
            out[iw, im] = interp_extrap_sep(separation_ref=sep_ref, separation_new=separation_out, y_ref=safe01(quantity[iw, im]), mode="logit", tail_model="flat")
    return safe01(out)


def estimate_spatial_strehl(PSF_profile_density_no_coro_3D,
                            PSF_profile_density_coro_3D, star_transmission_2D,
                            wave, separation):
    """Old-PCS-style Strehl inference from spatial PSF profiles.

    All radial profiles must be normalized individually to their OWN total flux.
    Multiplying by T_on converts the coronagraphic density to IMG flux units.
    Assumption: the transmitted on-axis coronagraphic stellar signal traces the
    atmospheric speckle halo of IMG. This is approximate for a real CLC/APLC.

    Returns inferred DL density, reference IMG speckle density, raw Strehl,
    Marechal-law Strehl and a single fitted RMS WFE per stellar magnitude.
    """
    N_wave, N_mag, _ = PSF_profile_density_no_coro_3D.shape
    PSF_profile_density_speck_3D = (PSF_profile_density_coro_3D
                                    * star_transmission_2D[:, :, None])
    negative = PSF_profile_density_speck_3D > PSF_profile_density_no_coro_3D
    print(f"  Halo > IMG profile in {100*np.mean(negative):.2f}% of spatial bins")
    if np.mean(negative) > 0.05:
        print("WARNING: significant halo/IMG incompatibility; inspect normalization and Strehl calibration")
    # A negative coherent component is unphysical; clipping is explicitly diagnosed.
    PSF_profile_density_speck_3D = np.minimum(
        PSF_profile_density_speck_3D,
        (1.0 - 1e-6) * PSF_profile_density_no_coro_3D,
    )
    SR_PSF_profile_density_DL_3D = np.maximum(
        PSF_profile_density_no_coro_3D - PSF_profile_density_speck_3D, 0.0
    )
    # As in the old script, restore unit total flux for each inferred DL PSF,
    # then combine estimates across magnitudes (DL should not depend on mag).
    PSF_profile_density_DL_3D = normalize_radial_profile(
        SR_PSF_profile_density_DL_3D, separation
    )
    PSF_profile_density_DL_2D = normalize_radial_profile(
        np.nanmedian(PSF_profile_density_DL_3D, axis=1), separation
    )

    SR_raw_2D = np.full((N_wave, N_mag), np.nan)
    WFE_nm_1D = np.full(N_mag, np.nan)
    SR_2D = np.full((N_wave, N_mag), np.nan)
    r_core = 0.5 * 1.029 * wave * 1e-6 / D * rad2arcsec * 1000.0
    for iw in range(N_wave):
        mask_core = separation <= r_core[iw]
        if mask_core.sum() < 3:
            raise ValueError(f"Insufficient spatial samples for Strehl fitting at {wave[iw]:.3f} um")
        x = PSF_profile_density_DL_2D[iw, mask_core]
        weights = np.sqrt(np.clip(x / np.max(x), 0, None))
        A = np.column_stack((x, np.ones_like(x))) * weights[:, None]
        for im in range(N_mag):
            y = PSF_profile_density_no_coro_3D[iw, im, mask_core] * weights
            fit = lsq_linear(A, y, bounds=([0.0, 0.0], [1.0, np.inf]))
            if not fit.success:
                raise RuntimeError(f"Strehl fit failed for lambda={wave[iw]}, magnitude index={im}")
            SR_raw_2D[iw, im] = np.clip(fit.x[0], EPS, 1.0 - EPS)
    for im in range(N_mag):
        WFE_nm_1D[im] = np.median(
            1000.0 * wave / (2.0 * np.pi) * np.sqrt(-np.log(SR_raw_2D[:, im]))
        )
        SR_2D[:, im] = np.exp(-(
            2.0 * np.pi * WFE_nm_1D[im] / (1000.0 * wave)
        )**2)
    return (PSF_profile_density_DL_2D, PSF_profile_density_speck_3D,
            SR_raw_2D, SR_2D, WFE_nm_1D)


def get_fpm_band_mapping(wave):
    """
    Map each wavelength to one of V/R/I/Y/J/H/K and return the corresponding
    broad-band centre and nominal physical FPM radius in mas.

    FastYield's broad-band ranges are approximate and can overlap or leave small
    gaps. We therefore use lmin_bands/lmax_bands to define each band centre, then:
      - among bands containing lambda, choose the closest centre;
      - if lambda falls in a small gap/outside the tabulated ranges, choose the
        closest centre and report it.
    """
    wave = np.asarray(wave, dtype=float)
    centres = {band: 0.5 * (float(lmin_bands[band]) + float(lmax_bands[band])) for band in FPM_BANDS}
    labels, lambda0 = [], np.empty_like(wave)
    fallback = []
    for iw, w in enumerate(wave):
        candidates = [band for band in FPM_BANDS if float(lmin_bands[band]) <= w <= float(lmax_bands[band])]
        if candidates:
            band = min(candidates, key=lambda b: abs(w - centres[b]))
        else:
            band = min(FPM_BANDS, key=lambda b: abs(w - centres[b]))
            fallback.append((w, band))
        labels.append(band)
        lambda0[iw] = centres[band]

    iwa_mas = FPM_LAMD * lambda0 * 1e-6 / D * rad2arcsec * 1000.0
    if fallback:
        txt = ", ".join(f"{w:.2f}um->{band}" for w, band in fallback)
        print(f"  NOTE: wavelengths outside the approximate FastYield broad-band limits were assigned to the nearest band centre: {txt}")
    return np.asarray(labels), lambda0, iwa_mas



def build_reference_coronagraph_products(img_native, coro_native, img_core_common, coro_core_common, profile_coro_common, separation_lamD_native):
    """
    Reconstruct the old FastYield decomposition from the new SUMMARY observables.

    - PSF_profile_density_coro is normalized to the total coronagraphic stellar flux.
    - radial_transmission(0) is anchored to the measured on-axis stellar throughput.
    - at large separation, radial_transmission is anchored to the core-throughput ratio.
    - fraction_core_coro is defined so that
          fraction_core_coro * radial_transmission == SUMMARY core_throughput,
      i.e. the planet signal uses the supplied core throughput exactly at the nominal slice.
    """
    area_lamD = annulus_areas(np.nanmedian(separation_lamD_native, axis=(0, 1)))
    flux_img  = np.sum(img_native["stellar_flux"]  * area_lamD[None, None, :], axis=-1)
    flux_coro = np.sum(coro_native["stellar_flux"] * area_lamD[None, None, :], axis=-1)
    T_on      = safe01(flux_coro / np.clip(flux_img, EPS, None)) # (wave, mag)

    ratio_native = safe01(coro_native["core_throughput"] / np.clip(img_native["core_throughput"], EPS, None))
    r_lamD       = np.nanmedian(separation_lamD_native, axis=(0, 1))
    tail         = r_lamD >= 7.0
    T_off        = np.nanmedian(ratio_native[..., tail], axis=-1) if np.any(tail) else ratio_native[..., -1]
    T_off        = safe01(T_off)

    ratio_common = safe01(coro_core_common / np.clip(img_core_common, EPS, None))
    y0           = ratio_common[..., 0]
    den          = T_off - y0
    shape        = np.divide(ratio_common - y0[..., None], den[..., None], out=np.zeros_like(ratio_common), where=np.abs(den[..., None]) > EPS)
    shape        = np.clip(shape, 0.0, 1.0)
    shape        = np.maximum.accumulate(shape, axis=-1)

    radial_transmission = T_on[..., None] + (T_off - T_on)[..., None] * shape
    radial_transmission = safe01(radial_transmission)
    fraction_core_coro  = safe01(coro_core_common / radial_transmission)

    return profile_coro_common, fraction_core_coro, radial_transmission, T_on, T_off


def nominal_correct(model, reference, idx_r_wfe, idx_iwa, bounded=False):
    """Multiplicative correction at the nominal (r_WFE=1, IWA=IWA_ref) slice."""
    nominal = model[:, idx_r_wfe, idx_iwa, :]
    corr    = np.divide(reference, nominal, out=np.ones_like(reference), where=np.isfinite(nominal) & (np.abs(nominal) > 0))
    model   = model * corr[:, None, None, :]
    return safe01(model) if bounded else safe_pos(model)


def normalize_4d_profiles(profile_4d, separation):
    area = annulus_areas(separation)
    norm = np.sum(profile_4d * area[None, None, None, :], axis=-1)
    return profile_4d / np.clip(norm[..., None], np.finfo(float).tiny, None)


def make_axis_with_values(base, values):
    out = np.asarray(base, dtype=float)
    for value in np.atleast_1d(values):
        out = add_if_necessary(array=out, value=float(value))
    return np.asarray(out, dtype=float)


def output_suffix(apodizer, coronagraph):
    return f"{apodizer}_{coronagraph}_{strehl_label}_{lmin}_{lmax}_{r_WFE_min}_{r_WFE_max}_{IWA_min}_{IWA_max}_{mag_star_min}_{mag_star_max}_{sep_max}"


def write_products(apodizer, coronagraph, wave, r_WFE, IWA, mags, separation, psf, core, transmission=None):
    suffix = output_suffix(apodizer, coronagraph)
    products = {
        "wave": wave,
        "r_WFE": r_WFE,
        "IWA": IWA,
        "mag_star": mags,
        "separation": separation,
        "PSF_profile_density_5D": psf,
        "fraction_core_5D": core,
    }
    if transmission is not None:
        products["radial_transmission_5D"] = transmission
    print(f"\nSaving {apodizer}/{coronagraph} products:")
    for name, arr in products.items():
        path = output_dir / f"{instru}_{name}_{suffix}.fits"
        fits.writeto(path, np.asarray(arr, dtype=np.float32), overwrite=True)
        print(f"  {path.name:<110} {np.shape(arr)}")


def plot_profile_slice(apodizer, coronagraph, wave, separation, psf2d, core2d, transmission2d, r_WFE_value=None, WFE_nominal=None, IWA_value=None, mag_value=None, title_suffix=None):
    if not show_plots:
        return
    if title_suffix is None:
        WFE_effective = r_WFE_value * WFE_nominal
        if coronagraph is None:
            title_suffix = f"\nr_WFE = {r_WFE_value:.2f} (WFE = {WFE_effective:.1f} nm) | $V_\\star$ = {mag_value:.1f}"
        else:
            title_suffix = f"\nr_WFE = {r_WFE_value:.2f} (WFE = {WFE_effective:.1f} nm) | IWA = {IWA_value:.2f}" + r"$\lambda_{0,\mathrm{band}}/D$" + f" | $V_\\star$ = {mag_value:.1f}"
    plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl_label, pxscale=None, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=None, separation=separation, PSF_profile_density_2D=psf2d, fraction_core_2D=core2d, radial_transmission_2D=transmission2d, type_PSF="post-AO", title_suffix=title_suffix)


def diagnostic_plot_set(apodizer, coronagraph, wave, r_WFE, IWA, mags, separation, psf, core, transmission, WFE_nominal_1D, iwa_ref=IWA_ref):
    """Diagnostic profiles on the common dimensionless IWA axis lambda0_band/D."""
    if not show_plots:
        return

    im = int(np.argmin(np.abs(mags - 0.0)))
    iwfe_list = [int(np.argmin(np.abs(r_WFE - value))) for value in (r_WFE_min, 1.0, r_WFE_max)]
    ii_ref = int(np.argmin(np.abs(IWA - iwa_ref)))

    if coronagraph is None:
        for iwfe in iwfe_list:
            plot_profile_slice(apodizer=apodizer, coronagraph=coronagraph, wave=wave, separation=separation, psf2d=psf[:, iwfe, ii_ref, im], core2d=core[:, iwfe, ii_ref, im], transmission2d=None, r_WFE_value=r_WFE[iwfe], WFE_nominal=WFE_nominal_1D[im], IWA_value=IWA[ii_ref], mag_value=mags[im])
        return

    # Relative WFE sequence at the true nominal PCS mask: IWA = 2.4 lambda0_band/D.
    for iwfe in iwfe_list:
        title = f"\nr_WFE = {r_WFE[iwfe]:.2f} (WFE = {r_WFE[iwfe] * WFE_nominal_1D[im]:.1f} nm) | FPM = {IWA[ii_ref]:.1f}" + r"$\lambda_{0,\mathrm{band}}/D$" + f" | $V_\\star$ = {mags[im]:.1f}"
        plot_profile_slice(apodizer=apodizer, coronagraph=coronagraph, wave=wave, separation=separation, psf2d=psf[:, iwfe, ii_ref, im], core2d=core[:, iwfe, ii_ref, im], transmission2d=transmission[:, iwfe, ii_ref, im], title_suffix=title)

    # IWA sequence at the nominal WFE for the chosen stellar magnitude.
    # multiplicative mask size relative to lambda0_band/D at every wavelength.
    iwfe = int(np.argmin(np.abs(r_WFE - 1.0)))
    for iwa_value in [IWA_min, iwa_ref, IWA_max]:
        iiwa = int(np.argmin(np.abs(IWA - iwa_value)))
        plot_profile_slice(apodizer=apodizer, coronagraph=coronagraph, wave=wave, separation=separation, psf2d=psf[:, iwfe, iiwa, im], core2d=core[:, iwfe, iiwa, im], transmission2d=transmission[:, iwfe, iiwa, im], r_WFE_value=r_WFE[iwfe], WFE_nominal=WFE_nominal_1D[im], IWA_value=IWA[iiwa], mag_value=mags[im])

def plot_strehl_diagnostic(wave, mags, WFE_nm_1D, SR_2D, SR_raw_2D):
    """Show spatially fitted Strehl and the associated Marechal-law fit."""
    if not show_plots:
        return

    cmap_mag = plt.get_cmap("inferno_r", len(mags))
    plt.figure(figsize=(10, 6), dpi=300)
    for im, mag in enumerate(mags):
        plt.plot(wave, 100 * SR_2D[:, im],     c=cmap_mag(im), label=rf"$V_\star$ = {mag:.1f} (WFE = {WFE_nm_1D[im]:.1f} nm RMS)")
        plt.plot(wave, 100 * SR_raw_2D[:, im], c=cmap_mag(im), ls="--")
    plt.legend(loc="lower right", fontsize=12, fancybox=True, shadow=True)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()
    plt.title(f"Strehl Ratio for {instru} with NO SP-apodizer in {strehl_label}-strehl", fontsize=14)
    plt.xlabel("Wavelength [µm]", fontsize=12)
    plt.ylabel("SR [%]", fontsize=12)
    plt.axvspan(wave[0], wave[-1], color="black", alpha=0.1, lw=0, label="Simulated data range")
    plt.xlim(wave[0], wave[-1])
    plt.ylim(0, 100)
    plt.show()



# %%
# =============================================================================
# LOAD + VALIDATE SUMMARY
# =============================================================================

if not summary_dir.is_dir():
    raise FileNotFoundError(f"SUMMARY directory not found: {summary_dir}")

labels, mag_star = parse_magnitude_labels(summary_dir / "IMG")
if mag_star[0] > mag_star_min or mag_star[-1] < mag_star_max:
    raise ValueError(f"Magnitude grid {mag_star} does not cover [{mag_star_min}, {mag_star_max}].")

img = load_summary_mode("IMG", labels)
validate_contrast_relation(img, "IMG")
coros = {}
for mode in coronagraph_modes:
    coros[mode] = load_summary_mode(mode, labels)
    validate_common_grids(img, coros[mode], mode)
    validate_contrast_relation(coros[mode], mode)

N_wave_raw = img["stellar_flux"].shape[0]
wave_raw   = raw_wave_min + raw_wave_step * np.arange(N_wave_raw, dtype=float)
wave_mask  = (wave_raw >= lmin - 1e-12) & (wave_raw <= lmax + 1e-12)
wave       = wave_raw[wave_mask]
if wave.size == 0 or not np.isclose(wave[0], lmin) or not np.isclose(wave[-1], lmax):
    raise ValueError(f"SUMMARY wavelength grid does not contain the requested [{lmin}, {lmax}] um interval.")

# Actual files currently contain 401 columns from 0 to 100 lambda/D in steps of 0.25,
# despite the README text mentioning 201 columns / 44 lambda/D. Trust the data files.
sep_lamD_raw = img["separation_lamD"][:, 0]
print(f"\nSUMMARY grid actually loaded: {N_wave_raw} wavelengths x {sep_lamD_raw.shape[-1]} separations")
print(f"  wavelength = [{wave_raw[0]:.2f}, {wave_raw[-1]:.2f}] um")
print(f"  lambda/D   = [{sep_lamD_raw[0,0]:.2f}, {sep_lamD_raw[0,-1]:.2f}], median step = {np.median(np.diff(sep_lamD_raw[0])):.3f}")
print(f"  V magnitudes = {mag_star.tolist()}")

# Verify that the saved separation_mas table follows the README D=39.15 m convention.
D_summary        = 39.15
sep_mas_expected = img["separation_lamD"] * wave_raw[:, None, None] * 1e-6 / D_summary * rad2arcsec * 1000.0
sep_mas_relerr   = np.nanmax(np.abs(img["separation_mas"] - sep_mas_expected) / np.clip(np.abs(sep_mas_expected), 1.0, None))
print(f"  separation_mas consistency with D=39.15 m: max relative error = {sep_mas_relerr:.3e}")
print(f"  FastYield output angular scaling will use D={D:.2f} m to stay consistent with PCS_FastYield.py.")

# Crop the input arrays to 0.6--2.5 um.
def crop_wave(data):
    return {key: value[wave_mask] for key, value in data.items()}

img   = crop_wave(img)
coros = {mode: crop_wave(data) for mode, data in coros.items()}

# Reconstruct angular separation with the same telescope diameter used by PCS_FastYield.py.
separation_native_2D = img["separation_lamD"][:, 0, :] * wave[:, None] * 1e-6 / D * rad2arcsec * 1000.0
separation           = np.r_[0.0, np.logspace(np.log10(1.0), np.log10(sep_max), N_sep_log)]

# Physical FPM used by the supplied coronagraphic simulations.
fpm_band_wave, fpm_lambda0_wave, IWA_ref_mas_wave = get_fpm_band_mapping(wave)
print("\nPCS nominal FPM prescription:")
for band in FPM_BANDS:
    mask = fpm_band_wave == band
    if np.any(mask):
        lambda0_band = 0.5 * (float(lmin_bands[band]) + float(lmax_bands[band]))
        iwa_band     = FPM_LAMD * lambda0_band * 1e-6 / D * rad2arcsec * 1000.0
        print(f"  {band:>2s}: lambda0 = {lambda0_band:.3f} um -> IWA = {iwa_band:.3f} mas")

# Resample all directly supplied quantities onto the common FastYield separation grid.
print("\nResampling SUMMARY profiles onto the FastYield separation grid...")
profile_img  = resample_stellar_profiles(img["stellar_flux"], separation_native_2D, separation)
core_img     = resample_bounded(img["core_throughput"], separation_native_2D, separation)
profile_coro = {}
core_coro    = {}
for mode in coronagraph_modes:
    profile_coro[mode] = resample_stellar_profiles(coros[mode]["stellar_flux"], separation_native_2D, separation)
    core_coro[mode]    = resample_bounded(coros[mode]["core_throughput"],       separation_native_2D, separation)



# %%
# =============================================================================
# AO / WFE CALIBRATION FROM IMG
# =============================================================================

print(f"\nEstimating DL profile / spatial Strehl from IMG and {STREHL_REFERENCE_MODE}...")
PSF_profile_density_no_coro_3D = profile_img
PSF_profile_density_coro_ref_3D = profile_coro[STREHL_REFERENCE_MODE]

# The returned on-axis transmission is the integral of the RAW SUMMARY
# stellar_flux densities, not the integral of the normalized output profiles.
_, _, _, star_transmission_2D, _ = build_reference_coronagraph_products(
    img_native=img, coro_native=coros[STREHL_REFERENCE_MODE],
    img_core_common=core_img, coro_core_common=core_coro[STREHL_REFERENCE_MODE],
    profile_coro_common=PSF_profile_density_coro_ref_3D,
    separation_lamD_native=img["separation_lamD"],
)

(PSF_profile_density_DL_2D, PSF_profile_density_speck_3D,
 SR_raw_2D, SR_2D, WFE_nm_1D) = estimate_spatial_strehl(
    PSF_profile_density_no_coro_3D=PSF_profile_density_no_coro_3D,
    PSF_profile_density_coro_3D=PSF_profile_density_coro_ref_3D,
    star_transmission_2D=star_transmission_2D,
    wave=wave, separation=separation,
)
print("\nSpatial Strehl / Marechal WFE estimates (not independently measured):")
for mag, wfe in zip(mag_star, WFE_nm_1D):
    print(f"  V = {mag:>4.1f}  ->  WFE = {wfe:7.3f} nm RMS")
if np.any(np.diff(WFE_nm_1D) < -1e-8):
    print("WARNING: estimated WFE is not monotonic with magnitude")
if not np.all(np.isfinite(WFE_nm_1D)) or np.any(WFE_nm_1D <= 0):
    raise ValueError("Nominal WFE estimates must be finite and strictly positive")
if not (r_WFE_min <= 1.0 <= r_WFE_max):
    raise ValueError("The relative WFE grid must contain r_WFE = 1")

# Approximation retained from old PCS: IMG core_throughput ~ SR x DL core fraction.
# Taking the maximum over magnitude avoids the DL cap clipping nominal slices.
fraction_core_DL_2D = safe01(np.nanmax(
    core_img / np.clip(SR_2D[:, :, None], EPS, None), axis=1
))
plot_strehl_diagnostic(wave=wave, mags=mag_star, WFE_nm_1D=WFE_nm_1D,
                       SR_2D=SR_2D, SR_raw_2D=SR_raw_2D)
            
r_WFE_base = np.linspace(r_WFE_min, r_WFE_max, N_r_WFE_base)
r_WFE      = make_axis_with_values(r_WFE_base, [1.0])
iw_ref     = int(np.argmin(np.abs(r_WFE - 1.0)))
assert np.isclose(r_WFE[iw_ref], 1.0), "Nominal r_WFE=1 must exist exactly"
print(f"\nCommon relative WFE grid: {len(r_WFE)} points, [{r_WFE.min():.2f}, {r_WFE.max():.2f}], nominal index={iw_ref}")

# Physical calibration of the relative axis for each reference magnitude.
# This information is not an additional dimension of the 5D products.
fits.writeto(output_dir / f"{instru}_WFE_nominal_mag_1D.fits",
             np.asarray(WFE_nm_1D, dtype=np.float32), overwrite=True)

# Common dimensionless coronagraph axis. IWA_ref = 2.4 is the actual SUMMARY mask;
# at each wavelength it corresponds physically to 2.4*lambda0_band/D.
IWA_base = np.logspace(np.log10(IWA_min), np.log10(IWA_max), N_IWA_base)
IWA      = make_axis_with_values(IWA_base, IWA_ref)

N_wave, N_mag, N_sep = profile_img.shape



# %%
# =============================================================================
# NON-CORONAGRAPHIC IMG OUTPUT
# =============================================================================

print("\nBuilding non-coronagraphic PSF FastYield grid...")
shape_nc = (N_wave, len(r_WFE), len(IWA), N_mag, N_sep)
psf_nc_5D  = np.empty(shape_nc, dtype=np.float32)
core_nc_5D = np.empty(shape_nc, dtype=np.float32)

for im in tqdm(range(N_mag), desc="non-coronagraphic PSF: r_WFE grid over magnitudes"):
    # Same halo used to construct the DL PSF and estimate the spatial Strehl.
    speck_ref = PSF_profile_density_speck_3D[:, im]

    r_IWA = IWA / IWA_ref # irrelevant without coronagraph; only preserves the 5D interface
    psf4, core4, _, _, _ = build_PSF_grid(SR=SR_2D[:, im], D=D, wave=wave, r_WFE=r_WFE, r_IWA=r_IWA, separation=separation, PSF_profile_density_no_coro_2D=profile_img[:, im], fraction_core_no_coro_2D=core_img[:, im], PSF_profile_density_DL_2D=None, fraction_core_DL_2D=fraction_core_DL_2D, coronagraph=None, IWA_ref=None, PSF_profile_density_coro_2D=None, fraction_core_coro_2D=None, radial_transmission_2D=None, PSF_profile_density_speck_2D=speck_ref)

    ii_ref = int(np.argmin(np.abs(IWA - IWA_ref)))
    psf4   = nominal_correct(psf4,  profile_img[:, im], iw_ref, ii_ref, bounded=False)
    core4  = nominal_correct(core4, core_img[:, im],    iw_ref, ii_ref, bounded=True)
    psf4   = normalize_4d_profiles(psf4, separation)

    psf_nc_5D[:, :, :, im]  = psf4.astype(np.float32)
    core_nc_5D[:, :, :, im] = core4.astype(np.float32)

write_products(apodizer="NO_SP", coronagraph=None, wave=wave, r_WFE=r_WFE, IWA=IWA, mags=mag_star, separation=separation, psf=psf_nc_5D, core=core_nc_5D)
diagnostic_plot_set(apodizer="NO_SP", coronagraph=None, wave=wave, r_WFE=r_WFE, IWA=IWA, mags=mag_star, separation=separation, psf=psf_nc_5D, core=core_nc_5D, transmission=None, WFE_nominal_1D=WFE_nm_1D)
del psf_nc_5D, core_nc_5D



# %%
# =============================================================================
# CORONAGRAPHIC OUTPUTS: CLC + APLC
# =============================================================================

for mode in coronagraph_modes:
    cfg = mode_to_fastyield[mode]

    print(f"\n{mode}: using IWA_ref = {IWA_ref:.1f}*lambda0_band/D.")
    print(f"  Dimensionless IWA grid = [{IWA.min():.3f}, {IWA.max():.3f}] lambda0_band/D")

    p_coro_ref, fc_coro_ref, rt_ref, T_on, T_off = build_reference_coronagraph_products(img_native=img, coro_native=coros[mode], img_core_common=core_img, coro_core_common=core_coro[mode], profile_coro_common=profile_coro[mode], separation_lamD_native=img["separation_lamD"])

    # Direct reconstructed SUMMARY reference. Every wavelength uses the mask of its
    # broad band, i.e. IWA_ref_mas_wave = 2.4*lambda0_band/D in physical mas.
    im0 = int(np.argmin(np.abs(mag_star - 0.0)))
    plot_profile_slice(apodizer=cfg["apodizer"], coronagraph=cfg["coronagraph"], wave=wave, separation=separation, psf2d=p_coro_ref[:, im0], core2d=fc_coro_ref[:, im0], transmission2d=rt_ref[:, im0], title_suffix=f"\nSUMMARY reference | FPM = {IWA_ref:.1f}" + r"$\lambda_{0,\mathrm{band}}/D$" + f" | $V_\\star$ = {mag_star[im0]:.1f}")

    rel_core_err = np.nanmax(np.abs(fc_coro_ref * rt_ref - core_coro[mode]) / np.clip(core_coro[mode], 1e-12, None))
    print(f"  nominal core-throughput decomposition max relative error = {rel_core_err:.3e}")
    print(f"  on-axis stellar transmission range = [{np.nanmin(T_on):.3e}, {np.nanmax(T_on):.3e}]")
    print(f"  off-axis transmission range        = [{np.nanmin(T_off):.3f}, {np.nanmax(T_off):.3f}]")

    shape_coro   = (N_wave, len(r_WFE), len(IWA), N_mag, N_sep)
    psf_coro_5D  = np.empty(shape_coro, dtype=np.float32)
    core_coro_5D = np.empty(shape_coro, dtype=np.float32)
    rt_coro_5D   = np.empty(shape_coro, dtype=np.float32)
    ii_ref       = int(np.argmin(np.abs(IWA - IWA_ref)))

    for im in tqdm(range(N_mag), desc=f"{mode}: r_WFE x IWA grid over V magnitudes"):
        
        # Argument refers to the IMG halo; the coronagraphic shape itself
        # is treated as speckle-dominated inside build_PSF_grid.
        speck_ref = PSF_profile_density_speck_3D[:, im]

        r_IWA = IWA / IWA_ref
        _, _, psf4, core4, rt4 = build_PSF_grid(SR=SR_2D[:, im], D=D, wave=wave, r_WFE=r_WFE, r_IWA=r_IWA, separation=separation, PSF_profile_density_no_coro_2D=profile_img[:, im], fraction_core_no_coro_2D=core_img[:, im], PSF_profile_density_DL_2D=None, fraction_core_DL_2D=fraction_core_DL_2D, coronagraph="LYOT", IWA_ref=IWA_ref_mas_wave, PSF_profile_density_coro_2D=p_coro_ref[:, im], fraction_core_coro_2D=fc_coro_ref[:, im], radial_transmission_2D=rt_ref[:, im], PSF_profile_density_speck_2D=speck_ref)

        # The nominal SUMMARY reference is now one single IWA plane (IWA=2.4) for
        # every wavelength, so the ordinary nominal correction is sufficient.
        psf4   = nominal_correct(psf4,  p_coro_ref[:, im],  iw_ref, ii_ref, bounded=False)
        core4  = nominal_correct(core4, fc_coro_ref[:, im], iw_ref, ii_ref, bounded=True)
        rt4    = nominal_correct(rt4,   rt_ref[:, im],      iw_ref, ii_ref, bounded=True)
        #psf4   = normalize_4d_profiles(psf4, separation) # the I_speck propto WFE^2 is not taken into account in T_coro(rho), so we should not renormalize

        psf_coro_5D[:, :, :, im]  = psf4.astype(np.float32)
        core_coro_5D[:, :, :, im] = core4.astype(np.float32)
        rt_coro_5D[:, :, :, im]   = rt4.astype(np.float32)

    # The nominal reference is now the same IWA index at every wavelength.
    errs_psf, errs_core = [], []
    for im in range(N_mag):
        p_nom  = psf_coro_5D[:, iw_ref, ii_ref, im]
        fc_nom = core_coro_5D[:, iw_ref, ii_ref, im]
        rt_nom = rt_coro_5D[:, iw_ref, ii_ref, im]
        c_nom  = fc_nom * rt_nom
        errs_psf.append(np.nanmax(np.abs(p_nom - p_coro_ref[:, im]) / np.clip(p_coro_ref[:, im], 1e-20, None)))
        errs_core.append(np.nanmax(np.abs(c_nom - core_coro[mode][:, im]) / np.clip(core_coro[mode][:, im], 1e-12, None)))
    print(f"  final nominal-plane PSF-profile max relative error = {np.nanmax(errs_psf):.3e}")
    print(f"  final nominal-plane core-throughput max relative error = {np.nanmax(errs_core):.3e}")

    write_products(apodizer=cfg["apodizer"], coronagraph=cfg["coronagraph"], wave=wave, r_WFE=r_WFE, IWA=IWA, mags=mag_star, separation=separation, psf=psf_coro_5D, core=core_coro_5D, transmission=rt_coro_5D)
    diagnostic_plot_set(apodizer=cfg["apodizer"], coronagraph=cfg["coronagraph"], wave=wave, r_WFE=r_WFE, IWA=IWA, mags=mag_star, separation=separation, psf=psf_coro_5D, core=core_coro_5D, transmission=rt_coro_5D, WFE_nominal_1D=WFE_nm_1D)
    del psf_coro_5D, core_coro_5D, rt_coro_5D

print("\nDone.")
print(f"FastYield PSF inputs written to: {output_dir}")
