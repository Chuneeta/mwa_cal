import numpy  as np
from astropy.io import fits
from mwa_qa.read_metafits import Metafits
from mwa_qa.read_uvfits import UVfits
from astropy.constants import c
from scipy.ndimage import uniform_filter
import copy


calpols = {'xx': 0,
        'yy': 3,
        'xy': 1,
        'yx': 2
        } 

calpol_inds = [0, 3, 1, 2]

uvpols = {'xx': 0,
        'yy': 1,
        'xy': 2,
        'yx': 3
        } 

calpols = [0, 3, 1, 2]

def create_calibration_fits(metafits, solutions_data=None, output_name=None, overwrite=None):
    """
    Creates fits file compatible with Hyperdrive calibration fits file
    Parameters
    ----------
    metafits: Metafits file containing metadata of the observation
    """

    #Reading metafits containing tile information
    m = Metafits(metafits)
 
    # Creating fits file
    # Primary HDU (no data)
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['COMMENT'] = "Primary HDU"

    # SOLUTIONS ImageHDU: (8, 768, 128, 1), float64
    if solutions_data is None:
        solutions_data = np.zeros((8, 768, 128, 1), dtype=np.float64)
    solutions_hdu = fits.ImageHDU(data=solutions_data, name='SOLUTIONS')

    # TIMEBLOCKS BinTableHDU: 1 row x 3 columns, [1D, 1D, 1D]
    cols_timeblocks = fits.ColDefs([
        fits.Column(name='Start', format='1D', array=[m.start_gpstime]),
        fits.Column(name='End', format='1D', array=[m.start_gpstime + m.exposure]),
        fits.Column(name='Average', format='1D', array=[0.5 * (m.start_gpstime + m.exposure)])
    ])
    timeblocks_hdu = fits.BinTableHDU.from_columns(cols_timeblocks, name='TIMEBLOCKS')

    # TILES BinTableHDU: 128 rows x 5 columns, [1J, 1I, 8A, 32D, 16J]
    cols_tiles = fits.ColDefs([
        fits.Column(name='Antenna', format='1J', array=m.antenna_numbers),
        fits.Column(name='Flag', format='1I', array=m.flag_array),
        fits.Column(name='TileName', format='8A', array=m.antenna_names),
        fits.Column(name='DipoleGains', format='32D', array=m.delays),
        fits.Column(name='DipoleDelays', format='16J', array=m.dipole_gains),
    ])
    tiles_hdu = fits.BinTableHDU.from_columns(cols_tiles, name='TILES')

    # CHANBLOCKS BinTableHDU: 768 rows x 3 columns, [1J, 1B, 1D]
    frequency_array = m.frequency_array
    cols_chanblocks = fits.ColDefs([
        fits.Column(name='Index', format='1J', array=np.arange(len(frequency_array))),
        fits.Column(name='Flag', format='1B', array=np.zeros(len(frequency_array))),
        fits.Column(name='Freq', format='1D', array=frequency_array),
    ])
    chanblocks_hdu = fits.BinTableHDU.from_columns(cols_chanblocks, name='CHANBLOCKS')

    # RESULTS ImageHDU: (768, 1), float64
    results_data = np.zeros((len(frequency_array), 1), dtype=np.float64)
    results_hdu = fits.ImageHDU(data=results_data, name='RESULTS')

    # BASELINES ImageHDU: (8128,), float64
    nbls = len(m.baseline_lengths) - len(m.antenna_names)
    baselines_data = np.zeros(nbls, dtype=np.float64)
    baselines_hdu = fits.ImageHDU(data=baselines_data, name='BASELINES')

    # Create the HDUList and write to file
    hdul = fits.HDUList([
        primary_hdu,
        solutions_hdu,
        timeblocks_hdu,
        tiles_hdu,
        chanblocks_hdu,
        results_hdu,
        baselines_hdu
    ])

    if output_name is None:
        output_name = 'calibration_sample.fits'

    hdul.writeto(output_name, overwrite=overwrite)

def rebin_columns(a, ax, shape, col_sizer):
    """
    Performs expansion on the columns of a 1D or 2D array using interpolation to fill in the values that are created
    by expanding in the space between existing values. This function assumes the rows have already been expanded
    to the required number.
    """
    # tile the range of col_sizer
    tiles = np.tile(np.arange(col_sizer), (shape[0], shape[1] // col_sizer - 1))
    # Get the differences between the columns
    differences = np.diff(a, axis=ax) / col_sizer
    # Multiply this by the tiles
    inferences_non_pad = np.repeat(differences, col_sizer, axis=ax) * tiles
    # Pad the zeros for the last two rows, and remove the extra zeros to make inferences same shape as desired shape
    inferences = np.pad(inferences_non_pad, (0, col_sizer))[:-col_sizer]
    if np.issubdtype(a.dtype, np.integer):
        inferences = np.floor(inferences).astype("int")
    # Now get our final array by adding the repeat of our rows rebinned to the inferences
    rebinned = inferences + np.repeat(a, col_sizer, axis=ax)
    return rebinned

def rebin(a, shape, sample=False):
    """
    Resize a 1D or 2D array to a new shape using sampling (repeat or slice)
    """
    old_shape = a.shape
    # Prevent more processing than needed if we want the same shape
    if old_shape == shape:
        return a
    if len(old_shape) == 1:
        old_shape = (1, old_shape[0])
    # Sample the original array using rebin
    if sample:
        # If its a 1D array then...
        if old_shape[0] == 1:
            if shape[1] > old_shape[1]:
                rebinned = np.repeat(a, shape[1] // old_shape[1], axis=0)
            else:
                rebinned = a[:: old_shape[1] // shape[1]]
        # Assume its a 2D array
        else:
            print ('Else')
            # Do the Rows first
            if shape[0] >= old_shape[0]:
                # Expand Rows
                rebinned = np.repeat(a, shape[0] // old_shape[0], axis=0)
            else:
                # Compress Rows
                rebinned = a[:: old_shape[0] // shape[0], :]
            # Then do the columns
            if shape[1] >= old_shape[1]:
                # Expand Columns
                rebinned = np.repeat(rebinned, shape[1] // old_shape[1], axis=1)
            else:
                # Compress columns
                rebinned = rebinned[:, :: old_shape[1] // shape[1]]
        # Return the rebinned without adjusting dtype as none of the functions above change it
        return rebinned
     # If we are downsizing
    elif shape[0] < old_shape[0] or shape[1] < old_shape[1]:
        if (max(old_shape[0], shape[0]) % min(old_shape[0], shape[0]) != 0) or (
            max(old_shape[1], shape[1]) % min(old_shape[1], shape[1]) != 0
        ):
            raise ValueError("Your new shape should be a factor of the original shape")
        # If we are increasing the rows or columns and reducing the other, increase them now and change the old shape
        if shape[0] > old_shape[0]:
            a = np.tile(a, (shape[0], 1))
            old_shape = a.shape
        elif shape[1] > old_shape[1]:
            a = np.tile(a, (1, shape[1]))
            old_shape = a.shape
        # Create the shape we need (rows, rows that can fit in old_shape, cols, cols that can fit into old_shape)
        sh = shape[0], old_shape[0] // shape[0], shape[1], old_shape[1] // shape[1]
        # Create the 4D array
        rebinned = np.reshape(a, sh)
        # Get the average of the columns first
        rebinned = rebinned.mean(-1)
        # To ensure that we get the result same as IDL
        # it seems to fix the values after every calculation if integer
        if np.issubdtype(a.dtype, np.integer):
            rebinned = np.fix(rebinned).astype("int")
        # Now get it for the rows
        rebinned = rebinned.mean(1)
        # If we are expecting 1D array ensure it gets returned as a 1D array
        if shape[0] == 1:
            rebinned = rebinned[0]
        # To ensure that we get the result same as IDL
        # it seems to fix the values after every calculation if integer
        if np.issubdtype(a.dtype, np.integer):
            rebinned = np.fix(rebinned).astype("int")

    # Otherwise we are expanding
    else:
        if shape[0] % old_shape[0] != 0 or shape[1] % old_shape[1] != 0:
            raise ValueError(
                "Your new shape should be a multiple of the original shape"
            )
        # Get the size changes of the row and column separately
        row_sizer = shape[0] // old_shape[0]
        col_sizer = shape[1] // old_shape[1]
        ax = 0
        # If 1D array then do along the columns
        if old_shape[0] == 1:
            rebinned = rebin_columns(a, ax, shape, col_sizer)
            if shape[0] == 1:
                rebinned = rebinned[0]
        # Else its a 2D array
        else:
            row_rebinned = rebin_rows(a, ax, shape, old_shape, row_sizer)
            # If it matches the new shape, then return it
            if row_rebinned.shape == shape:
                return row_rebinned
            else:
                ax = 1
                rebinned = rebin_columns(row_rebinned, ax, shape, col_sizer)
    return rebinned
    
def get_data(uvfits_file):
    """
    Extract autocorrelations from the uvfits files
    """
    uvf = UVfits(uvfits_file)
    data_array = uvf.data_array
    # get autocorrelations
    auto_antpairs = uvf.auto_antpairs() 
    auto_array = uvf.data_for_antpairs(auto_antpairs)
    return data_array, auto_array

def evaluate_auto_init(data_uvfits,  model_uvfits):
    """
    Initialize calibration solutions using autocorrelations.
    """
    # getting autos from data
    data_all, data_autos = get_data(data_uvfits)
    # average data autos across time
    data_autos = np.nanmean(data_autos, axis=0)
    n_ant, n_freq, n_pol = data_autos.shape
    model_all, model_autos = get_data(model_uvfits)
    # average model autos across time
    model_autos = np.nanmean(model_autos, axis=0)
    auto_gain = np.ones((n_ant, n_freq, n_pol), dtype=np.complex28)

    for pol in range(n_pol):
        # Cross-baseline scaling
        data_mean = resistant_mean(np.abs(data_all[:, :, pol]), deviations=2)
        model_mean = resistant_mean(np.abs(model_all[:, :, pol]), deviations=2)
        auto_scale[pol] = np.sqrt(data_mean / model_mean)

        gain = np.sqrt(data_autos[:, :, pol] * weight_invert(model_autos[:, :, pol]))
        gain *= auto_scale[pol_i] * weight_invert(np.mean(gain))

        gain[np.isnan(gain)] = 1
        gain[gain <= 0] = 1

        auto_gain[pol_i, :, :][:, auto_tile_i] = gain[:, auto_tile_i]
        
    return auto_gain


def resistant_mean(data, deviations=3):
    """
    The resistant_mean function calculates a robust average by filtering 
    out outliers using median absolute deviation and an adjusted standard 
    deviation threshold.
    """
    # values for pyFHD
    mad_scale: float = 0.67449999999999999,
    sigma_coeff: NDArray[np.float64] = np.array(
        [
            0.020142000000000000,
            -0.23583999999999999,
            0.90722999999999998,
            -0.15404999999999999,
        ]
    )

    median = np.nanmedian(data.real)
    abs_dev = np.abs(data - median)
    mad = np.nanmedian(abs_dev) / mad_scale
    threshold = deviations * mad
    filtered = data[abs_dev <= threshold]
    # If the deviations is less than 4.5, change the sigma (standard deviation of the subarray) by using a polyval with set sigma coefficient
    # This compensates Sigma for truncation
    # Calculate the standard deviation of the rela and imag separately
    sigma = np.std(filtered.real) + 1j * np.std(filtered.imag)
    if deviations <= 4.5:
        sigma /= np.polyval(sigma_coeff, deviations)
    filtered = data[abs_dev <= np.abs(deviations * sigma)]
    return np.mean(filtered)

def weight_invert(weights, threshold = None, use_abs = None):
    """"
    The function processes input weights by excluding 
    values that are zero, NaN, or infinite, ensuring they're suitable 
    for further calculations. If a threshold is provided, 
    it replaces the default zero-check and is used 
    to filter out values below that threshold instead.
    """
    weights = np.asarray(weights)
    result = np.zeros_like(weights, dtype=np.result_type(weights, np.float64))
    weights_eval = np.abs(weights) if use_abs or np.iscomplexobj(weights) else weights
    if threshold is not None:
        valid = weights_eval >= threshold
    else:
        valid = weights_eval != 0
    result[valid] = 1.0 / weights[valid]
    result[np.isnan(result) | np.isinf(result)] = 0
    return result

def cal_auto_ratio_divide(gains, vis_auto, ref_antenna):
    """
    Removes antenna-dependent effects from gain calibration using autocorrelations.
    """
    vis_auto_avg = np.nanmean(vis_auto, axis=0)
    gain_avg = np.nanmean(gains, axis=0)
    n_tile, n_freq, n_pol = vis_auto_avg.shape
    auto_ratio = np.empty((n_tile, n_freq, n_pol))
    auto_gains = np.empty((n_tile, n_freq, n_pol), dtype=complex)
    for pol_i in range(n_pol):
        v0 = vis_auto_avg[ref_antenna, :, pol_i]
        norm = weight_invert(v0)
        auto_ratio[: , :, calpols[pol_i]] = np.sqrt(
            vis_auto_avg[:, :, pol_i] * norm
        )
        auto_gains[: ,:, calpols[pol_i]] = gain_avg[:, :, calpols[pol_i]] * weight_invert(auto_ratio[:, :, calpols[pol_i]])
    return auto_gains, auto_ratio

def cal_auto_ratio_remultiply(gains, auto_ratio):
    """
    Reapply antenna-dependent parameters to gain values after averaging steps.
    This operation remultiplies the gain with the normalized square roots of
    autocorrelation visibilities.

    """
    gains *= np.abs(auto_ratio)
    return gains

def vis_cal_bandpass(gains, metafits, freq_use, cable_fit=None):
    """
    Calibrate per-frequency amplitudes by averaging over tiles (global or by cable group).
    """
    n_tile, n_freq, n_pol = gains.shape
    gain2 = np.zeros((n_tile, n_freq, n_pol), dtype=np.complex128)
    
    cal_bandpass_gain = np.empty((n_tile, n_freq, n_pol), dtype=np.complex128)
    cal_remainder_gain = np.empty((n_tile, n_freq, n_pol), dtype=np.complex128)

    meta = Metafits(metafits)
    antenna_numbers = meta.antenna_numbers
    flag_tiles = np.array(meta.flag_array)
    tile_use = np.arange(len(antenna_numbers))
    tile_use = tile_use[np.where(flag_tiles !=1)]

    if cable_fit:
        cable_length = np.array(meta.cable_lengths)
        unique_cable_length = np.unique(cable_length)
        tile_use_arr = {}
        bandpass_arr = np.zeros(
            (cable_length.size, n_freq, n_pol))
        bandpass_col_count = 1

        for cable_ind, cable_i in enumerate(unique_cable_length):
            tile_inds = np.where(cable_length == cable_i)
            tile_use_arr[cable_i] = tile_inds
            tile_use_cable = tile_use_arr.get(cable_i)
            if tile_use_cable is None or tile_use_cable[0].size == 0:
                tile_use_cable = np.arange(n_tile)
            else:
                tile_use_cable = tile_use_cable[0]  # unwrap from tuple
            for pol_i in range(n_pol):
                temp_gain = gains[:, :, pol_i]
                if cable_i == 0 and pol_i == 0:
                    gain2 = np.zeros((n_tile, n_freq, n_pol), dtype=np.complex128)
                gain_use = temp_gain[tile_use, :][:, freq_use]
                amp = np.abs(gain_use)
                # amp2 is a temporary variable used in place of the amp array for an added layer of safety
                amp2 = np.zeros((tile_use_cable.size, freq_use.size))
                # This is the normalization loop for each tile. If the mean of gain amplitudes over all frequencies is nonzero, then divide
                # the gain amplitudes by that number, otherwise make the gain amplitudes zero.
                for tile_i in range(tile_use_cable.size):
                    res_mean = resistant_mean(amp[:, tile_i], 2)
                    if res_mean != 0:
                        amp2[tile_i, :] = amp[tile_i, :] / res_mean
                    else:
                        amp2[tile_i, :] = 0
                # This finds the normalized gain amplitude mean per frequency over all tiles, which is the final bandpass per cable group.
                bandpass_single = np.empty(freq_use.size)
                # If this is slow, resistant_mean can be vectorized
                for f_i in range(freq_use.size):
                    bandpass_single[f_i] = resistant_mean(amp2[:, f_i], 2)
                # Want iterative to start at 1 (to not overwrite freq) and store final bandpass per cable group.
                bandpass_arr[bandpass_col_count - 1, freq_use, pol_i] = bandpass_single
               #print (bandpass_arr[bandpass_col_count - 1, freq_use, pol_i])
                bandpass_col_count += 1
                
                # Fill temporary variable gain2, set equal to final bandpass per cable group for each tile that will use that bandpass.
                for tile_i in range(tile_use_cable.size):
                    gain2[tile_use_cable[tile_i], freq_use, pol_i] = bandpass_single
                # For the last bit at the end of the cable
                if cable_ind == unique_cable_length.size - 1:
                    # Set gain3 to the input gains
                    gain3 = gains[:, :, pol_i].copy()
                    # Set what will be passed back as the output gain as the final bandpass per cable type.
                    gain2_input = gain2[:, :, pol_i]
                    cal_bandpass_gain[:, :, pol_i] = gain2_input
                    # Set what will be passed back as the residual as the input gain divided by the final bandpass per cable type.
                    gain3[:, freq_use] /= gain2_input[:, freq_use]
                    cal_remainder_gain[:, :, pol_i] = gain3

    else:
        bandpass_arr = np.zeros((n_freq, n_pol + 1))
        for pol_i in range(n_pol):
            gain = gains[:, :, pol_i]
            gain_use = gain[tile_use, :][:, freq_use]
            amp = np.abs(gain[tile_use, :][:, freq_use])

            amp2 = np.zeros((tile_use.size, freq_use.size))
            for tile_i in range(tile_use.size):
                res_mean = resistant_mean(amp[tile_i, :], 2)
                if res_mean != 0:
                    amp2[tile_i, :] = amp[tile_i, :] / res_mean
                else:
                    amp2[tile_i, :] = 0

            bandpass_single = np.empty(freq_use.size)
            # If this is slow, resistant_mean can be vectorized
            for f_i in range(freq_use.size):
                bandpass_single[f_i] = resistant_mean(amp2[:, f_i], 2)
            bandpass_arr[freq_use, pol_i + 1] = bandpass_single

            gain2 = np.zeros_like(gain, dtype=np.complex128)
            gain3 = gain.copy()

            # Apply bandpass to gains
            for tile_i in range(n_tile):
                gain2[tile_i, freq_use] = bandpass_single
                gain3[tile_i, freq_use] /= bandpass_single

            cal_bandpass_gain[:, :, pol_i] = gain2
            cal_remainder_gain[:, :, pol_i] = gain3

    return cal_bandpass_gain, cal_remainder_gain


def vis_cal_polyfit(gains,
                    freq_use,
                    metafits,
                    auto_ratio,
                    poly_amplitude_order=2, 
                    poly_phase_order=1,
                    reflection_fit=None,
                    cal_reflection_mode_theory=None,
                    cal_reflection_mode_delay=None,
                    cal_reflection_hyperresolve=None
):
    og_gain_arr = np.copy(gains)
    n_tile, n_freq, n_pol = gains.shape
    gain_residual = np.empty((n_tile, n_freq, n_pol), dtype=complex)
    gains_cfit = np.empty((n_tile, n_freq, n_pol), dtype=complex)
    meta = Metafits(metafits)
    antenna_numbers = meta.antenna_numbers
    flag_tiles = np.array(meta.flag_array)
    tile_use = np.arange(len(antenna_numbers))
    tile_use = tile_use[np.where(flag_tiles !=1)]

    for pol_i in range(n_pol):
        gain_arr = np.copy(gains[:, :, pol_i])
        gain_amp = np.abs(gain_arr)
        gain_phase = np.arctan2(gain_arr.imag, gain_arr.real)
        for tile_i in range(n_tile):
            gain = np.squeeze(gain_amp[tile_i, freq_use])
            gain_fit = np.zeros(n_freq)
            # fit for amplitude
            fit_params = (
                    np.polynomial.Polynomial.fit(
                        freq_use, gain, deg=poly_amplitude_order
                    )
                    .convert()
                    .coef
                )
            
            for di in range(poly_amplitude_order):
                gain_fit += fit_params[di] * np.arange(n_freq) ** di

            gain_residual[tile_i, :, pol_i] = gain_amp[tile_i, :] - gain_fit

            # Fit for phase
            phase_use = np.unwrap(np.squeeze(gain_phase[tile_i, freq_use]))
            phase_params = (
            np.polynomial.Polynomial.fit(
                freq_use, phase_use, poly_phase_order
            )
            .convert()
                .coef
        )
            
            phase_fit = np.zeros(n_freq)
            for di in range(phase_params.size):
                phase_fit += phase_params[di] * np.arange(n_freq) ** di
            gain_arr[tile_i, :] = gain_fit * np.exp(1j * phase_fit)
        gains_cfit[:, :, pol_i] = gain_arr

    if reflection_fit:
        if cal_reflection_mode_theory:
                print ("Using theory calculation in nominal reflection mode calibration.")
                cable_lengths = np.array(meta.cable_lengths).astype(float)
                cable_vf = 0.81
                tile_ref_flag = np.minimum(np.maximum(0, np.ones(len(cable_lengths))), 1)
                # Nominal Reflect Time
                reflect_time = (2 * cable_lengths) / (c.value * cable_vf)
                bandwidth = (
                        (
                            np.max(meta.frequency_array * 1e6)
                            - np.min(meta.frequency_array * 1e6)
                        )
                        * n_freq
                    ) / (n_freq - 1)
                    # Modes in fourier transform units
                mode_i_arr = np.tile(
                        bandwidth * reflect_time * tile_ref_flag, [n_pol, 1]
                    )
                mode_i_arr = mode_i_arr.T

        elif cal_reflection_mode_delay:
            print ("Using calibration delay spectrum to calculate nominal reflection modes")
            spec_mask = np.zeros(n_freq)
            spec_mask[freq_use] = 1
            freq_cut = np.where(spec_mask == 0)
            spec_psf = np.abs(np.fft.fftn(spec_mask, norm="forward"))
            spec_inds = np.arange(n_freq // 2)
            spec_psf = spec_psf[spec_inds]
            mode_test = np.zeros(n_freq // 2)
            for pol_i in range(n_pol):
                for ti in range(tile_use.size):
                    tile_i = tile_use[ti]
                    spec0 = np.abs(np.fft.fftn(avg_fitted_gains[tile_i, :, pol_i]))
                    mode_test += spec0[spec_inds]
                psf_mask = np.zeros(n_freq // 2)

            if freq_cut[0].size > 0:
                psf_mask[np.where(spec_psf > (np.max(spec_psf) / 1000))] = 1
                # Replaces IDL smooth with edge_truncate
                psf_mask = uniform_filter(psf_mask, size=5, mode="nearest")
                mask_i = np.nonzero(psf_mask)
                if mask_i[0].size > 0:
                    mode_test[mask_i] = 0
            mode_i_arr = np.zeros((n_freq, n_pol)) + np.argmax(mode_test)
            
        for pol_i in range(n_pol):
            # Divide the polyfit to reveal the residual cable reflections better
            gain_arr_dv = og_gain_arr[:, :, pol_i] / gains_cfit[:, :, pol_i] 
            for ti in range(tile_use.size):
                tile_i = tile_use[ti]
                mode_i = mode_i_arr[tile_i, pol_i]
                if mode_i == 0:
                    continue
                else:
                    # Options to hyperresolve or fit the reflection modes/amp/phase given the nominal calculations
                    if cal_reflection_hyperresolve:
                        # start with nominal cable length
                        mode0 = mode_i
                        # overresolve the FT used for the fit (normal resolution would be dmode=1)
                        dmode = 0.05
                        # range around the central mode to test
                        nmodes = 101
                        # array of modes to try
                        modes = (np.arange(nmodes) - nmodes // 2) * dmode + mode0
                        # reshape for ease of computing
                        modes = rebin(modes, (freq_use.size, nmodes)).T

                        if auto_ratio is not None:
                            # Find tiles which will *not* be accidently coherent in their cable reflection in order to reduce bias
                            #inds = np.where(
                            #    (tile_use)
                            #    & (mode_i_arr[:, pol_i] > 0)
                            #    & ((np.abs(mode_i_arr[:, pol_i] - mode_i)) > 0.01)
                            #)
                            inds = np.where(
                                (mode_i_arr[:, pol_i] > 0)
                                & ((np.abs(mode_i_arr[:, pol_i] - mode_i)) > 0.01)
                            )
                            # mean over frequency for each tile
                            freq_mean = np.nanmean(auto_ratio[:, :, pol_i], axis=0)
                            norm_autos = auto_ratio[:, :, pol_i] / rebin(
                                freq_mean, (n_tile, n_freq)
                            )
                            # mean over all tiles which *are not* accidently coherent as a func of freq
                            incoherent_mean = np.nanmean(norm_autos[inds[0], :], axis=0)
                            # Residual and normalized (using incoherent mean) auto-correlation
                            resautos = (
                                norm_autos[tile_i, :] / incoherent_mean
                                ) - np.nanmean(norm_autos[tile_i, :] / incoherent_mean)
                            gain_temp = rebin(
                                resautos[freq_use], (nmodes, freq_use.size)
                            )
                        else:
                            # dimension manipulation, add dim for mode fitting
                            # Subtract the mean so aliasing is reduced in the dft cable fitting
                            gain_temp = rebin(
                                gain_arr_dv[tile_i, freq_use]
                                - np.mean(gain_arr_dv[tile_i, freq_use]),
                                (nmodes, freq_use.size),
                            )
                        # freq_use matrix to multiply/collapse in fit
                        freq_mat = rebin(freq_use, (nmodes, freq_use.size))
                        # Perform DFT of gains to test modes
                        test_fits = np.sum(
                            np.exp(1j * 2 * np.pi / n_freq * modes * freq_mat)
                            * gain_temp,
                            axis=1,
                            )
                        # Pick out highest amplitude fit (mode_ind gives the index of the mode)
                        amp_use = np.max(np.abs(test_fits)) / freq_use.size
                        mode_ind = np.argmax(np.abs(test_fits))
                        # Phase of said fit
                        phase_use = np.arctan2(
                            test_fits[mode_ind].imag, test_fits[mode_ind].real
                        )
                        mode_i = modes[mode_ind, 0]

                        # Using the mode selected from the gains, optionally use the phase to find the amp and phase
                        if auto_ratio is not None:
                            # Find tiles which will not be accidently coherent in their cable reflection in order to reduce bias
                            #inds = np.where(
                            #    (n_tile)
                            #    & (mode_i_arr[:, pol_i] > 0)
                            #    & (np.abs(mode_i_arr[:, pol_i] - mode_i) > 0.01)
                            #)
                            inds = np.where(
                                (mode_i_arr[:, pol_i] > 0)
                                & ((np.abs(mode_i_arr[:, pol_i] - mode_i)) > 0.01)
                            )
                            residual_phase = np.arctan2(
                                gain_arr_dv[:, freq_use].imag, gain_arr_dv[:, freq_use].real
                            )
                            incoherent_residual_phase = residual_phase[
                                tile_i, :] - np.nanmean(residual_phase[inds[0], :], axis=0)
                            test_fits = np.sum(
                                np.exp(
                                    1j * 2 * np.pi / n_freq * mode_i * freq_use
                                )
                                * incoherent_residual_phase
                            )
                            # Factor of 2 from fitting just the phase
                            amp_use = 2 * np.abs(test_fits) / freq_use.size
                            # Factor of pi/2 from just fitting the phase
                            phase_use = (
                                np.arctan2(test_fits.imag, test_fits.real) + np.pi / 2
                            )
                    
                        else:
                            # Use nominal delay mode, but fit amplitude and phase of reflections
                            mode_fit = np.sum(
                            np.exp(1j * 2 * np.pi / n_freq * mode_i * freq_use)
                            * gain_arr_dv[tile_i, freq_use]
                            )
                            amp_use = np.abs(mode_fit) / freq_use[0].size
                            phase_use = np.arctan2(mode_fit.imag, mode_fit.real)

                        gain_mode_fit = amp_use * np.exp(
                        -1j * 2 * np.pi * (mode_i * np.arange(n_freq) / n_freq) + 1j * phase_use
                        )
                        if auto_ratio is not None:
                            # Only fit for the cable reflection in the phases
                            gains_cfit[tile_i, :, pol_i] *= np.exp(1j * gain_mode_fit.imag)
                        else:
                            gains_cfit[tile_i, :, pol_i] *= 1 + gain_mode_fit
                   
    return gains_cfit 

def vis_cal_auto_fit(vis_auto, 
                     model_auto,
                     gains,
                     freq_use,
                     ):
    """
    Solve for each tile's calibration amplitude via the square root of the ratio of the data autocorrelation
    to the model autocorrelation using the definition of a gain. Then, remove dependence on the correlated
    noise term in the autocorrelations by scaling this amplitude down to the crosscorrelations gain via a
    simple, linear fit. Build a full, scaled autocorrelation gain solution by adding in the phase term via
    the crosscorrelation gains.
    """
    avg_vis_auto = np.nanmean(vis_auto, axis=0)
    avg_model_auto = np.nanmean(model_auto, axis=0)
    n_tile, n_freq, n_pol = avg_vis_auto.shape
    freq_i_use = np.nonzero(freq_use)
    freq_i_flag = np.where(freq_use == 0)[0]
    auto_tile_i = np.arange(n_tile)
    # If the number of frequencies not being used is above 0, then ignore the frequencies surrounding them.
    if freq_i_flag.size > 0:
        freq_flag = np.zeros(n_freq)
        freq_flag[freq_i_use] = 1
        for freq_i in range(freq_i_flag.size):
            minimum = max(0, freq_i_flag[freq_i] - 1)
            maximum = min(n_freq, freq_i_flag[freq_i] + 2)
            freq_flag[minimum:maximum] = 0
        freq_i_use = np.nonzero(freq_flag)
    # Vectorized loop for via_cal_auto_fit lines 45-55 in IDL
    # However the logic still indexes the full 128 tiles, so need to shove
    # outputs into an empty array of correct size
    # We're not using the cross polarizations if they are present
    auto_gain = np.empty((n_tile, n_freq, n_pol))
    auto_gain[auto_tile_i, :, :] = np.sqrt(
        avg_vis_auto * weight_invert(avg_model_auto))
    gain_cross = gains
    gains_autofit = copy.deepcopy(gain_cross)
    fit_slope = np.empty((n_tile, n_pol))

    # Didn't vectorize as the polyfit won't be vectorized
    for pol_i in range(n_pol):
        for tile in range(auto_tile_i.size):
            tile_idx = auto_tile_i[tile]
            phase_cross_single = np.arctan2(
                gain_cross[tile_idx, :, pol_i].imag, gain_cross[tile_idx, :, pol_i].real
            )

            gain_auto_single = np.abs(auto_gain[tile_idx, :, calpols[pol_i]])
            gain_cross_single = np.abs(gain_cross[tile_idx, :, pol_i])

            # mask out any NaN values; numpy doesn't like them,
            # I assume the IDL equiv function just masks them?
            # or maybe we need to do a catch for NaNs here, and abandon all
            # hope for a fit if there are NaNs?
            notnan = np.where(
                (np.isnan(gain_auto_single[freq_i_use]) != True)
                & (np.isnan(gain_cross_single[freq_i_use]) != True)
            )
            gain_auto_single_fit = gain_auto_single[freq_i_use][notnan]
            gain_cross_single_fit = gain_cross_single[freq_i_use][notnan]

            # linfit from IDL uses chi-square error calculations to do the linear fit, instead of least squares.
            # The polynomial fit uses least square method
            x = np.vstack([gain_auto_single_fit, np.ones(gain_auto_single_fit.size)]).T
            fit_single = np.linalg.lstsq(x, gain_cross_single_fit, rcond=None)[0]
            # IDL gives the solution in terms of [A, B] while Python does [B, A] assuming we're
            # solving the equation y = A + Bx
            gains_autofit[tile_idx, :, pol_i] = (
                gain_auto_single * fit_single[0] + fit_single[1]
            ) * np.exp(1j * phase_cross_single)

    gains_auto_scale = np.sum(fit_slope, axis=1) / auto_tile_i.size

    return gains_autofit, gains_auto_scale
