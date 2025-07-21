import numpy as np
from mwa_qa.read_uvfits import UVfits
from mwa_qa.read_calfits import CalFits
import calfits_tools as ct
import argparse


def initial_auto_scaling(uvfits, model_uvfits, metafits, filename=None):
    init_gain = evaluate_auto_init(uvfits, model_uvfits)
    if filename is None:
        filename = uvfits.replace('.uvfits', '_initial_gains.fits')
    ct.create_calibration_fits(metafits, solutions_data=init_gain, output_name=filename)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description ='performing autocalibration.')
    parser.add_argument('--uvfits', dest='uvfits', help='Path to observation file')
    parser.add_argument('--model_uvfits', dest='model_uvfits', help='Path to model file')
    parser.add_argument('--calfile', dest='calfile', default=None, help='Path to calfits file')
    parser.add_argument('--metafits', dest='metafits', help='Path to metafits file')
    parser.add_argument('--reference_antenna', dest='reference_antenna', default=127, type=int, help='Reference antenna')
    parser.add_argument('--initialise_gains', dest='initialise_gains', action='store_true', default=None, help='Inilitialise the gains for faster convergence') 
    parser.add_argument('--phase_fit', dest='phase_fit', action='store_true', default=None, help='Allows for linear fitting of the gain solutions')
    parser.add_argument('--cal_reflection_mode_theory', dest='cal_reflection_mode_theory', action='store_true', default=None, help='Evaluate reflection modes using analytical calcualtions')
    parser.add_argument('--cal_reflection_mode_delay', dest='cal_reflection_mode_delay', action='store_true', default=None, help='Evaluate reflection modes using delay transform')
    parser.add_argument('--cal_reflection_hyperresolve', dest='cal_reflection_hyperresolve', default=None, action='store_true', help='Allows for hypersolving of the modes')
    parser.add_argument('--cable_fit', dest='cable_fit', default=None, action='store_true', help='Calibrate per-frequency amplitudes by averaging by cable group') 
    parser.add_argument('--reflection_fit', dest='reflection_fit', default=None, action='store_true', help='Solve for cable reflection') 
    parser.add_argument('--poly_amplitude_order', dest='poly_amplitude_order', default=2, help='Solve for cable reflection. Default order is 1.')
    parser.add_argument('--poly_phase_order', dest='poly_phase_order', default=1, help='Polyfitting order for ampitude fitting. Default order is 2.') 
    parser.add_argument('--outfile', dest='outfile', default=None, help='Name of output file') 

    args = parser.parse_args()
    # reading observation
    #uvf = UVfits(args.uvfits)
    # model uvfits
    #mod_uvf = UVfits(args.model_uvfits)
    # reading calfits
    cal = CalFits(args.calfile)
    gains = cal.gain_array
    freq_use = np.arange(len(cal.frequency_flags))
    if cal.frequency_flags.ndim > 1: 
        freq_use =  freq_use[np.where(cal.frequency_flags[:, 0] ==0 )]
    else:
        freq_use =  freq_use[np.where(cal.frequency_flags ==0 )]
    convergence = cal.convergence
    frequency_array = cal.frequency_array
    #if args.initialise_gains:
    #    initial_calfile = uvfits.replace('.uvfits', '_initial_gains.fits') 
    #    initial_auto_scaling(uvfits, model_uvfits, arg.metafits,filename=initial_calfile)
    
    # Perform bandpass (amp + phase per fine freq) and polynomial fitting (low order amp + phase fit plus cable reflection fit)
    _, vis_auto = ct.get_data(args.uvfits)
    _, mod_auto = ct.get_data(args.model_uvfits)  
    auto_gains, auto_ratio = ct.cal_auto_ratio_divide(gains, vis_auto, args.reference_antenna)
    cal_bandpass, cal_remainder = ct.vis_cal_bandpass(auto_gains, args.metafits, freq_use, cable_fit=args.cable_fit)
    
    if args.phase_fit:
        cal_polyfit = ct.vis_cal_polyfit(cal_remainder,
                    freq_use,
                    args.metafits,
                    auto_ratio,
                    poly_amplitude_order=2, 
                    poly_phase_order=1,
                    reflection_fit=args.reflection_fit,
                    cal_reflection_mode_theory=args.cal_reflection_mode_theory,
                    cal_reflection_mode_delay=args.cal_reflection_mode_theory,
                    cal_reflection_hyperresolve=args.cal_reflection_hyperresolve)
        # Replace vis_cal_combine with this line as the gain is the same size for polyfit and bandpass
        cal_gains = cal_polyfit * cal_bandpass
    else:
        cal_gains = cal_bandpass

    #cal_gains = ct.cal_auto_ratio_remultiply(cal_gains, auto_ratio)
    # writing to fits file
    cal_gains = ct.cal_auto_ratio_remultiply(cal_gains, auto_ratio)
    cal_gains_autos, cal_gain_scaler = ct.vis_cal_auto_fit(vis_auto, mod_auto, cal_gains, freq_use)
    # These subtractions replace vis_cal_subtract
    _sh = cal_gains_autos.shape
    cal.gain_array = cal_gains_autos.reshape((1, _sh[0], _sh[1], _sh[2]))
    if args.outfile is None:
        outfile = args.calfile.replace('.fits', '.auto_scaled.fits')
    else:
        outfile = args.outfile
    cal.write_to(outfile, overwrite=True)