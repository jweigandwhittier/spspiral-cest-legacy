#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:28:13 2026

@author: jonah

GRE readout code taken from https://github.com/imr-framework/pypulseq/blob/master/examples/scripts/write_gre.py
"""
import numpy as np
import pypulseq as pp
from utils import sim_cest_rf
from seqeyes import seqeyes

# ------------------------------------------
# System limits
# ------------------------------------------
sys = pp.opts.Opts( 
    max_grad = 40, 
    grad_unit = 'mT/m',
    max_slew = 100, 
    slew_unit = 'mT/m/ms',
    rf_ringdown_time = 20e-6, 
    rf_dead_time = 100e-6,
    adc_dead_time = 20e-6, 
    adc_raster_time = 10e-6,
    B0 = 3.00
    )

# ------------------------------------------
# CEST prep parameters
# ------------------------------------------
pulse_shape = 'gauss' # Gauss or SPSP
n_pulses = 1 
tp = 10e-3 # [s]; pulse length
td = 3e-3 # [s]; interpulse delay (total with spoilers if n_pulses > 1)
b1s = np.arange(0, 16, 4) # [uT]; can be a single value or a list
offsets = np.arange(4, 22, 2) # [ppm]; can be a single value or a list
spoil_rise_time = 1e-3
spoil_dur = 6.5e-3
spoil_amp = 0.8 * sys.max_grad

# ------------------------------------------
# Readout parameters
# ------------------------------------------
fov = 150e-3 # [m]
n_x = n_y = 64
flip_angle_deg = 11.421 # [°]; using Ernst angle for T1 = 2000 ms 
slice_thickness = 8e-3 # [m]
tr = 40e-3 # [s]
te = 5e-3 # [s]

# ------------------------------------------
# Other scan parameters
# ------------------------------------------
dummy = 32 # Number of dummy scans (prep and readout)
b1_map_fn = 'string' # Filename for B1 map (.npy or .dcm) for SPSP pulses
b1_seq_fn = 'dicom' # Filename for sequence with defs or 'dicom'

gamma_hz = sys.gamma * 1e-6
freq = sys.B0 * gamma_hz

rf_spoiling_inc = 117

# ------------------------------------------
# Prepare to write sequence
# ------------------------------------------
align_to_raster = lambda t: np.ceil(t / sys.grad_raster_time) * sys.grad_raster_time

# Align times to raster
tp = align_to_raster(tp)
td = align_to_raster(td)
spoil_rise_time = align_to_raster(spoil_rise_time)
spoil_dur = align_to_raster(spoil_dur)
    
# Get FOV parameters 
fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

# ------------------------------------------
# Make CEST prep objects
# ------------------------------------------
# Calculate spsp pulse shape (if needed)
if pulse_shape == 'spsp':
    spsp_objects = sim_cest_rf.calc_spsp(b1_map_fn, b1_seq_fn, tp, sys)
    spsp_grad_x = spsp_objects['full_gx']
    spsp_grad_y = spsp_objects['full_gy']
    spsp_rf_shape = spsp_objects['full_rf']
# Make (placeholder) sat pulse
sat_pulse = pp.make_gauss_pulse(flip_angle=np.pi,
                                duration=tp,
                                time_bw_product=0.2,
                                apodization=0.5,
                                delay=sys.rf_dead_time,
                                freq_offset=0,
                                system=sys,
                                use='preparation')

# Make delay in case B1 = 0 uT (for reference)
sat_pulse_dur = pp.calc_duration(sat_pulse)
sat_delay = pp.make_delay(sat_pulse_dur)

# Make spoiler
gx_spoil_cest, gy_spoil_cest, gz_spoil_cest = [
    pp.make_trapezoid(channel=c, system=sys, amplitude=spoil_amp,
                      duration=spoil_dur, rise_time=spoil_rise_time)
    for c in ["x", "y", "z"]
    ]

# Make interpulse delay (if needed)
interpulse_delay = None
if n_pulses > 1 and td:
    if td < spoil_dur:
        print('CEST spoiler duration is longer than requested interpulse delay. Defaulting to spoiler duration.')
    else:
        td -= spoil_dur
        interpulse_delay = pp.make_delay(td)

# ------------------------------------------
# Make readout objects
# ------------------------------------------
# Excitation and slice select
rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=sys,
        return_gz=True,
        delay=sys.rf_dead_time,
        use='excitation')

# Gradients and ADC
delta_kx = 1 / fov_x
delta_ky = 1 / fov_y
gx = pp.make_trapezoid(channel='x', flat_area=n_x * delta_kx, flat_time=3.2e-3, system=sys)
adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=sys)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=sys)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=sys)
phase_areas = (np.arange(n_y) - n_y / 2) * delta_ky

# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel='x', area=2 * n_x * delta_kx, system=sys)
gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=sys)

# Calculate timing
te_delay = (
    te
    - (pp.calc_duration(gz, rf) - pp.calc_rf_center(rf)[0] - rf.delay)
    - pp.calc_duration(gx_pre)
    - pp.calc_duration(gx) / 2
    - pp.eps)
te_delay = align_to_raster(te_delay)

tr_delay = tr - pp.calc_duration(gz, rf) - pp.calc_duration(gx_pre) - pp.calc_duration(gx) - te_delay
tr_delay = align_to_raster(tr_delay)

assert np.all(te_delay >= 0)
assert np.all(tr_delay >= pp.calc_duration(gx_spoil, gz_spoil))

# ------------------------------------------
# Construct sequence
# ------------------------------------------
# Initialize sequence
seq = pp.Sequence()

# Iterate through events
rf_phase = 0
rf_inc = 0
n_offsets = len(offsets)
for i, b1 in enumerate(b1s): # Iterate through B1 in first loop 
    if b1 != 0:
        target_peak_hz = b1 * gamma_hz
        current_peak_hz = np.max(np.abs(sat_pulse.signal))
        sat_pulse.signal *= (target_peak_hz / current_peak_hz)
        dt = sys.rf_raster_time
        total_flip_angle = np.abs(np.sum(sat_pulse.signal)) * dt * 2 * np.pi
        if pulse_shape == 'spsp':
            spsp_pulse = pp.make_arbitrary_rf(spsp_rf_shape, 
                                              flip_angle=total_flip_angle, 
                                              dwell=sys.rf_raster_time, # JWW change for now
                                              delay=sys.rf_dead_time,
                                              freq_offset=0,
                                              system=sys,
                                              use='preparation')
    for m, offset in enumerate(offsets): # Iterate through offsets in inner 
        i_rep = i * n_offsets + m
        seq.add_block(pp.make_label('REP', 'SET', i_rep))
        offset_hz = offset * freq
        sat_pulse.freq_offset = offset_hz
        if pulse_shape == 'spsp':
            spsp_pulse.freq_offset = offset_hz
        if i == m == 0: # Only run dummy scans once at the very beginning
            for d in range(dummy):
                for n in range(n_pulses):
                    is_last = (n == n_pulses - 1)
                    if b1 == 0:
                        seq.add_block(sat_delay)
                    elif pulse_shape == 'spsp':
                        seq.add_block(spsp_pulse, spsp_grad_x, spsp_grad_y)
                    else:
                        seq.add_block(sat_pulse)
                    seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)
                    if interpulse_delay and not is_last:
                        seq.add_block(interpulse_delay)
                rf.phase_offset = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi
                rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
                seq.add_block(rf, gz)
                gy_pre = pp.make_trapezoid(
                    channel='y',
                    area=0, # No phase encode for dummy
                    duration=pp.calc_duration(gx_pre),
                    system=sys)
                seq.add_block(gx_pre, gy_pre, gz_reph)
                seq.add_block(pp.make_delay(te_delay))
                seq.add_block(gx) # No ADC for dummy 
                gy_pre.amplitude = -gy_pre.amplitude
                seq.add_block(pp.make_delay(tr_delay), gx_spoil, gy_pre, gz_spoil)
        for i_phase in range(n_y):
            for n in range(n_pulses):
                is_last = (n == n_pulses - 1)
                if b1 == 0:
                    seq.add_block(sat_delay)
                elif pulse_shape == 'spsp':
                    seq.add_block(spsp_pulse, spsp_grad_x, spsp_grad_y)
                else:
                    seq.add_block(sat_pulse)
                seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)
                if interpulse_delay and not is_last:
                    seq.add_block(interpulse_delay)
            seq.add_block(pp.make_label('LIN', 'SET', i_phase))
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
            seq.add_block(rf, gz)
            gy_pre = pp.make_trapezoid(
                channel='y',
                area=phase_areas[i_phase],
                duration=pp.calc_duration(gx_pre),
                system=sys)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(pp.make_delay(te_delay))
            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(pp.make_delay(tr_delay), gx_spoil, gy_pre, gz_spoil)

# ------------------------------------------
# Check and write sequence
# ------------------------------------------
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]
    
seq_filename = f'sequences/qmt/qmt_gre_{pulse_shape}_{n_offsets}_offsets_{len(b1s)}_b1s.seq'

seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness])

seq.write(seq_filename)
seqeyes(seq_filename)
                
        
    


