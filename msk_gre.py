#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

@author: jonah

Combines the dummy -> reverse-spoiled centric-polarity GRE readout scheme
from write_gre_cest.py with the ppm-offset / generic-SPSP machinery from
continuous_spiral.py.

Produces TWO distinct sequences depending on FLAG_MODE (run the script
twice, once per mode):

  1) FLAG_MODE = "crpcr"   -> ONE .seq containing: Ref, +2ppm, -2ppm with
                               Gauss, immediately followed by a repeat of
                               Ref, +2ppm, -2ppm with SPSP (calculated or
                               generic, per FLAG_GENERIC). Pulse shape is
                               now a per-segment property, not a global
                               switch -- both halves land in one file.
  2) FLAG_MODE = "ph" -> Ref, +3.5ppm x N powers, -3.5ppm x N powers,
                               all Gauss, all in ONE .seq, CWPE-targeted.
"""
import os
import numpy as np
import pypulseq as pp
from scipy.optimize import brentq
from utils import sim_cest_rf
from seqeyes import seqeyes
from bmctool.utils.pulses.calc_power_equivalents import calc_power_equivalent

# ========================================== #
# Sequence flags
# ========================================== #
FLAG_MODE = "crpcr"    # "crpcr" or "ph"
FLAG_GENERIC = True    
FLAG_SEQEYES = True
FLAG_PLOT = False
FLAG_SIM = False    
FLAG_MINIMIZE_TR = False  
FLAG_GRAPPA = True

# ========================================== #
# System limits (from write_gre_cest.py)
# ========================================== #
system = pp.opts.Opts(
    max_grad=40,
    grad_unit='mT/m',
    max_slew=100,
    slew_unit='mT/m/ms',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=20e-6,
    adc_raster_time=10e-6,
    B0=3.00
)

gamma_hz = system.gamma * 1e-6
freq = system.B0 * gamma_hz  # Hz per ppm, for ppm -> Hz conversion

align_to_raster = lambda t: np.ceil(t / system.grad_raster_time) * system.grad_raster_time

# ========================================== #
# CEST prep parameters
# ========================================== #
tp = align_to_raster(36e-3)                  # [s] sat pulse duration
spoil_rise_time = align_to_raster(1e-3)
spoil_dur = align_to_raster(6.5e-3)
spoil_amp = 0.8 * system.max_grad

CRPCR_OFFSETS_PPM = [2.00, -2.00]            
CRPCR_B1_UT = 1.20                            
PH_OFFSET_PPM = 3.50                    
PH_CWPE_LIST_UT = [0.05, 0.125, 0.25, 0.50]

# ========================================== #
# Readout parameters (from write_gre_cest.py, unchanged)
# ========================================== #
fov = 150e-3
n_x = n_y = 64
GRAPPA_R = 2       # Acceleration factor, undersampling outside the ACS region
GRAPPA_N_ACS = 32  # Fully-sampled auto-calibration lines at k-space center
slice_thickness = 8e-3
tr = 50e-3  
te = 5e-3
dummy = 30
rf_spoiling_inc = 117
t1_guess = 1.9
ernst_angle = np.arccos(np.exp(-tr / t1_guess))  

recovery_delay_s = align_to_raster(5 * t1_guess)  # 5x T1 recovery between segments

# ========================================== #
# SPSP pulse (from continuous_spiral.py) -- only needed for the "crpcr"
# ========================================== #
USE_SPSP = (FLAG_MODE == "crpcr")

if USE_SPSP:
    if FLAG_GENERIC:
        b1_map = np.load('data/recon/cindy_example_b1.npy')
        wasabi_seq_filename = 'sequences/example/cindy_b1.seq'
        mask = np.load('generic_spsp/mask.npy')
    else:
        b1_map = np.load('data/recon/YOUR_B1_MAP.npy') 
        wasabi_seq_filename = 'YOUR_WASABI_SEQ.seq'          
        mask = None

    spsp_objects = sim_cest_rf.calc_spsp(b1_map, wasabi_seq_filename, tp, system, mask)
    spsp_grad_x = spsp_objects['full_gx']
    spsp_grad_y = spsp_objects['full_gy']
    spsp_rf_shape = spsp_objects['full_rf']

# ========================================== #
# Sat pulse template (Gauss) -- always built; used directly for pulse_type
# 'gauss', and as the flip-angle reference for SPSP scaling otherwise.
# ========================================== #
def make_gauss_sat_template():
    return pp.make_gauss_pulse(
        flip_angle=np.pi,
        duration=tp,
        time_bw_product=0.2,
        apodization=0.5,
        delay=system.rf_dead_time,
        freq_offset=0,
        system=system,
        use='preparation'
    )


def build_sat_block(pulse_type, b1p_uT, offset_hz):
    """
    Returns the RF pulse object(s) to add_block for a saturation shot at
    the given peak B1 [uT] and frequency offset [Hz], for pulse_type in
    {'gauss', 'spsp'}. Always anchors the target flip via a scaled Gauss
    template; for 'spsp', transfers that flip angle onto the SPSP waveform
    (matches continuous_spiral.py).
    """
    gauss_pulse = make_gauss_sat_template()
    target_peak_hz = b1p_uT * gamma_hz
    current_peak_hz = np.max(np.abs(gauss_pulse.signal))
    gauss_pulse.signal *= (target_peak_hz / current_peak_hz)
    gauss_pulse.freq_offset = offset_hz

    if pulse_type == 'gauss':
        return (gauss_pulse,)
    elif pulse_type != 'spsp':
        raise ValueError(f"Unknown pulse_type: {pulse_type}")

    dt = system.rf_raster_time
    total_flip_angle = np.abs(np.sum(gauss_pulse.signal)) * dt * 2 * np.pi
    spsp_pulse = pp.make_arbitrary_rf(
        spsp_rf_shape,
        flip_angle=total_flip_angle,
        dwell=system.rf_raster_time,
        delay=system.rf_dead_time,
        freq_offset=offset_hz,
        system=system,
        use='preparation'
    )
    return (spsp_pulse, spsp_grad_x, spsp_grad_y)


# ========================================== #
# CWPE -> B1 solver (numeric, pulse-shape agnostic on the Gauss anchor)
# ========================================== #
def cwpe_for_b1(b1p_uT):
    gauss_pulse = make_gauss_sat_template()
    target_peak_hz = b1p_uT * gamma_hz
    current_peak_hz = np.max(np.abs(gauss_pulse.signal))
    gauss_pulse.signal *= (target_peak_hz / current_peak_hz)
    cwpe_td = tr - tp
    return calc_power_equivalent(rf_pulse=gauss_pulse, tp=tp, td=cwpe_td, gamma_hz=gamma_hz)


def solve_b1_for_cwpe(target_cwpe_uT, b1_hi=5.0):
    if target_cwpe_uT <= 0:
        return 0.0
    f = lambda b1: cwpe_for_b1(b1) - target_cwpe_uT
    # Expand upper bracket if needed (should not be, in practice)
    while f(b1_hi) < 0:
        b1_hi *= 2
        if b1_hi > 50:
            raise RuntimeError(f"Could not bracket target CWPE={target_cwpe_uT} uT")
    return brentq(f, 1e-6, b1_hi)


# ========================================== #
# CEST spoilers
# ========================================== #
gx_spoil_cest, gy_spoil_cest, gz_spoil_cest = [
    pp.make_trapezoid(channel=c, system=system, amplitude=spoil_amp,
                       duration=spoil_dur, rise_time=spoil_rise_time)
    for c in ["x", "y", "z"]
]
sat_delay_block = pp.make_delay(pp.calc_duration(make_gauss_sat_template()))

# ========================================== #
# Readout objects (from write_gre_cest.py, unchanged)
# ========================================== #
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=ernst_angle,
    duration=3e-3,
    slice_thickness=slice_thickness,
    apodization=0.42,
    time_bw_product=4,
    system=system,
    return_gz=True,
    delay=system.rf_dead_time,
    use='excitation'
)

delta_kx = 1 / fov
delta_ky = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=-n_x * delta_kx, flat_time=3.2e-3, system=system)
adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=system)
adc_sim = pp.make_adc(num_samples=1, dwell=system.adc_raster_time, delay=system.adc_dead_time, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
phase_areas = -(np.arange(n_y) - n_y / 2) * delta_ky
center = n_y // 2
acquired_lines = np.arange(n_y)
if FLAG_GRAPPA:  # Prune to ACS region + every GRAPPA_R-th line in the wings
    idx_center = np.arange(center - GRAPPA_N_ACS // 2, center + GRAPPA_N_ACS // 2)
    idx_wings_bottom = np.arange(0, idx_center[0], GRAPPA_R)
    idx_wings_top = np.arange(idx_center[-1] + 1, n_y, GRAPPA_R)
    acquired_lines = np.unique(np.concatenate([idx_wings_bottom, idx_center, idx_wings_top]))
    print(f"GRAPPA R={GRAPPA_R} enabled. Acquired lines: {len(acquired_lines)} / {n_y} "
          f"({GRAPPA_N_ACS}-line ACS region).")
# Center (still in the ACS region when FLAG_GRAPPA is on) lands last either way.
phase_order = sorted(acquired_lines.tolist(), key=lambda k: abs(k - center), reverse=True)

gx_spoil = pp.make_trapezoid(channel='x', area=2 * n_x * delta_kx, system=system)
gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

te_delay = (
    te
    - (pp.calc_duration(gz, rf) - pp.calc_rf_center(rf)[0] - rf.delay)
    - pp.calc_duration(gx_pre)
    - pp.calc_duration(gx) / 2
    - pp.eps)
te_delay = align_to_raster(te_delay)

prep_duration = pp.calc_duration(make_gauss_sat_template()) + pp.calc_duration(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)
exc_duration = pp.calc_duration(gz, rf)
prephaser_duration = pp.calc_duration(gx_pre)
readout_duration = pp.calc_duration(gx)
end_spoil_duration = pp.calc_duration(gx_spoil, gz_spoil)

min_tr = align_to_raster(
    prep_duration + exc_duration + prephaser_duration + te_delay + readout_duration + end_spoil_duration
)

if FLAG_MINIMIZE_TR:
    print(f"[tr] FLAG_MINIMIZE_TR=True -> using minimum feasible tr = {min_tr * 1e3:.3f} ms "
          f"(requested tr was {tr * 1e3:.3f} ms)")
    tr = min_tr
elif tr < min_tr:
    print(f"[tr] Requested tr={tr * 1e3:.3f} ms is below the minimum feasible tr "
          f"({min_tr * 1e3:.3f} ms, dominated by prep_duration={prep_duration * 1e3:.3f} ms) "
          f"-- clamping up to the minimum.")
    tr = min_tr

ernst_angle = np.arccos(np.exp(-tr / t1_guess))
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=ernst_angle,
    duration=3e-3,
    slice_thickness=slice_thickness,
    apodization=0.42,
    time_bw_product=4,
    system=system,
    return_gz=True,
    delay=system.rf_dead_time,
    use='excitation'
)

tr_delay = (
    tr
    - prep_duration
    - exc_duration
    - prephaser_duration
    - te_delay
    - readout_duration
    - end_spoil_duration
)
tr_delay = align_to_raster(tr_delay)

assert np.all(te_delay >= 0)
assert np.all(tr_delay >= 0), (
    f"tr_delay is negative ({tr_delay * 1e3:.3f} ms): prep ({prep_duration * 1e3:.3f} ms) "
    f"+ exc ({exc_duration * 1e3:.3f} ms) + prephaser "
    f"({prephaser_duration * 1e3:.3f} ms) + te_delay ({te_delay * 1e3:.3f} ms) "
    f"+ readout ({readout_duration * 1e3:.3f} ms) + end_spoil "
    f"({end_spoil_duration * 1e3:.3f} ms) exceeds tr ({tr * 1e3:.3f} ms). "
    f"Increase tr or shorten tp/spoilers -- tp alone ({tp * 1e3:.3f} ms) may not fit."
)

# ========================================== #
# Labels
# ========================================== #
seg_label_ref = pp.make_label('TRID', 'SET', 1)        # reference (no sat)
seg_label_sat_gauss = pp.make_label('TRID', 'SET', 2)   # saturated, Gauss
recovery_label = pp.make_label('TRID', 'SET', 3)        # inter-segment recovery delay
seg_label_sat_spsp = pp.make_label('TRID', 'SET', 4)    # saturated, SPSP

# ========================================== #
# Core acquisition: dummy shots + reverse-polarity-spoiled centric readout
# ========================================== #
def run_dummy_prep(seq, is_ref, pulse_type, b1p_uT, offset_hz, is_sim=False):
    """
    Runs the `dummy` steady-state prep shots (sat + spoiler + excitation +
    imaging gradient, no ADC) shared by both the real image acquisition and
    its SIM counterpart, so both reach the same magnetization steady state
    before the "actual image" portion. Returns (rf_phase, rf_inc) so the
    caller's real readout continues RF spoiling from the right phase.

    is_sim: per BMCTool's actual block classification (checked in this
    priority order): 1) ADC present -> record readout, ignore rest;
    2) RF present (no ADC) -> simulate the RF pulse, ignore rest;
    3) z-gradient present (no ADC/RF) -> assume spoiling, ignore rest;
    4) otherwise (x/y-gradient or delay only) -> simulate as a delay.
    The (gx_pre, gy_pre, gz_reph) prephaser block below carries a
    z-gradient (gz_reph) and has no RF/ADC, so under rule 3 it WOULD be
    read as a spoiler -- but it's not supposed to be one, it sits right
    before the echo we want to read out. For is_sim=True we swap it for
    a plain delay of the same duration (no gradient channels at all, so
    it falls under rule 4 instead) -- gradient content there does
    nothing in a non-spatial BMC sim anyway. The actual end-of-TR
    spoiler block (gx_spoil, gy_pre, gz_spoil) keeps its z-gradient
    (gz_spoil) on purpose in both cases -- that one SHOULD hit rule 3,
    since it's modeling real RF+gradient spoiling between TRs.
    """
    rf_phase = 0
    rf_inc = 0
    for d in range(dummy):
        if is_ref:
            seq.add_block(sat_delay_block)
        else:
            seq.add_block(*build_sat_block(pulse_type, b1p_uT, offset_hz))
        seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)

        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        seq.add_block(rf, gz)
        gy_pre = pp.make_trapezoid(channel='y', area=0, duration=pp.calc_duration(gx_pre), system=system)
        if is_sim:
            seq.add_block(pp.make_delay(pp.calc_duration(gx_pre, gy_pre, gz_reph)))
        else:
            seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(pp.make_delay(te_delay))
        seq.add_block(gx)
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(pp.make_delay(tr_delay), gx_spoil, gy_pre, gz_spoil)
    return rf_phase, rf_inc


def acquire_segment(seq, is_ref, pulse_type, b1p_uT, offset_hz, rep_idx):
    if is_ref:
        seg_label = seg_label_ref
    else:
        seg_label = seg_label_sat_spsp if pulse_type == 'spsp' else seg_label_sat_gauss

    rf_phase, rf_inc = run_dummy_prep(seq, is_ref, pulse_type, b1p_uT, offset_hz)

    # --- Actual image --- (outside-in / reverse-centric phase order --
    # see phase_order above; center k-space lands last, right after the
    # dummy prep has reached steady state)
    seq.add_block(pp.make_label('REP', 'SET', rep_idx), seg_label)
    for i_phase in phase_order:
        if is_ref:
            seq.add_block(sat_delay_block)
        else:
            seq.add_block(*build_sat_block(pulse_type, b1p_uT, offset_hz))
        seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)

        seq.add_block(pp.make_label('LIN', 'SET', i_phase))
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        seq.add_block(rf, gz)

        gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[i_phase],
                                    duration=pp.calc_duration(gx_pre), system=system)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(pp.make_delay(te_delay))
        seq.add_block(gx, adc)
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(pp.make_delay(tr_delay), gx_spoil, gy_pre, gz_spoil)


def acquire_segment_sim(seq, is_ref, pulse_type, b1p_uT, offset_hz, rep_idx):
    """
    SIM counterpart of acquire_segment: same dummy prep AND the same
    outside-in phase_order loop as the real scan (so RF spoiling phase,
    per-TR timing, and T1 recovery through the non-center lines all
    match the real acquisition) -- but only the LAST line in phase_order
    (i_phase == center) carries a real ADC (adc_sim); every other line
    gets a plain delay of the same duration instead of (gx, adc), so
    BMCTool only records one readout per segment while still evolving
    magnetization through the full n_y-TR acquisition.

    This matters beyond just timing cosmetics: the real scan takes
    n_y-1 extra outside-in TRs (continued sat/excite/spoil) between the
    end of dummy prep and the center-k-space read. The earlier collapsed
    version (dummy prep -> single acquisition) skipped modeling all of
    that -- i.e. it under-ran the total saturation exposure relative to
    the real scan by (n_y-1)*tr, which would suppress simulated contrast
    below what the real sequence should produce.
    """
    if is_ref:
        seg_label = seg_label_ref
    else:
        seg_label = seg_label_sat_spsp if pulse_type == 'spsp' else seg_label_sat_gauss

    rf_phase, rf_inc = run_dummy_prep(seq, is_ref, pulse_type, b1p_uT, offset_hz, is_sim=True)

    seq.add_block(pp.make_label('REP', 'SET', rep_idx), seg_label)
    ro_dur = pp.calc_duration(gx, adc)  # real combined readout-block duration, for non-center padding
    for i_phase in phase_order:
        if is_ref:
            seq.add_block(sat_delay_block)
        else:
            seq.add_block(*build_sat_block(pulse_type, b1p_uT, offset_hz))
        seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)

        seq.add_block(pp.make_label('LIN', 'SET', i_phase))
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        adc_sim.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        seq.add_block(rf, gz)

        gy_pre = pp.make_trapezoid(channel='y', area=phase_areas[i_phase],
                                    duration=pp.calc_duration(gx_pre), system=system)
        seq.add_block(pp.make_delay(pp.calc_duration(gx_pre, gy_pre, gz_reph)))
        seq.add_block(pp.make_delay(te_delay))

        if i_phase == center:
            seq.add_block(adc_sim)
            pad = ro_dur - pp.calc_duration(adc_sim)
            if pad > 0:
                seq.add_block(pp.make_delay(align_to_raster(pad)))
        else:
            seq.add_block(pp.make_delay(ro_dur))

        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(pp.make_delay(tr_delay), gx_spoil, gy_pre, gz_spoil)


# ========================================== #
# Build the segment list for the selected mode
# ========================================== #
segments = []

if FLAG_MODE == "crpcr":
    segments.append(("ref", True, 'gauss', 0.0, 0.0, 0.0))  # shared: ref is 0 uT sat, pulse_type is moot
    for pulse_type in ('gauss', 'spsp'):
        for off_ppm in CRPCR_OFFSETS_PPM:
            segments.append((f"{off_ppm:+.2f}ppm_{pulse_type}", False, pulse_type, CRPCR_B1_UT, off_ppm, 0.0))

elif FLAG_MODE == "ph":
    segments.append(("ref_gauss", True, 'gauss', 0.0, 0.0, 0.0))
    cwpe_b1_pairs = [(c, solve_b1_for_cwpe(c)) for c in PH_CWPE_LIST_UT]
    for sign, off_ppm in [(+1, PH_OFFSET_PPM), (-1, -PH_OFFSET_PPM)]:
        for cwpe_uT, b1p_uT in cwpe_b1_pairs:
            print(f"[ph] target CWPE={cwpe_uT:.3f} uT -> solved peak B1={b1p_uT:.4f} uT "
                  f"@ offset={off_ppm:+.2f} ppm")
            segments.append((f"{off_ppm:+.2f}ppm_cwpe{cwpe_uT:.3f}uT_gauss", False, 'gauss', b1p_uT, off_ppm, cwpe_uT))
else:
    raise ValueError(f"Unknown FLAG_MODE: {FLAG_MODE}")

# ========================================== #
# Build sequence(s)
# ========================================== #
seq = pp.Sequence(system=system)
seq_sim = pp.Sequence(system=system) if FLAG_SIM else None
seg_names = []
sim_offsets_ppm = []  # one entry per segment == one ADC event in seq_sim
sim_b1_ut = []        # parallel array, same order (solved peak B1)
sim_cwpe_ut = []      # parallel array, same order (CWPE target, 0.0 where n/a)
rep_idx = 0
for i, (name, is_ref, pulse_type, b1p_uT, off_ppm, cwpe_uT) in enumerate(segments):
    offset_hz = off_ppm * freq
    acquire_segment(seq, is_ref, pulse_type, b1p_uT, offset_hz, rep_idx)
    if FLAG_SIM:
        acquire_segment_sim(seq_sim, is_ref, pulse_type, b1p_uT, offset_hz, rep_idx)
        sim_offsets_ppm.append(off_ppm)
        sim_b1_ut.append(b1p_uT)
        sim_cwpe_ut.append(cwpe_uT)
    seg_names.append(name)
    rep_idx += 1

    is_last = (i == len(segments) - 1)
    if not is_last:
        seq.add_block(pp.make_delay(recovery_delay_s), recovery_label)
        if FLAG_SIM:
            seq_sim.add_block(pp.make_delay(recovery_delay_s), recovery_label)

# ========================================== #
# Timing check + save
# ========================================== #
output_dir = 'sequences/gre_cest_ppm'
os.makedirs(output_dir, exist_ok=True)

if FLAG_MODE == "crpcr":
    seq_name = 'crpcr_gre_gauss_then_spsp'
else:
    seq_name = 'ph_gre_gauss'
if FLAG_GRAPPA:
    seq_name += f'_grappa_r{GRAPPA_R}'

ok, error_report = seq.check_timing()
if ok:
    print(f'[{seq_name}] Timing check passed successfully!')
else:
    print(f'[{seq_name}] Timing check FAILED! Error listing follows:')
    [print(e) for e in error_report]

seq.set_definition('Name', seq_name)
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Mode', FLAG_MODE)
seq.set_definition('SegmentNames', seg_names)
seq.set_definition('RecoveryDelay', float(recovery_delay_s))
seq.set_definition('Grappa', FLAG_GRAPPA)
if FLAG_GRAPPA:
    seq.set_definition('GrappaR', GRAPPA_R)
    seq.set_definition('GrappaAcsLines', GRAPPA_N_ACS)
if FLAG_MODE == "crpcr":
    seq.set_definition('DistinctOffsetsPpm', [0.0] + CRPCR_OFFSETS_PPM)
    seq.set_definition('B1_uT', CRPCR_B1_UT)
    seq.set_definition('SpspGeneric', FLAG_GENERIC)
else:
    seq.set_definition('DistinctOffsetsPpm', [0.0, PH_OFFSET_PPM, -PH_OFFSET_PPM])
    seq.set_definition('CwpeTargets_uT', PH_CWPE_LIST_UT)

seq_filename = f'{output_dir}/{seq_name}.seq'
seq.write(seq_filename)

if FLAG_SEQEYES:
    seqeyes(seq_filename)

if FLAG_PLOT:
    seq.plot()

if FLAG_SIM:
    sim_output_dir = f'{output_dir}/sim'
    os.makedirs(sim_output_dir, exist_ok=True)

    ok_sim, error_report_sim = seq_sim.check_timing()
    if ok_sim:
        print(f'[{seq_name}_sim] Timing check passed successfully!')
    else:
        print(f'[{seq_name}_sim] Timing check FAILED! Error listing follows:')
        [print(e) for e in error_report_sim]

    seq_sim.set_definition('Name', f'{seq_name}_sim')
    seq_sim.set_definition('Mode', FLAG_MODE)

    seq_sim.set_definition('offsets_ppm', np.array(sim_offsets_ppm))
    seq_sim.set_definition('SegmentB1_uT', np.array(sim_b1_ut))
    seq_sim.set_definition('SegmentCwpe_uT', np.array(sim_cwpe_ut))
    seq_sim.set_definition('RecoveryDelay', float(recovery_delay_s))
    seq_sim.set_definition('Grappa', FLAG_GRAPPA)
    if FLAG_GRAPPA:
        seq_sim.set_definition('GrappaR', GRAPPA_R)
        seq_sim.set_definition('GrappaAcsLines', GRAPPA_N_ACS)
    if FLAG_MODE == "crpcr":
        seq_sim.set_definition('DistinctOffsetsPpm', [0.0] + CRPCR_OFFSETS_PPM)
        seq_sim.set_definition('B1_uT', CRPCR_B1_UT)
        seq_sim.set_definition('SpspGeneric', FLAG_GENERIC)
    else:
        seq_sim.set_definition('DistinctOffsetsPpm', [0.0, PH_OFFSET_PPM, -PH_OFFSET_PPM])
        seq_sim.set_definition('CwpeTargets_uT', PH_CWPE_LIST_UT)

    seq_sim_filename = f'{sim_output_dir}/{seq_name}_sim.seq'
    seq_sim.write(seq_sim_filename)
    print(f'[sim] wrote {seq_sim_filename} ({len(seg_names)} segments, '
          f'1 ADC sample each -> {len(seg_names)} total ADC events)')
    print(f'[sim] offsets_ppm: {sim_offsets_ppm}')
    print(f'[sim] SegmentB1_uT: {sim_b1_ut}')
    print(f'[sim] SegmentCwpe_uT: {sim_cwpe_ut}')