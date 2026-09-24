"""PhysioNet EEG Motor Movement/Imagery dataset (downstream, 4-class motor imagery).

Writes <out_dir>/PhysioNet/sub/<subject>/<label>/*.pkl directly.
"""
import argparse
import os
import pickle

import mne

TASKS = ['04', '06', '08', '10', '12', '14']  # motor imagery runs

SELECTED_CHANNELS = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                     'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                     'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                     'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                     'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                     'O1..', 'Oz..', 'O2..', 'Iz..']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='eegmmidb/1.0.0 folder (contains S001/ ... S109/)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.raw_dir) if os.path.isdir(os.path.join(args.raw_dir, f))])
    chs = [ch.replace('..', '').replace('.', '').upper() for ch in SELECTED_CHANNELS]

    for file in files:
        for task in TASKS:
            raw = mne.io.read_raw_edf(os.path.join(args.raw_dir, file, f'{file}R{task}.edf'), preload=True)
            raw.pick_channels(SELECTED_CHANNELS, ordered=True)
            if len(raw.info['bads']) > 0:
                raw.interpolate_bads()
            raw.set_eeg_reference(ref_channels='average')
            raw.filter(l_freq=0.1, h_freq=None)
            raw.notch_filter(60)
            raw.resample(200)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw,
                                events_from_annot,
                                event_dict,
                                tmin=0,
                                tmax=4. - 1.0 / raw.info['sfreq'],
                                baseline=None,
                                preload=True)
            data = epochs.get_data(units='uV')[:, :, -800:]
            events = epochs.events[:, 2]
            count = 0
            for i, (sample, event) in enumerate(zip(data, events)):
                # event 1 (T0) is rest; runs 04/08/12 are left/right fist, 06/10/14 are both fists/feet
                if event != 1:
                    data_dict = {
                        'X': sample,
                        'Y': event - 2 if task in ['04', '08', '12'] else event,
                        'ch_names': chs
                    }
                    dump_path = os.path.join(args.out_dir, 'PhysioNet', 'sub', file, str(data_dict['Y']))
                    os.makedirs(dump_path, exist_ok=True)
                    pickle.dump(data_dict, open(os.path.join(dump_path, f'PhysioNet_{file}R{task}_{count}.pkl'), 'wb'))
                    count += 1
                    print(f'{file}-{task}: {i + 1}/{data.shape[0]}', end='\r')
