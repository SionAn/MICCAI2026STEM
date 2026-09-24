# Data preprocessing

All scripts write into one processed root (`$DATA_DIR`), which is later passed to training as `--data_dir`.
The final layout read by the training code is

```
$DATA_DIR/<Name>/sub/<subject>/<class>/<trial>.pkl            # raw windows, used to build the index
$DATA_DIR/<Name>/sorted/<subject>/<class>/<trial>.pkl         # + channel sorting and 3D coordinates
$DATA_DIR/<Name>/sorted_scaled/<subject>/<class>/<trial>.pkl  # + amplitude scaling (used for training)
$DATA_DIR/<Name>.pkl                                          # cached index {Name: {subject: {class: [trials]}}}
```

Unlabeled recordings use the class folder `Other`. Every sample is a 4-second window at 200 Hz.

## Pipeline

Run the scripts from this folder (they import `common.py`, `dataset_index.py` and `edf_.py`).

**1. Per-dataset extraction**

```bash
DATA_DIR=/path/to/processed

python preprocess_bci_iv_1.py   --raw_dir /path/to/BCI_IV_1          --out_dir $DATA_DIR
for d in BNCI2014_002 BNCI2015_001 BNCI2015_004 Liu2024 Schirrmeister2017 Weibo2014 Zhou2016; do
    python preprocess_moabb.py  --dataset $d                          --out_dir $DATA_DIR
done
python preprocess_emobrain.py   --raw_dir /path/to/enterface06_EMOBRAIN/Data/EEG --out_dir $DATA_DIR
python preprocess_seed.py       --raw_dir /path/to/SEED/SEED_EEG/SEED_RAW_EEG    --out_dir $DATA_DIR
python preprocess_seed_iv.py    --raw_dir /path/to/SEED_IV/eeg_raw_data          --out_dir $DATA_DIR
python preprocess_seed_fra_ger.py --dataset SEED-FRA --raw_dir /path/to/SEED_FRA/French/01-EEG-raw --out_dir $DATA_DIR
python preprocess_seed_fra_ger.py --dataset SEED-GER --raw_dir /path/to/SEED_GER/German/01-EEG-raw --out_dir $DATA_DIR
python preprocess_siena.py      --raw_dir /path/to/SienaScalpDatabase --out_dir $DATA_DIR
python preprocess_gal.py        --raw_dir /path/to/WAY-EEG-GAL        --out_dir $DATA_DIR
python preprocess_tueg.py       --raw_dir /path/to/tuh_eeg/v2.0.1/edf --out_dir $DATA_DIR
python preprocess_spis.py       --raw_dir /path/to/SPIS               --out_dir $DATA_DIR
python preprocess_raw_rest.py   --dataset Raw  --raw_dir /path/to/RawEEGData          --out_dir $DATA_DIR
python preprocess_raw_rest.py   --dataset Rest --raw_dir /path/to/RestingStateEEGData --out_dir $DATA_DIR

python preprocess_physionet.py  --raw_dir /path/to/eegmmidb/1.0.0 --out_dir $DATA_DIR
python preprocess_shu.py        --raw_dir /path/to/SHU-MI/mat     --out_dir $DATA_DIR
python preprocess_isruc.py      --raw_dir /path/to/ISRUC/subgroup1 --out_dir $DATA_DIR
```

**2. Regroup into subject/class folders** (only for scripts that write flat files:
BCI-IV-1, EmoBrain, GAL, Siena, TUEG, SPIS, Raw, Rest, SEED-IV, SEED-FRA, SEED-GER)

```bash
python regroup_subjects.py --data_dir $DATA_DIR \
    --datasets BCI-IV-1 EmoBrain GAL Siena TUEG SPIS Raw Rest SEED-IV SEED-FRA SEED-GER
```

**3. Channel sorting and electrode coordinates** (all datasets)

```bash
ALL="BCI-IV-1 BNCI2014_002 BNCI2015_001 BNCI2015_004 Liu2024 Schirrmeister2017 Weibo2014 Zhou2016 \
     EmoBrain SEED SEED-IV SEED-FRA SEED-GER Siena GAL TUEG SPIS Raw Rest PhysioNet SHU ISRUC"
python assign_channel_locations.py --data_dir $DATA_DIR --datasets $ALL
```

**4. Amplitude scaling** (all datasets)

```bash
python scale.py --data_dir $DATA_DIR --datasets $ALL
```

The index `$DATA_DIR/<Name>.pkl` is created in step 3 and reused afterwards; delete it if you re-run step 1 or 2.
