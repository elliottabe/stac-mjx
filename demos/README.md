# STAC Pipeline Demos

This directory contains scripts for running the STAC inverse kinematics pipeline on preprocessed keypoint data.

## Main Script: run_stac_fly_model.py

The primary entry point for running STAC IK on fruit fly keypoint data.

### Quick Start

```bash
# Test your configuration first
python ../test_stac_configs.py paths=workstation dataset=

# Run STAC on a single version
python run_stac_fly_model.py paths=workstation dataset=

# Run STAC across multiple versions (multirun)
python run_stac_fly_model.py -m \
  paths=workstation \
  dataset= \
  version=Predictions_3D_20260114-145343,Predictions_3D_20260202-171900
```

### Prerequisites

1. **Preprocessed keypoint data** - Must run preprocessing first:
   ```bash
   cd /home/eabe/Research/MyRepos/3d_tracking_dataset
   python scripts/preprocess_keypoints_for_ik.py \
     paths=workstation \
     dataset= \
     version=Predictions_3D_20260114-145343
   ```
   This creates: `preprocessed_bout.h5` in the version directory

2. **MuJoCo body model** - Ensure body models exist:
   - Located in: `/home/eabe/Research/MyRepos/fruitfly_body_models/`
   - V1: `fruitfly_v1/fruitfly_v1_free.xml`
   - V2: `fruitfly_v2.1/fruitfly_v2.1.xml`

### Configuration

Controlled via Hydra config groups:

- **paths**: Machine-specific paths (`workstation`, `hyak`, `desktop`)
- **dataset**: Dataset + version (``, `courtship`)
- **anatomy**: Body model version (`v1`, `v2`)
- **stac**: Pipeline settings (`stac_fly_free`, `stac_fly_free_v2`)
- **model**: MuJoCo solver params (`fly_free`, `fly_free_v2`)

### Output

STAC generates:
- `Fruitfly_fit_V1_free.h5` - Fit with offset optimization
- `Fruitfly_ik_V1_free.h5` - IK-only (no offset fitting)
- `{dataset}_{anatomy}.mp4` - Visualization video (1000 frames)

Saved to: `{data_dir}/` (same directory as input data)

### Examples

**Different versions:**
```bash
# Specify version explicitly
python run_stac_fly_model.py \
  paths=workstation \
  dataset= \
  version=Predictions_3D_20260203-103416
```

**Different anatomy:**
```bash
# Use V2 model
python run_stac_fly_model.py \
  paths=workstation \
  dataset= \
  anatomy=v2 \
  stac=stac_fly_free_v2
```

**Courtship data:**
```bash
python run_stac_fly_model.py \
  paths=workstation \
  dataset=courtship
```

**On cluster (Hyak):**
```bash
python run_stac_fly_model.py \
  paths=hyak \
  dataset= \
  version=Predictions_3D_20260114-145343
```

**Multirun across multiple configs:**
```bash
# Multiple versions
python run_stac_fly_model.py -m \
  paths=workstation \
  dataset= \
  version=Predictions_3D_20260114-145343,Predictions_3D_20260202-171900,Predictions_3D_20260203-103416

# Multiple anatomies
python run_stac_fly_model.py -m \
  paths=workstation \
  dataset= \
  anatomy=v1,v2

# Combination
python run_stac_fly_model.py -m \
  paths=workstation \
  dataset= \
  version=Predictions_3D_20260114-145343,Predictions_3D_20260203-103416 \
  anatomy=v1,v2
```

### Troubleshooting

**FileNotFoundError: preprocessed_bout.h5**
- Run preprocessing first (see Prerequisites)
- Check that version directory exists
- Verify `dataset.preprocessing.input_filename` matches actual filename

**Missing XML model**
- Check `body_model_dir` path in paths config
- Ensure anatomy name matches directory: `fruitfly_v1`, `fruitfly_v2.1`

**Out of memory**
- Reduce `n_frames_per_clip` in dataset config
- Set `enable_padding: False` to process variable-length clips

**View resolved configuration:**
```bash
python run_stac_fly_model.py paths=workstation dataset= --cfg job
```

### Development

To modify STAC behavior:

1. **Dataset parameters** - Edit `configs/dataset/{dataset_name}.yaml`
2. **Pipeline settings** - Edit `configs/stac/stac_fly_{variant}.yaml`
3. **Model parameters** - Edit `configs/model/fly_{variant}.yaml`
4. **Path templates** - Edit `configs/paths/{machine}.yaml`

All configs use Hydra interpolations for automatic path resolution.

## See Also

- **[STAC_MULTIRUN_GUIDE.md](../STAC_MULTIRUN_GUIDE.md)** - Complete multirun documentation
- **[test_stac_configs.py](../test_stac_configs.py)** - Test configuration system
- **[3d_tracking_dataset/.github/copilot-instructions.md](../../3d_tracking_dataset/.github/copilot-instructions.md)** - Full pipeline documentation
