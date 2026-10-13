# RSAM: multi-region MR-to-CT synthesis

PyTorch implementation of region-specific attention for 2.5D MR-to-CT synthesis, with Region-Aware Vectors (RAV) and Dynamic Gating Factors (DGF).

[Paper](https://doi.org/10.1109/TIP.2026.3658010)

This repository provides training, native-grid inference, and PNG image comparison.

## Install

Use Python 3.10 or newer. Install a matched PyTorch 2.8 / torchvision 0.23 build appropriate for your CPU or CUDA environment, then:

```bash
python -m pip install -e .
rsam --help
```

The perceptual training loss uses torchvision's pretrained VGG19. Its standard ImageNet weights are downloaded on first use and cached by PyTorch; they are not included here. Internet access or an existing local cache is required when this loss is enabled. Optional TorchIO preprocessing can be installed with `python -m pip install -e '.[torchio]'`; it is not required by the supplied configuration.

## Data

Prepare finite 3D NIfTI volumes and a CSV manifest using [examples/manifest.csv](examples/manifest.csv) as a schema. Example filenames are placeholders. Paths are absolute or relative to the CSV file.

| Column | Meaning |
| --- | --- |
| `case_id` | Unique filename-safe case identifier |
| `patient_id` | MRI patient identifier |
| `ct_patient_id` | CT patient identifier; same as MRI for paired cases |
| `region` | `brain`, `pelvis`, or `abdomen` in the supplied configuration |
| `split` | Your existing `train`, `val`, or `test` assignment |
| `paired` | `true` for spatially matched MRI/CT, otherwise `false` |
| `mr`, `ct` | NIfTI file paths; CT may be empty for inference-only rows |

All training rows require MRI and CT. Paired data must have matching shapes and physical grids. Unpaired training samples independent CT from the same region's training pool. For inference without a reference CT, use `paired=false` and leave `ct` empty. The loader checks patient and volume-path overlap between splits. This package reads your assignments and does not generate dataset splits or include any patient list.

Input data should already be registered and resampled as required by your study. The default pipeline reads native array axes; ensure consistent axial orientation before use. MRI uses foreground percentile normalization and CT uses a fixed [-1000, 2000] HU window. Three adjacent slices form each input; the middle output slice is used for volume inference.

## Train

```bash
rsam train --config configs/rsam.yaml --manifest /path/to/manifest.csv --output runs/rsam --device cuda:0
```

Outputs include `generator_mr_to_ct.pt`, `generator_ct_to_mr.pt`, `training_last.pt`, and training logs. The inference checkpoint contains model parameters, architecture and preprocessing metadata. These are generated locally and ignored by Git.

Resume from a completed epoch with the same configuration, manifest and number of processes:

```bash
rsam train --config configs/rsam.yaml --manifest /path/to/manifest.csv --output runs/rsam --resume runs/rsam/training_last.pt --device cuda:0
```

Keep the rank-state files beside `training_last.pt`. `--max-steps` is an absolute step limit; stopping inside an epoch exports inference weights but does not create a resumable checkpoint for that partial epoch.

For distributed training, choose a process count that divides `global_batch_size` (6 in the example):

```bash
torchrun --standalone --nproc_per_node=2 --module rsam.cli train --config configs/rsam.yaml --manifest /path/to/manifest.csv --output runs/rsam_ddp
```

The generator uses nine residual blocks with attention in blocks 1 and 3. Training combines adversarial, cycle, identity and VGG perceptual objectives with configurable paired reconstruction, anatomical and patch contrastive terms. RAV and DGF remain part of the generator. The configuration declares every loss coefficient. CT discriminator body cropping applies only to configured unpaired regions during training; inference returns the generator's output on the native MRI grid. Cropping requires `replay_size: 0` so each mask stays associated with its image.

## Infer and visualize

```bash
rsam infer --checkpoint runs/rsam/generator_mr_to_ct.pt --manifest /path/to/manifest.csv --split test --output outputs/predictions --device cuda:0 --batch-size 1
rsam visualize --mr /path/to/mri.nii.gz --prediction outputs/predictions/brain_test_sct.nii.gz --slice 40 --output outputs/comparison.png
```

Add `--ct /path/to/paired_ct.nii.gz` to show the registered reference and an error map. CT panels share a fixed display window; `--ct-min` and `--ct-max` control display only. Do not pass an unpaired subject as a voxelwise reference.

```bash
rsam inspect --checkpoint runs/rsam/generator_mr_to_ct.pt
```

## Citation

See [CITATION.cff](CITATION.cff) for the associated paper, *Anatomy-Aware MR-Imaging-Only Radiotherapy* (IEEE Transactions on Image Processing, 2026), DOI: [10.1109/TIP.2026.3658010](https://doi.org/10.1109/TIP.2026.3658010).
