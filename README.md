# Automated Brain Lesion Segmentation Pipeline (TBI)

This repository provides a **complete neuroimaging pipeline** for automated **brain lesion segmentation** using **ROBEX**, **ANTs**, and **nnU-Net**.  
It is specifically designed for **closed traumatic brain injury (TBI)** lesion segmentation on **T1-weighted MRI scans**.

The pipeline performs:

- Skull stripping  
- Atlas-based registration  
- Intensity normalization  
- Deep learning segmentation with nnU-Net  

Once the Docker container is executed, the entire processing pipeline runs automatically.

This project provides:
- Automated brain extraction (ROBEX)
- Spatial normalization using ANTs
- Intensity normalization methods
- Lesion segmentation using nnU-Net
- A fully reproducible Docker environment

No manual commands are required inside the container.  
All processing is executed automatically when Docker starts.

---

## Fine-tuning for small lesions (Alejandro Guerrero Cano, Universidad del Valle)

This repository extends the TBI segmentation pipeline and model by Jenifer Castaño Ledesma
([CastanoJenifer/TCEC-Segmentation](https://github.com/CastanoJenifer/TCEC-Segmentation)).
Starting from her trained nnU-Net, the model was fine-tuned to improve the detection of small lesions
(< 100 mm³) by changing the loss function and the patch sampling strategy:

- **ADL v1**: patch-weighted Dice loss that gives more weight to patches with a smaller lesion fraction
  (`AdaptiveDiceLoss` in `src/nnunetv2/training/loss/dice.py`).
- **ADL v2**: volume-aware Dice loss (Chartrand et al., 2022), where each voxel is weighted by the volume
  of the lesion it belongs to (`AdaptiveDiceLoss_v2` in `src/nnunetv2/training/loss/dice.py`).
- **ADS**: volume-aware patch sampling, so that every lesion has the same probability of being sampled
  regardless of its size (`src/nnunetv2/training/dataloading/data_loader.py`).
- **Per-lesion evaluation**: detection rate by lesion size (small < 100 mm³, medium 100–1000 mm³,
  large > 1000 mm³), Dice of detected lesions and false positives
  (`src/nnunetv2/evaluation/evaluate_predictions.py`).

The CE + ADL loss is defined in `src/nnunetv2/training/loss/compound_losses.py`, and the ADL v2 + ADS
trainer in `src/nnunetv2/training/nnUNetTrainer/variants/sampling/nnUNetTrainer_ADL_v2_ADS.py`.
The selected model was ADL v2 + ADS with an initial learning rate of 1e-3 (`fold_all`).

> **Note:** the fine-tuning used an initial learning rate of 1e-3, while the trainer keeps nnU-Net's default of 1e-2.
> The learning rate is read from the `nnUNet_initial_lr` environment variable and the number of epochs from
> `nnUNet_num_epochs` (200 by default).

---

## Repository structure

```
TCEC-Segmentation/
├── container/                  ← Docker configuration (Dockerfile, docker-compose.yml)
├── data/
│   ├── atlas/                  ← place SRI24_atlas.nii here
│   ├── nnUNet_raw/             ← training data (only for training, not included)
│   ├── nnUNet_preprocessed/    ← created by nnU-Net preprocessing (only for training, not included)
│   └── nnUNet_results/
│       └── Dataset015_TCEC/
│           └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│               ├── dataset.json
│               ├── dataset_fingerprint.json
│               ├── plans.json
│               └── fold_all/   ← place checkpoint_final.pth here
├── evaluation/
│   └── labels_input/           ← reference masks (only for evaluation)
├── input/                      ← raw T1 images to process
├── output/                     ← ROBEX/, ants/ (registered images) and predictions/
├── ROBEX/                      ← place the downloaded ROBEX folder here
├── src/nnunetv2/               ← nnU-Net with the ADL v1, ADL v2 and ADS modifications
├── pipeline.py                 ← processing workflow
├── run_pipeline.sh
└── requirements.txt
```

---

## Instructions

#### 1. Clone the repository.

```bash
git clone <this-repository>
```
#### 2. Download [ROBEX](https://www.nitrc.org/projects/robex) for Linux.
#### 3. Move the ROBEX folder. Place the downloaded ROBEX folder inside the cloned repository.
#### 4. Give execution permissions.
```bash
chmod +x ./ROBEX/ROBEX
```
#### 5. Request the required files. The model weights and the other files needed to run the pipeline are not included in this repository. To obtain them, contact Alejandro Guerrero Cano at alejandro.cano@correounivalle.edu.co. Place `checkpoint_final.pth` in `data/nnUNet_results/Dataset015_TCEC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/` and `SRI24_atlas.nii` in `data/atlas/` (see [Repository structure](#repository-structure)).
#### 6. Add your MRI scans. Place the raw T1 images in `input/` in `.nii.gz` format, named `TCEC_XXXX.nii.gz`, where `XXXX` is the case number (e.g. `TCEC_0001.nii.gz`). Do not add the `_0000` suffix: the pipeline adds it after the registration, as required by nnU-Net. The pipeline skull-strips the images (ROBEX), registers them to SRI24 (ANTs) and segments them; the predicted masks are saved in `output/predictions/` as `TCEC_XXXX.nii.gz`. If you want to evaluate the results, the reference masks must also be named `TCEC_XXXX.nii.gz`, with the same case number as their image and without `_0000` (e.g. the mask of `TCEC_0001.nii.gz` is also `TCEC_0001.nii.gz`), and placed in `evaluation/labels_input/`

> **Note:** if nnU-Net does not find the images, or if the images are already skull-stripped and registered to SRI24
> and are sent directly to nnU-Net (`nnUNetv2_predict`) without ROBEX and ANTs, name them `TCEC_XXXX_0000.nii.gz`
> (e.g. `TCEC_0016_0000.nii.gz`), as required by nnU-Net. The `_0000` suffix only goes in the image name; the
> reference masks keep the name `TCEC_XXXX.nii.gz` (e.g. `TCEC_0016.nii.gz`).
#### 7. Install PyTorch following the official installation instructions (Conda or Pip). The container uses the GPU (`gpus: all` in `docker-compose.yml`), so an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) are required.
#### 8. Move into the container folder and run Docker.
```bash
cd container
docker compose up --build
```
Once the Docker container is running, the entire pipeline executes automatically. To modify the workflow, all functionality is located in:
```bash
pipeline.py
```

# References

ROBEX
J. E. Iglesias, C.-Y. Liu, P. M. Thompson, and Z. Tu, “Robust brain extraction across datasets and comparison with publicly available methods,” IEEE Trans. Med. Imaging, 30(9):1617–1634, 2011.

ANTs
B. Avants, N. Tustison, and G. Song, “Advanced normalization tools (ANTS),” Insight Journal, 2009.

nnU-Net
F. Isensee, P. Kickingereder, et al., “nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation,” Nature Methods, 2021.

Intensity normalization
J. C. Reinhold, B. E. Dewey, A. Carass, and J. L. Prince, “Evaluating the impact of intensity normalization on MR image synthesis,” Medical Imaging 2019: Image Processing, 10949, 2019.

Base pipeline and model
J. Castaño Ledesma, “Segmentación mediante Inteligencia Artificial para la detección de lesiones por Trauma Craneoencefálico Cerrado a través de Imágenes por Resonancia Magnética,” Trabajo de grado, Universidad del Valle, 2025. https://hdl.handle.net/10893/39426

ADL v2 and ADS (volume-aware Dice loss and sampling)
G. Chartrand, R. D. Emiliani, S. A. Pawlowski, et al., “Automated Detection of Brain Metastases on T1-Weighted MRI Using a Convolutional Neural Network: Impact of Volume Aware Loss and Sampling Strategy,” Journal of Magnetic Resonance Imaging, 56(6):1885–1898, 2022. https://doi.org/10.1002/jmri.28274

ADL and ADS naming in nnU-Net
Y. Yoo, E. Gibson, G. Zhao, et al., “Extended nnU-Net for Brain Metastasis Detection and Segmentation in Contrast-Enhanced Magnetic Resonance Imaging With a Large Multi-Institutional Data Set,” International Journal of Radiation Oncology, Biology, Physics, 121(1):241–249, 2025. https://doi.org/10.1016/j.ijrobp.2024.07.2318

ADL v1 (weighting idea)
C. H. Sudre, W. Li, T. Vercauteren, S. Ourselin, and M. J. Cardoso, “Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations,” in Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (DLMIA/ML-CDS 2017), LNCS 10553, pp. 240–248, Springer, 2017. https://doi.org/10.1007/978-3-319-67558-9_28

Dice loss 
F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation,” in 2016 Fourth International Conference on 3D Vision (3DV), IEEE, pp. 565–571, 2016. https://doi.org/10.1109/3DV.2016.79