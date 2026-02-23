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
#### 5. Download the required files from [Drive](https://drive.google.com/drive/folders/1DaNsrBZot7z07-ihFZdEv-TGTAJxvvHS?usp=drive_link) Place them into the indicated folders inside the repository.
#### 6. Add your MRI scans. Make sure the images you want to process are located inside the input folder.
#### 7. Install PyTorch following the official installation instructions (Conda or Pip).
#### 8. Move into the container folder and run Docker.
```bash
cd container
docker compose up --build
```
Once the Docker container is running, the entire pipeline executes automatically. To modify the workflow, all functionality is located in:
```bash
pipeline.py
```

## Usage Notes
#### No manual commands are needed inside the container.
#### All preprocessing and segmentation is automatic.
#### Results are saved automatically in the output directory.

# References

ROBEX
J. E. Iglesias, C.-Y. Liu, P. M. Thompson, and Z. Tu, “Robust brain extraction across datasets and comparison with publicly available methods,” IEEE Trans. Med. Imaging, 30(9):1617–1634, 2011.

ANTs
B. Avants, N. Tustison, and G. Song, “Advanced normalization tools (ANTS),” Insight Journal, 2009.

nnU-Net
F. Isensee, P. Kickingereder, et al., “nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation,” Nature Methods, 2021.

Intensity normalization
J. C. Reinhold, B. E. Dewey, A. Carass, and J. L. Prince, “Evaluating the impact of intensity normalization on MR image synthesis,” Medical Imaging 2019: Image Processing, 10949, 2019.
