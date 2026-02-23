#!/usr/bin/env python3
import stat
from pathlib import Path
import subprocess

ALLOWED_EXTENSIONS = {'nii.gz'}

"""
Execute an operating system command and handle its output/errors.

Parameters:
cmd : list
List of strings representing the command to execute and its arguments.

Returns:
subprocess.CompletedProcess
Object containing the execution results, which includes:
- returncode: Return code (0 = success)
- stdout: Standard output of the command as a string
- stderr: Standard error output of the command as a string

Example:
result = run_cmd(['echo', 'Hello world'])
[CMD] echo Hello world

Hello world
"""
def run_cmd(cmd):
    print(f"\n[CMD] {' '.join(cmd)}\n")
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True)
    
    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)
        raise RuntimeError(f"Error running: {' '.join(cmd)}")
    
    print(result.stdout)
    return result


"""
Extract the file extension and check if it is in the list of allowed extensions defined in ALLOWED_EXTENSIONS.

Parameters:
file : str
Full name of the file (e.g., "document.pdf", "image.png").

Returns:
bool
True if the file extension is in ALLOWED_EXTENSIONS.
False if the extension is not allowed or the file has no extension.

Examples:
ALLOWED_EXTENSIONS = {'nii.gz'}
allowed_file("subject.nii.gz")
True
"""
def allowed_file(file):
    extension = ".".join(file.split(".")[1:])
    return extension in ALLOWED_EXTENSIONS


"""
This function verifies that the specified directory contains files and that all files have the correct extension (.nii.gz) for processing.

Parameters:
input_dir : str
Path to the directory containing the images to verify.

Returns:
The function does not return any value, it only validates the files.

Raises:
RuntimeError
1. If no files are found in the specified directory.
2. If any file does not have the .nii.gz extension.
"""
def verify_inputs(input_dir):
    print("=== Checking input images ===")

    images = sorted(Path(input_dir).glob("*"))

    if len(images) == 0:
        raise RuntimeError("No files were found in the input")

    for img in images:
        name = img.name
        
        if not allowed_file(name):
            raise RuntimeError(
                f"The file '{name}' is not valid"
                f"Must have the extension .nii.gz"
            )


"""
This function executes the ROBEX script on all .nii.gz images in the input directory, saving the results in the specified output directory.

Parameters:
input_dir : str
Directory containing the input .nii.gz images to be processed.

robex_dir : str
Directory where the ROBEX-processed images will be saved. It is created automatically if it does not exist.

Returns:
The function does not return any values; it processes the images and saves them in the output directory.
"""
def run_robex(input_dir, robex_dir):
    print("=== Running ROBEX ===")
    print("\n This may take a long time")
    Path(robex_dir).mkdir(parents=True, exist_ok=True)
    
    robex_script = Path("./ROBEX/runROBEX.sh")
    
    try:
        robex_script.chmod(robex_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception as e:
        print(f"Warning: Could not set permissions with Python: {e}") 
        pass 

    for img in Path(input_dir).glob("*.nii.gz"):
        output_img = Path(robex_dir) / img.name
        run_cmd(
            [str(robex_script), str(img.resolve()), str(output_img.resolve())] 
        )


"""
This function performs nonlinear registration of .nii.gz brain images to a reference atlas using the antsRegistrationSyN.sh script.

Parameters:
input_dir : str
Directory containing the .nii.gz images to be registered.

ants_dir : str
Directory where the ANTs registration results will be saved. It is created automatically if it does not exist.

atlas_path : str
Path to the reference atlas file (.nii.gz) to be used as the target for registration.

Returns:
The function does not return any values; it generates ANTs output files in the specified directory.
"""
def run_ants(input_dir, ants_dir, atlas_path):
    print("=== Running ANTs Registration ===")
    print("\n This may take a long time")
    Path(ants_dir).mkdir(parents=True, exist_ok=True)
    
    ants_cmd_path = "/usr/local/bin/antsRegistrationSyN.sh"

    for img_path in sorted(Path(input_dir).glob("*.nii.gz")):
        
        file_prefix = img_path.stem.split(".")[0]
        out_prefix = Path(ants_dir) / file_prefix
        
        cmd = [
            ants_cmd_path,
            "-d", "3",
            "-f", atlas_path,
            "-m", str(img_path.resolve()),
            "-o", str(out_prefix)
        ]
        
        try:
            run_cmd(cmd)
        except RuntimeError as e:
            print(f"Warning: ANTs failed for {img_path.name}. Continuing...")
            print(e)
            
    print("=== ANTs Registration completed ===")


"""
This function searches for images processed by ANTs (*Warped.nii.gz files) and renames them with the '_0000.nii.gz' suffix required by the nnUNet framework for segmentation.

Parameters:
ants_dir : str
Directory containing the output files generated by ANTs. Must include files matching the pattern *Warped.nii.gz.

Returns:
list
List of Path objects containing the paths of the renamed images. Each element is the full path to a renamed file.
"""
def rename_after_ants(ants_dir):
    print("=== Renaming registered images for nnUNet ===")

    registered_images = []
    for img in Path(ants_dir).glob("*Warped.nii.gz"):
        if not img.name.endswith("InverseWarped.nii.gz"):
            registered_images.append(img)
    
    registered_images = sorted(registered_images)
    
    if len(registered_images) == 0:
        print("No *Warped.nii.gz files were found")
        all_files = list(Path(ants_dir).glob("*.nii.gz"))
        print(f".nii.gz files found in {ants_dir}:")
        for f in all_files:
            print(f"  - {f.name}")
        raise RuntimeError("ANTs did not generate appropriate Warped.nii.gz images")

    out_list = []
    for img in registered_images:
        if "Warped.nii.gz" in img.name:
            base = img.name.replace("Warped.nii.gz", "")  
            new_name = f"{base}_0000.nii.gz"       
            new_path = Path(ants_dir) / new_name
            
            print(f"Renaming: {img.name} → {new_name}")
            run_cmd(["mv", str(img), str(new_path)])
            out_list.append(new_path)
    
    print(f"{len(out_list)} images were renamed for nnUNet")
    return out_list


"""
This function executes the nnUNet prediction command on the previously processed and renamed images in the ANTs directory, generating segmentation masks in the predictions directory.

Parameters:
ants_dir : str
Directory containing the ANTs-processed and renamed images.

pred_dir : str
Directory where the nnUNet-generated predictions/segmentations will be saved. It is created automatically if it does not exist.

dataset_id : int
Numeric ID of the nnUNet dataset to use for prediction.

configuration : str
Name of the nnUNet model configuration to use: "3d_fullres".

Returns:
The function does not return any values; it generates prediction files in the specified directory.
"""
def run_nnunet(ants_dir, pred_dir, dataset_id, configuration):
    print("=== Running nnUNetv2_predict ===")
    print("\n This may take a long time")
    
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", ants_dir,
        "-o", pred_dir,
        "-d", str(dataset_id), 
        "-c", configuration,
        "-f", "all"
    ]
    run_cmd(cmd)
    

"""
This function removes temporary and transformation files generated by ANTs during the registration process, keeping only the final registered images.

Parameters:
ants_dir : str
Directory containing the files generated by ANTs.

keep_registered : bool, optional
If True (default), keeps the final registered images (files ending with '_0000.nii.gz').
If False, deletes ALL files in the directory.

Returns:
The function does not return any values; it removes files from the filesystem.
"""    
def cleanup_intermediate(ants_dir, keep_registered=True):
    print("\n === Cleaning intermediate files ===")

    for f in Path(ants_dir).glob("*"):
        if keep_registered and f.name.endswith("_0000.nii.gz"):
            print(f"Keeping final registered image: {f}")
            continue

        print(f"Removing ANTs: {f}")
        f.unlink()


"""
This function initiates the entire workflow from preprocessing to prediction.

Parameters:
input_dir : str
Directory containing the input brain images.

atlas_path : str
Full path to the reference atlas file (.nii.gz) to be used as the target for nonlinear registration.

out_root : str
Root directory where all output subdirectories will be created.

dataset_id : int
Numeric ID of the configured nnUNet dataset (15).

config : str
nnUNet model configuration to use for segmentation: "3d_fullres".

Returns:
The function does not return any values; it executes the complete pipeline and generates results in the output directories.
"""
def run_process(input_dir, atlas_path, out_root, dataset_id, config):
    robex_dir   = f"{out_root}/ROBEX"
    ants_dir    = f"{out_root}/ants"
    pred_dir    = f"{out_root}/predictions"

    print("=== PIPELINE STARTED ===")

    run_robex(input_dir, robex_dir)
    run_ants(robex_dir, ants_dir, atlas_path)
    rename_after_ants(ants_dir)
    cleanup_intermediate(ants_dir, keep_registered=True)
    run_nnunet(ants_dir, pred_dir, dataset_id, config)

    print("\n=== PIPELINE SUCCESSFULLY COMPLETED ===")


"""
This function defines the fixed parameters of the pipeline and executes the complete brain image processing workflow.
"""
def main():
    input_dir  = "/workspace/input"
    atlas_path = "/workspace/data/atlas/SRI24_atlas.nii"
    out_root   = "/workspace/output"
    dataset_id = 15
    config     = "3d_fullres"

    run_process(
        input_dir   = input_dir,
        atlas_path  = atlas_path,
        out_root    = out_root,
        dataset_id  = dataset_id,
        config      = config
    )


if __name__ == "__main__":
    main()

