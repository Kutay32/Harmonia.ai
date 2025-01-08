import os
import requests
import pydicom
import nibabel as nib
import numpy as np

# MONAILabel server URL
MONAI_SERVER_URL = "http://localhost:8000"

# Path to your DICOM folder (relative to the script)
DICOM_FOLDER = os.path.join("..", "data", "dicom")  # Adjusted for the folder structure

# Output folder for saving masks (relative to the script)
OUTPUT_FOLDER = os.path.join("..", "data", "masks")  # Adjusted for the folder structure
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_dicom(dicom_path):
    """Load a DICOM file and return its pixel array."""
    dicom = pydicom.dcmread(dicom_path)
    return dicom.pixel_array

def save_mask_as_nifti(mask, output_path):
    """Save a mask as a NIfTI file."""
    mask_nifti = nib.Nifti1Image(mask, affine=np.eye(4))
    nib.save(mask_nifti, output_path)

def generate_mask(dicom_path):
    """Send a DICOM file to the MONAILabel server and generate a mask."""
    try:
        with open(dicom_path, "rb") as f:
            # Correct file upload format
            files = {"file": (os.path.basename(dicom_path), f, "application/dicom")}
            response = requests.post(
                f"{MONAI_SERVER_URL}/infer",
                files=files,
                data={"model": "segmentation_unetpp"},  # Ensure this matches your model name
            )
        if response.status_code == 200:
            return response.json()["mask"]
        else:
            print(f"Error generating mask for {dicom_path}: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception while generating mask for {dicom_path}: {str(e)}")
        return None

def process_dicom_folder(dicom_folder, output_folder):
    """Process all DICOM files in a folder and generate masks."""
    for root, _, files in os.walk(dicom_folder):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                print(f"Processing: {dicom_path}")

                # Generate mask
                mask = generate_mask(dicom_path)
                if mask is not None:
                    # Save mask as NIfTI
                    output_path = os.path.join(
                        output_folder,
                        os.path.splitext(file)[0] + "_mask.nii.gz"
                    )
                    save_mask_as_nifti(mask, output_path)
                    print(f"Mask saved to: {output_path}")

# Run the pipeline
process_dicom_folder(DICOM_FOLDER, OUTPUT_FOLDER)