
Set up the environment:

Install dependencies:

bash

pip install -r requirements.txt

Start the MONAILabel server:

bash

monailabel start_server --app /path/to/monailabel/sample-apps/segmentation --studies /path/to/dicom_folder

Generate masks:

Run the generate_masks.py script:

bash


python scripts/generate_masks.py
Train U-Net++:

Run the train_unetpp.py script:

bash

python scripts/train_unetpp.py
