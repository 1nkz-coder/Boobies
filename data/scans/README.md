# Put your labelled scans here

    data/scans/benign/      <- non-cancerous scans
    data/scans/malignant/   <- cancerous scans
    data/scans/normal/      <- optional third class

Any PNG / JPG / TIFF / DICOM works. Aim for at least 100 images per class;
more data = a more trustworthy model. Then run:

    python train_scan_model.py --modality MRI

Use `--polarity dark` for ultrasound (lesions are dark) and `--polarity bright`
for contrast-enhanced MRI / mammography.
