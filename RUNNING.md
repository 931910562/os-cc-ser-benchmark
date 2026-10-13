# Running the selected launchers

Install requirements.txt in an environment with suitable PyTorch/CUDA versions. Edit the selected SH's original data/output paths before running from its model directory. Inspect its feature dimensions and task coverage in RUNNERS.md. The commands below identify the sole retained launcher for each method; they were syntax-checked, not used to train models.

| Model directory | Launch command |
|---|---|
| models/OCSVM | `bash run.sh` |
| models/DAOD | `bash script_20260701151306.sh` |
| models/OSBP | `bash train1.sh` |
| models/SDEOTR | `bash run.sh` |
| models/ANNA | `bash run_re.sh` |
| models/BCL | `bash run1.sh` |
| models/STA | `bash run3.sh` |
| models/STIENet | `bash run3.sh` |
| models/UADAL | `bash run2.sh` |

Most models consume features.npy and labels.npy. STIENet's selected array branch also consumes label_map.json. DAOD uses MAT inputs and MATLAB Statistics and Machine Learning Toolbox functions such as fitclinear; configure feature_sets in run1.m and its SH's cluster module command. BCL uses h5py MAT inputs and CUDA, with its latest SH configured for ResNet.

STA runs wavstep_1.py followed by wavstep_2.py. The first stage generates discriminator_a.pkl at the SH's checkpoint path; the second stage consumes it. Its selected NPY loader loads arrays from the roots. The list flags are required CLI arguments but are not read by that NPY loader; their archived names remain in the unchanged SH.

UADAL's selected SH fixes MODEL=UADAL. Its imported loader helpers remain necessary at import time. cUADAL files are excluded; changing to that model requires restoring its dependencies from the input archive.

ResNet data are deferred. ANNA includes a ResNet entry that should be skipped while it is on hold; BCL's selected configuration cannot be run with the current four active data packages. Its old non-ResNet launchers were removed as requested.

Check semantic labels, feature dimensions, MAT formats, source known-class filtering and target-label checkpoint selection against REPRODUCIBILITY.md. The latest recorded configurations have varying feature/task coverage and do not by themselves execute all paper experiments.
