# Reproducibility audit

The manuscript defines a uniform protocol. The supplied archives are historical implementations, and several checks show that they cannot yet be described as a verified end-to-end reproduction of that protocol.

1. **All nine source implementations are present.** OSBP and SDEOTR were supplied in additional archives. Their source replaces the earlier weights-only/missing-source status.
2. **Target-label model selection:** STIENet `train1.py`, OSBP `train1.py`, and SDEOTR `train.py` explicitly track or save the best target HOS. This contradicts the paper's final-evaluation-only label constraint. Supplied algorithms are preserved; a strict rerun must define a source-only or label-free checkpoint criterion before training. Results selected by target HOS must not be called strict unsupervised protocol results.
3. **Label indices:** ResNet maps emotions separately for each corpus. For example, CREMA-D has happy=3, sad=5, angry=0. Most array baselines assume known indices 0–2. Use `scripts/prepare_features.py` on a separate copy; raw ResNet arrays cannot be passed directly to these baselines under that assumption.
4. **Corpus counts:** array ESD has 35,000 samples for eGeMAPS and all three SSL features, but 17,500 for ResNet. MSP array counts are 15,730 versus 14,799 for ResNet. The manuscript cites English-only ESD and 8,438 MSP utterances. Without stable utterance IDs, language annotations and split indices, duplicated/augmented/bilingual coverage cannot be determined. No samples are silently dropped or presumed English-only.
5. **MAT versus NPY:** both supplied formats are retained for their respective loaders. They are not assumed to have identical ordering or label semantics. BCL uses `h5py`, so standard MATLAB v5 files need an explicit HDF5 conversion before that loader can read them. DAOD supports several MAT/HDF5 field conventions.
6. **Architecture constraints:** OSBP defaults to 1024-dimensional features, the retained STIENet array entry uses the 1024-dimensional branch, and BCL/UADAL require CUDA in retained code. Shape or hardware compatibility is not fixed by renaming a feature directory. Other feature dimensions require architecture review and must be documented as an implementation change.
7. **Article tables:** 540 appendix rows and the paper's separate summary table are published as supplied. Thirty-six displayed HOS values fail a >0.05-point formula check. Per-run aggregation could explain some differences, but the supplied article does not provide a reconciliation. Do not silently recompute or claim the summary table has been validated from raw predictions.
8. **Splits and training:** no verified cross-representation sample IDs, fixed split manifests, original package lockfile, or complete 540-run execution logs were established. This preparation performs integrity, syntax, dependency and small utility checks; it does not execute large-scale training or establish paper-level accuracy.

`metadata/data-audit.json` records actual shapes, label counts and maps. The 20 array datasets have matching feature/label row counts. Original feature files, sample order and source behavior are preserved in the release packages.

## ResNet hold

ResNet is deferred at the user's request. No further conversion is being performed and neither ResNet package is in the active upload manifest. Its supplied MSP NPY has 14,799 rows, while the supplied MAT has 18,768 rows. Original archive files remain unchanged. WavLM MELD has 12,921 rows, compared with 12,923 in the other arrays.

## Selected launch scope

Only the latest recorded launcher and its dependencies are retained per method. Their feature/task coverage differs; see RUNNERS.md. Omitted old configurations can be recovered from the untouched original archives. This cleanup does not resolve the protocol differences above.
