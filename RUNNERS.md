# Selected experiment launchers

Each model directory contains exactly one original `.sh` file. The publication tree contains 55 original files: 9 Bash launchers, 38 Python files, 6 MATLAB files and 2 supplied licenses. Other launchers, generated dispatchers and source files outside the selected dependency graph have been removed.

Most ZIP timestamps are invalid (1980-00-00). Selection follows the command in the latest dated `script_YYYYMMDDHHMMSS.sh` submission. This is the latest recorded launch configuration; the archives do not establish a complete source revision history. For Python models the referenced launcher is retained and the submission snapshot is recorded in metadata. DAOD has no separate original run.sh, so its latest submission is its sole retained shell entry, invoking `run1.m`.

| Method | Latest recorded submission | Sole retained SH | Called source scripts | Files |
|---|---|---|---|---|
| OCSVM | 2026-07-01 14:46:54 | `run.sh` | `ocsvm_baseline.py` | 2 |
| DAOD | 2026-07-01 15:13:06 | `script_20260701151306.sh` | `run1.m` | 7 |
| OSBP | 2025-12-30 17:03:36 | `train1.sh` | `train1.py` | 8 |
| SDEOTR | 2026-06-16 11:03:32 | `run.sh` | `train.py`, `train.py` | 5 |
| ANNA | 2026-06-16 11:00:23 | `run_re.sh` | `train.py` | 6 |
| BCL | 2026-06-17 11:54:10 | `run1.sh` | `train.py` | 3 |
| STA | 2026-01-13 16:20:38 | `run3.sh` | `wavstep_1.py`, `wavstep_2.py` | 7 |
| STIENet | 2026-01-08 16:20:32 | `run3.sh` | `train1.py` | 6 |
| UADAL | 2025-12-30 16:00:53 | `run2.sh` | `train1.py` | 11 |

## Dependency rules

Starting from each selected SH, trace its executed Python/MATLAB files, recursive local imports, package initialization files and local function calls. Preserve modules imported by an evaluator even when that evaluator is used only for metrics. For example, STIENet needs both `DataLoader_w.py` and `DataLoader.py` because `train1.py` imports `eval.py`, which imports `DataLoader.py`.

UADAL's selected SH fixes `MODEL="UADAL"`; its unselected cUADAL branch and `models/basenet.py` are excluded. Changing `--model` to cUADAL would require restoring that branch's files from the original archive. All other conditional imports are conservatively retained when the selected configuration cannot prove them unreachable. UADAL's image-loader helpers remain because `get_loader.py` imports them at module import time.

The selected sources preserve their executable structures; publication copies have consistent formatting and English comments/docstrings. See CODE_STYLE.md. File-level origins and SHA256: `metadata/source-manifest.csv`. Dependency edges: `metadata/dependency-graph.json`. Removed-file list: `metadata/removed-model-files.json`. Submission filenames, hashes and choices: `metadata/runner-selection.json`.

## Configuration before running

Use Bash on Linux or WSL and launch from the corresponding model directory. Replace archived `/seu_nvme/...` and `/seu_share/...` data/output paths with actual paths. DAOD's SH also contains a cluster-specific `module load matlab2024b`; configure that for your MATLAB environment.

The selected configurations differ in coverage: OCSVM uses HuBERT; DAOD uses WavLM; STA uses eGeMAPS; STIENet uses HuBERT and only M/Me directions; OSBP and UADAL use the three SSL features. SDEOTR contains its supplied multi-feature and HuBERT sections. ANNA includes ResNet/eGeMAPS; BCL selects ResNet exclusively. ResNet data remain on hold: skip ANNA's ResNet configuration and defer BCL's selected experiment. No old configuration is retained as a fallback.

Latest recorded scripts do not necessarily cover all 540 paper configurations. Review the protocol differences in REPRODUCIBILITY.md before a strict rerun.

## Validation

All 55 retained source origins were verified against the archives. The 53 formatted code copies preserve executable structures; the two licenses remain byte-identical. All 38 Python files parse successfully, all 9 Bash files pass `bash -n`, and SDEOTR's inline Python parses. The selected entry scripts and traced local dependencies are present. Full training, dependency installation and MATLAB execution were not performed.
