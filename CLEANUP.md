# Publication cleanup

Original archives are unchanged. Cleanup applies to the assembled publication repository only.

Each of the nine models keeps one SH selected from the latest dated submission and the code required by that launch configuration. Trace shell commands, Python local imports and package initializers, MATLAB local function calls, and literal local runtime resources. Preserve the original ANNA/STA licenses.

The previous broad model tree contained 156 files. The current tree contains 55 files; 101 files were removed from the publication tree, including 37 other SH files/dispatchers, unused training variants, standalone extraction/conversion/search/result collectors, original READMEs and unrelated helpers. Required imports remain even when their functions are not called. The only folded model-selector branch is UADAL versus cUADAL: its selected SH fixes UADAL.

Most ZIP timestamps are invalid; recency follows the latest dated submission command, not numeric run-script suffixes. See RUNNERS.md and metadata/runner-selection.json for exact choices. No additional run_latest.sh wrappers are included.

Virtual environments, installed packages, repository/IDE state, caches, bytecode, logs, checkpoints, unrelated datasets and generated outputs remain excluded. Pre-extracted arrays, labels, maps and MAT inputs are distributed separately as release assets. ResNet preparation and uploads remain on hold.

Audit files:

- metadata/source-manifest.csv: retained file provenance, original SHA256 and published SHA256.
- metadata/dependency-graph.json: traced file dependencies and retention reasons.
- metadata/removed-model-files.json: all 101 removals from the previous model tree.
- metadata/script-reference-audit.json: selected source-reference and syntax checks.
- metadata/data-manifest.csv: feature-resource provenance; existing four active package integrity checks remain valid.

Retained code has subsequently received formatting and English documentation changes only. Executable structures were compared with the original code; see CODE_STYLE.md. Cleanup does not change algorithms or establish successful training or protocol compliance.

The published data manifest covers only the four active representation packages. The complete historical manifest, including deferred ResNet resources, remains preserved locally. Deferred ResNet data are excluded from this publication.
