# Code formatting and English documentation

The nine selected model implementations retain the same 55 files: 38 Python files, 9 Bash launchers, 6 MATLAB files and 2 licenses. Formatting applies to the 53 code files; license texts remain unchanged. No old launchers or removed code have been restored.

- Python: Black, 88-character target line length, with original string quote normalization disabled. Imports remain in their original order.
- Bash: shfmt, Bash dialect and four-space indentation, without code simplification.
- MATLAB: consistent leading indentation, blank lines and comment presentation. Each non-comment code line retains its original content after trimming leading/trailing whitespace.
- Documentation: 134 Chinese or corrupted comments/docstrings were rewritten in English from their surrounding code. English file-purpose comments were added. Executable string literals, messages, paths, variable/function names and parameters remain unchanged.

Validation compares the Python AST with only docstring contents normalized, the Bash AST with only comments and source positions removed, and MATLAB non-comment code lines. All checks pass. The nine Bash launchers also pass bash -n; embedded Python parses successfully. Shebangs and Slurm directives remain unchanged. Formatting consistency and English documentation were checked without training the models.

Original ZIP archives are unchanged. In metadata/source-manifest.csv, sha256 identifies the original archive file; published_sha256 identifies its formatted publication copy. metadata/style-validation.json records each file's before/after hash and preservation check. These formatted files intentionally differ in bytes from their original sources.

This pass changes presentation and documentation only. Dataset processing, algorithm corrections, training and publication were outside this formatting pass; ResNet remains deferred.
