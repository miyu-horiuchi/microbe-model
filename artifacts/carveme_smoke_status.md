# CarveMe and gapseq External Baseline Status

## CarveMe

CarveMe was installed in an isolated `uv` environment and run on one held-out
protein FASTA generated for the GenomeSPOT smoke benchmark.

- Input: `data/external_benchmark_fastas/GCA_000820045.faa.gz`
- Output: `artifacts/carveme_smoke/GCA_000820045.xml`
- Output size: 9.4 MB
- Extra local dependency installed: `diamond 2.2.0` via Homebrew
- Command:

```bash
uv run --python 3.11 --isolated --with carveme carve \
  data/external_benchmark_fastas/GCA_000820045.faa.gz \
  -o artifacts/carveme_smoke/GCA_000820045.xml \
  --solver scip
```

Result: reconstruction smoke test passed.

Medium feasibility is not scored yet. The missing piece is not model
reconstruction; it is a fair mapping from MediaDive recipe labels to compounds or
media definitions that CarveMe can gap-fill/test. Without that mapping, a
CarveMe-vs-MediaDive hit@k number would be mostly bookkeeping noise.

## gapseq

The local conda platform check did not find `gapseq`:

```bash
conda search -c conda-forge -c bioconda gapseq --info
```

Result: `PackagesNotFoundError` on the local macOS conda channels. gapseq is also
not just a Python command: the official setup requires downloading Bacteria and
Archaea reference sequence databases before real inference.

Given the current disk state, the full gapseq run should be moved to a Linux
machine or cloud instance with substantially more free disk.
