# Overnight 8xH100 Epiplex Recipe

Use this when launching the single-node overnight LM run on an 8xH100 machine.

Primary entrypoint:

```bash
cd /Users/jjmayo/projects/demo_day/arc-agi
RUN_ROOT=/local_nvme/arc_agi_overnight_$(date +%Y%m%d_%H%M%S) \
bash /Users/jjmayo/projects/demo_day/arc-agi/scripts/run_epiplex_generic_8h100_overnight.sh
```

What it does:

1. Builds the final DCLM+OSCAR manifest.
2. Fits a `generic` Epiplex tokenizer.
3. Packs a `seq_len=2048` corpus.
4. Validates the packed manifest.
5. Verifies or creates the Hugging Face checkpoint repo.
6. Launches the 7B single-node FSDP run with a wall-clock guard.

Important defaults:

- `DCLM_MAX_DOCUMENTS=1000000`
- `TOKENIZER_SAMPLE_MAX_DOCUMENTS=50000`
- `SEQ_LEN=2048`
- `TOKENIZER_TASK=generic`
- `TRAIN_TIMEOUT_HOURS=36`
- 7B preset from `scripts/launch_pretraining_lm_8h100_7b.sh`

Useful overrides:

- `RUN_ROOT=/local_nvme/...`
- `TRAIN_TIMEOUT_HOURS=0` to disable the timeout wrapper
- `REBUILD_MANIFEST=1`
- `REFIT_TOKENIZER=1`
- `REPACK_CORPUS=1`
- `TRAIN_STEPS=20000`

For checkpoint uploads, put `HF_UPLOAD_*` plus `HF_TOKEN` in
`config/runtime/hf_upload.env.local`. The recipe sources that file before the
HF repo preflight and before the training launch. If `HF_UPLOAD_PATH_PREFIX`
is not set, the recipe defaults it to `runs/<basename of RUN_ROOT>`.

The DCLM corpus source used by this recipe is the Hugging Face parquet mirror
`mlfoundations/dclm-baseline-1.0-parquet`, with provenance recorded back to the
original [datacomplm/DCLM](https://github.com/datacomplm/DCLM) submission repo
and the maintained [mlfoundations/dclm](https://github.com/mlfoundations/dclm)
upstream.

Packed-manifest validation is performed by:

```bash
/Users/jjmayo/projects/demo_day/arc-agi/scripts/validate_packed_pretraining_manifest.py
```

The validation gate requires:

- tokenizer kind `epiplex`
- tokenizer task `generic`
- `seq_len=2048`
- train corpus includes `dclm`
- train corpus includes at least one `oscar*` family
- train sequence count above `100000`
