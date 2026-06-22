# Benchmark Datasets

This directory holds evaluation datasets. Files are **not committed** to the repository — download them manually using the links below.

## LoCoMo

Multi-session conversational memory benchmark.

**Paper**: [LoCoMo: Very Long-Term Conversational Memory](https://arxiv.org/abs/2309.01550)  
**Download**: https://huggingface.co/datasets/snap-stanford/locomo

```bash
# Download the 10-session test split (small, used by scripts/bench/locomo.py)
huggingface-cli download snap-stanford/locomo locomo10.json --repo-type dataset --local-dir data/
```

Expected file: `data/locomo10.json`

**License**: CC BY 4.0

## LongMemEval

Single-session long-context memory retrieval benchmark.

**Paper**: [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)  
**Download**: https://huggingface.co/datasets/xiaowu0162/longmemeval

```bash
huggingface-cli download xiaowu0162/longmemeval --repo-type dataset --local-dir data/longmemeval/
```

Expected directory: `data/longmemeval/`

**License**: Apache 2.0
