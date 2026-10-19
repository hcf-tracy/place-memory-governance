# Memory, Know Your PLACE

**Implementation of _Memory, Know Your PLACE: Governing Lifecycle State and Query-Time Eligibility in Persistent-Memory Agents_.**

[Research project page](https://hcf-tracy.github.io/memory-know-your-place/)

Persistent memory can fail both by maintaining an incorrect state and by allowing ineligible information to influence a response. **PLACE** separates these concerns into **Certified Semantic State (CSS)**, which governs write-side lifecycle consistency, and **Query-Authorized Semantic Closure (QASC)**, which governs read-side eligibility. The implementation supports the four conditions of a 2 × 2 factorial design: Native, CSS, QASC, and PLACE.

## Repository scope

This repository provides source code for the framework, the main experimental runtime, and the **strict-general v3.2** evaluation implementation. It is a **code-only release**: benchmark text, trajectories, event/query annotations, oracle gold, causal-pair cases, model traces, and recorded result files are not distributed here.

The repository does **not** claim that all figures and tables on the project page can be reproduced from a public checkout alone. Experiments require the separately maintained evaluation inputs, and fresh LLM runs may differ across environments.

## Structure

| Path | Purpose |
| --- | --- |
| `hf13/` | PLACE, CSS/QASC governance, Native and baseline execution, evaluation, and strict-general scoring utilities |
| `lgcm/` | Minimal memory-backend adapter dependencies |
| `configs/` | Qwen2.5 3B and 7B model configuration |
| `run_hf13_*.sh` | Main runtime stages for the factorial design and targeted diagnostics |
| `scripts/` | Strict-general metric scoring and auxiliary helpers |
| `server/` | Local runtime setup and model/server utilities |
| `tests/` | Data-independent strict-general evaluator checks |

The parser-robustness implementation is included as experiment code; its original case definitions and benchmark-specific deterministic-parser rules are not published. The LLM parser path requires the unpublished original inputs.

The **public primary evaluator** is `hf13/answer_evaluator_strict_general.py`, invoked through `score_hf13_strict_general_refined_evaluator.sh` and its `scripts/score_hf13_strict_general_*.py` components. Earlier evaluator versions are **not** separate public scoring interfaces. Some internal compatibility utilities retain historical module names and compute intermediate metrics; they should not be mistaken for alternative published scores.

## Environment

The experiments were developed with Python 3.11, [Ollama](https://ollama.com/), [Mem0](https://github.com/mem0ai/mem0), Qdrant, Qwen2.5 3B/7B, and `nomic-embed-text`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r server/requirements-server.txt
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

The configuration files assume a locally available Ollama server. See `server/` for runtime setup utilities.

## Entrypoints

```bash
# Inspect experiment subcommands.
python -m hf13 --help

# Data-independent checks for the strict-general evaluator.
python -m pytest -q tests/test_answer_evaluator_strict_general.py
```

**For the original experiment only:** after the unpublished benchmark and, where applicable, the private parser mapping have been made available in the expected `data/longitudinal/` structure and the model services have been configured, the main experiment stages are accessible through `run_hf13_core.sh` (one model at a time). The submitted strict-general rescore for a completed campaign is invoked with:

```bash
bash score_hf13_strict_general_refined_evaluator.sh \
  /path/to/completed-campaign 7b,3b
```

This command requires the original evaluator-only data and completed campaign outputs. It does not work on an empty public checkout. Query IDs, target mappings, and benchmark-specific deterministic parser rules are omitted as part of the dataset restriction. Any original parser-diagnostic replication also requires those privately held components.

**Evaluation caution:** this public release preserves the core runtime and the v3.2 scoring entrypoint, but removing the unpublished benchmark-specific compatibility material changes the set of available auxiliary paths. No claim of byte-for-byte or numerical equivalence for a full rerun of the sanitized release is made without additional private-input verification.

## Data policy

The research benchmark and its answer annotations are intentionally not available in this repository. Evaluation source code is included to document the methodology, not to distribute the benchmark. Please do not interpret the absence of public input files as evidence that model inference or scoring can be reproduced without them.

## License

See [LICENSE](LICENSE) for the repository license. External dependencies have their own licensing terms.
