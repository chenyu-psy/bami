# bami

`bami` provides Bayesian amortized model inference workflows for user-defined and preset models.

The package includes generic BayesFlow workflow builders, prior and posterior-transform helpers, preset simulators, recovery utilities, and evaluation plots used by research-facing analysis notebooks.

## Install locally for development

From a project that should use the local package:

```bash
uv add --editable /Users/chenyu/Packages/bami
```

## Smoke test

```bash
uv run python -c "import bami; from bami.workflows import SimpleWorkflow, HierarchicalWorkflow"
```
