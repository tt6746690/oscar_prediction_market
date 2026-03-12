# Copilot Instructions

Prediction market arbitrage — identifying price discrepancies across markets.

---

## Quick Reference

```bash
make install      # Install dependencies
make dev          # Format + lint + typecheck + test
make all          # Full validation before commit
```

---

## Environment

- Run all Python via `uv run` (e.g., `uv run python ...`, `uv run pytest`) or `make` targets
- Install deps: `uv sync --all-extras` (or `make install`)
- Add a dep: `uv add <package>` (updates `pyproject.toml` + `uv.lock`)
- Never use bare `python`, `pip install`, or `pip`

---

## Project Structure

```
prediction_market_arbitrage/
├── prediction_market_arbitrage/   # Source code
├── tests/                         # test_*.py files
├── storage -> external dir        # Symlink to ../prediction_market_arbitrage_storage
├── pyproject.toml                 # Config & dependencies
└── Makefile                       # Dev commands
```

### Naming Conventions

| Pattern | Example |
|---------|---------|
| Date-prefixed project dirs | `d20260201_oscar/` |
| One-off scripts | `one_offs/d<YYYYMMDD>_<name>/` |
| Storage dirs match one-off dirs | `storage/d<YYYYMMDD>_<name>/` |

---

## Code Conventions

### Style

| Setting | Value |
|---------|-------|
| Python | 3.12 |
| Line length | 100 |
| Formatter | ruff (double quotes) |
| Imports | ruff/isort |

- Use absolute imports from package root. No relative imports.
- Library modules use `logging.getLogger(__name__)`. Scripts and one-offs use `print()`.

### Type Safety

- Type hints on all function parameters and return values
- Prefer local `# type: ignore` over global ignores in `pyproject.toml` unless necessary
- When `# type: ignore` is needed, add a comment explaining why (e.g., `# type: ignore[prop-decorator]  # Pydantic computed field`)
- Missing library stubs: try to find corresponding stubs/types and add to dev dependencies. If not available, use `[[tool.mypy.overrides]]` with `ignore_missing_imports = true`
- Exclude third-party code in `pyproject.toml`, don't fix it

### Pydantic v2

Use Pydantic `BaseModel` for all data structures. Prefer over `dataclass`.

```python
class MyModel(BaseModel):
    model_config = {"extra": "forbid"}           # prevent typos in field names

    name: str = Field(...)                        # required (no default)
    count: int = Field(ge=0)                      # validation at ingestion
    items: list[str] = Field(min_length=1)

    @computed_field  # type: ignore[prop-decorator]  # Pydantic computed field
    @property
    def derived_value(self) -> int:
        return self.x + self.y
```

- Set `model_config = {"extra": "forbid"}` on all models to catch field name typos early
- Use `Field(...)` with validation constraints for data integrity at ingestion
- Use `TypeAdapter` for loading typed data from JSON files
- Use discriminated unions for polymorphic configs: `Annotated[Union[TypeA, TypeB], Discriminator(...)]`
- Pass Pydantic models directly — don't extract to dicts
- Use `model_validate_json` / `validate_python` for data loading
- Use `StrEnum` for categorical constants that appear in data. Any `Literal["a", "b"]` used across multiple files should be a `StrEnum`.
- Derived fields (computable from other fields on the same model) should be `@computed_field` properties, not stored fields. This makes input vs derived explicit and prevents sync bugs.
- `@computed_field` vs `@property`: Use `@computed_field` when the value will be serialized (logged, saved as JSON, displayed in reports). Use plain `@property` for convenience accessors used only in code.
- Result models should not store the config that produced them — the caller already has it. Serialize them side-by-side at the call site if needed.
- If a `_make_foo()` helper exists only to compute derived values and construct `Foo(...)`, it's a code smell — make `Foo` compute those values via `@computed_field` instead. More broadly, don't write thin factory functions that just forward arguments to a model constructor — let callers construct the model directly.
- Embed sub-configs as fields rather than flattening their fields and writing `to_sub_config()` converters.
- Don't attach consumer-specific data to shared models. If only one caller needs a field (e.g., adding `model_prob: float | None = None` to a trade record so the backtest can annotate it), keep it out of the core model — the caller already has that data and can join it at the analysis site.
- If a function produces a list of items, return `list[Item]` directly. Don't create a wrapper model with `items: list[Item]` plus computed aggregates — callers can aggregate themselves. Wrappers are justified only when they carry non-derivable state (e.g., an optimization convergence flag).

### Error Handling

Fail fast. No try/except unless there's a real recovery strategy.

- Let errors propagate with clear stack traces
- Catch bad data at ingestion via Pydantic validation
- Use `ValueError` for impossible states
- Only catch exceptions at API/IO boundaries where recovery is possible

### Default Values & Required Parameters

Wrong defaults cause silent, hard-to-debug errors. The litmus test: does this parameter affect what the function computes, or how it runs?

- Require (no defaults): Semantic parameters — anything that changes the function's output or meaning. If a caller gets this wrong, results are silently incorrect. This includes all Pydantic config/model fields (`Field(...)` for required, `None` for truly absent data).
- Defaults OK: Mechanical parameters — infrastructure, performance, and presentation concerns that don't change correctness (caching, verbosity, rate limits, API client instances).

Another litmus test: If no caller ever overrides a default, or the function is always called with an explicit argument, remove the default entirely.

### Code Organization

- Extract functions when logic gets complex — includes summary/reporting logic
- No hardcoding — use JSON configs or Pydantic models
- Separate concerns: data loading / feature engineering / modeling / evaluation
- Check for existing similar logic before writing new code; consolidate
- Clean data at boundaries — normalize inputs at ingestion, not downstream
- Prefer stateless functions — pass data in, get results out. Avoid classes that accumulate state unless there's a clear lifecycle (e.g., API client with auth).
- Use domain-agnostic names in reusable modules. Generic concepts (edge, Kelly, signals) should use generic terminology (`outcome`, `event`), not domain-specific names (`nominee`, `oscar`). Domain-specific naming belongs in the domain-specific wrapper.
- Name symmetric parameters symmetrically (`buy_edge_threshold` / `sell_edge_threshold`, not `min_edge` / `sell_threshold_edge`).

### Documentation

- Pedagogical docstrings — explain the *why* and the math, not just API surface. Someone unfamiliar with the domain should learn the concept from the docstring alone.
- Put docs in code — don't create separate markdown summaries
- Module-level README.md for non-trivial packages — architecture diagram, module responsibilities, concept glossary
- Data-fetching functions (API wrappers, market data) should include concrete `Example::` blocks in docstrings showing realistic input/output so callers can see the data shape without running the code.

---

## Testing

- Avoid adding tests for all new functionality unless the user requests it
- Mirror source structure: `tests/<module>/test_<file>.py` when the source module has multiple files. One monolithic test file per package is OK only when the package is small.
- Pattern: `tests/test_*.py`
- Naming: `test_<functionality>_<scenario>`
- Check coverage: `make test-cov`

### Test Style

- **Pedagogical**: tests should teach. Show step-by-step math, explain what's being verified and why. A reader should understand the underlying concept from the test alone.
- **Test the interface**: prefer testing input/output behavior of functions over mocking internals. Use mocks only when external dependencies (APIs, filesystem) make direct testing impractical.
- **Synthetic or real data**: either is fine. Construct test data that makes the expected result obvious and verifiable by hand.
- Fixtures and conftest are fine when they reduce duplication.

---

## Experiments

Every experiment lives in `storage/d<YYYYMMDD>_<name>/` and must be fully reproducible from that directory alone.

If applicable, include:
1. Copied configs (not symlinks to source)
2. `run.log` with full stdout/stderr
3. Artifacts (e.g., metrics, plots, model checkpoints, etc.)

Usually there is a corresponding one-off script in `one_offs/d<YYYYMMDD>_<name>/` that orchestrates the experiment and conduct analysis of results. Include:
1. `run.sh` (or `run.py` or other executable) to reproduce the results
2. `README.md` with findings — this is the canonical location for experiment results
3. scripts for config generation or result analysis if needed


### Key Principles

- **Configs go in storage** — copy configs into the experiment directory before running. Source configs may change later; the storage copy is the ground truth.
- **Log everything** — `bash one_offs/d<YYYYMMDD>_<name>/run.sh 2>&1 | tee storage/d<YYYYMMDD>_<name>/run.log`
- **Self-contained** — anyone should be able to `bash one_offs/d<YYYYMMDD>_<name>/run.sh` or similar and get the same results.
- **Source configs are defaults** — `modeling/configs/` reflects best-known settings. Experiment-specific configs go in storage.

### run.sh Template

All run scripts follow this structure:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"           # worktree-safe (never dirname $0)

# Configuration — uppercase variables
DATASET_DIR="storage/d20260201_build_dataset"
EXP_DIR="storage/d<date>_<name>"

# Validation — check required files exist
[[ -f "${DATASET_DIR}/data.json" ]] || { echo "Missing dataset"; exit 1; }

# Execution — skip if output already exists (idempotent)
if [[ ! -f "${EXP_DIR}/output.json" ]]; then
    echo "=== Step 1: Description ==="
    uv run python -m fully.qualified.module --arg value
fi
```

### One-off README Template

Each one-off directory has a `README.md` with findings. This is the canonical
location for experiment results — not a separate log file.

```markdown
# Descriptive Title

**Storage:** `storage/d<date>_<name>/`

Brief description of what was tested and why.

## Motivation

Why this experiment exists, what question it answers.

## Setup

Design, configs, dimensions explored.

## Findings

### Takeaway-oriented subsection title

Supporting data, tables, images. Group results by narrative/takeaway,
not by raw data dump. Each subsection title should BE the takeaway.

## How to Run

\```bash
cd "$(git rev-parse --show-toplevel)"
bash .../run.sh 2>&1 | tee storage/.../run.log
\```

## Output Structure

\```
storage/d<date>_<name>/
├── ...
\```
```

### Experiment Index (`experiment_index.md`)

Thin reverse-chronological index linking to one-off READMEs. Format:

```markdown
**[d20260214_trade_signal_ablation](one_offs/d20260214_trade_signal_ablation/)** (2026-02-14)
- Key finding bullet 1.
- Key finding bullet 2 (optional).
- Key finding bullet 3 (optional).
```

Bolded directory name links to the one-off. 1–3 bullets per entry. Don't
duplicate findings here — just enough to find the right README.

### Image Assets Convention

Images referenced in one-off READMEs use this format:
```markdown
![storage/d<date>_<name>/plot.png](assets/plot.png)
```
- Alt text = source path in `storage/` (relative to repo root)
- Link target = local `assets/` copy (renders on GitHub)
- Run `bash one_offs/sync_assets.sh` to copy images from storage to assets/

### README Analytical Depth

READMEs are the **canonical record** of experiment results — not summaries.
They should contain the full analytical depth that a reader needs to
understand findings without reading code or raw logs:

- **Results tables** with all configs, metrics, and model comparisons
- **Commentary** explaining *why* results look the way they do (not just what)
- **Deep dives** for interesting sub-analyses (trade-by-trade replays, failure anatomies, probability concentration analysis)
- **Cross-references** to related experiments where results inform each other
- **Images** with substantive captions explaining what the plot shows

Organize around takeaways, not raw data dumps. Each subsection title should
BE the takeaway. A reader should understand the full experiment narrative
from the README alone.

### Follow-up Experiments

When a follow-up extends an earlier experiment, add a dated subsection to the
same one-off README rather than creating a new directory. Cross-reference
related experiments via relative links: `[earlier study](../d20260207_feature_ablation/)`.

Use dates in titles, not "Experiment 1/2/3". Organize around takeaways, not data dumps. Include tables/plots that support the narrative. Keep markdown simple. Goal is to allow reader to get the gist of the experiment without needing to read code or raw logs.

---

## Git & Worktrees

### Workflow

- Run `make all` before committing — all must pass (format, lint, typecheck, test, build)
- Brainstorm first — ask clarifying questions before implementing complex or vague tasks
- Fix errors completely — don't suppress warnings or leave for later
- Update copilot-instructions.md with learned preferences

### Worktree Setup

See `.github/skills/using-git-worktrees/SKILL.md` for the general workflow. Post-checkout steps:

```bash
git submodule update --init --recursive
ln -s "$(git worktree list | head -1 | awk '{print $1}')/../prediction_market_arbitrage_storage" storage
git check-ignore -q storage || echo "WARNING: storage not gitignored"
uv sync --all-extras
```

If `make all` reformats files you didn't touch, commit those separately:

```bash
git add -u && git commit -m "style: auto-format"
```

### Storage Safety

`storage/` is a symlink to `../prediction_market_arbitrage_storage` — outside the git repo. All worktrees share this directory for experiment results and API caches.

**Critical rules:**
- Never use `git add -A` or `git add .` — these can stage the storage symlink. Use `git add -u` or `git add <specific-files>`.
- Pre-commit check: `git diff --cached --name-only | grep -q '^storage$' && echo "DANGER" || echo "OK"`
- Pre-merge check: `git ls-tree -r --name-only <branch> | grep -q '^storage$' && echo "DANGER" || echo "OK"`
- Never use `cd "$(dirname "$0")/../.."` in scripts — it resolves through the symlink to the wrong repo. Use `cd "$(git rev-parse --show-toplevel)"`.

### Worktree Cleanup

```bash
rm -rf .worktrees/feature/<name>
git worktree prune
git branch -d feature/<name>
```

### Merge Conflicts

Simple conflicts (one side added, other didn't touch; whitespace-only): resolve automatically.

Ambiguous conflicts (both sides modified same region):
1. Show the conflicting hunks with context
2. Explain options: take ours / take theirs / blend
3. Recommend with rationale
4. Wait for user approval

### Commit Messages

Use `git commit -F <file>` for multi-line messages with special characters:

```bash
cat > /tmp/commit_msg.txt << 'EOF'
feat: description here

Body with special chars -- arrows, etc.
EOF
git commit -F /tmp/commit_msg.txt
```

---

## VS Code / Agent Gotchas

- `run_in_terminal` with `isBackground=true` spawns a shell at the workspace root, not the worktree. Use non-background terminals or `cd` explicitly.
- VS Code workspace root ≠ worktree root. Be explicit about paths.
- Run from the worktree root, not via absolute paths through `storage/`. Verify CWD (`pwd`) before running experiments.

---

## Domain-Specific Notes

- Keep market logic separate from API/fetching code
- Cache expensive API calls with diskcache
- Handle edge cases: missing prices, stale data, invalid markets
- Domain data has quirks (e.g., ceremony year ≠ release year) — document assumptions

---

## Updating This Document

Update when:
- A pattern emerges from repeated feedback
- A workaround becomes standard practice
- Domain-specific quirks are discovered

Keep tips generalizable and actionable. Consolidate duplicates. Use tables for reference, bullets for principles.
