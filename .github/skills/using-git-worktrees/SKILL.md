---
name: using-git-worktrees
description: Use when starting feature work that needs isolation from current workspace or before executing implementation plans - creates isolated git worktrees with smart directory selection and safety verification
---

# Using Git Worktrees

## Overview

Git worktrees create isolated workspaces sharing the same repository, allowing work on multiple branches simultaneously without switching. This enables parallel agent workflows where each agent operates in its own worktree.

**Core principle:** Systematic directory selection + safety verification + repo-specific setup = reliable isolation.

**Announce at start:** "I'm using the using-git-worktrees skill to set up an isolated workspace."

## Creation Flow

### 1. Plan Work and Confirm Worktree Details

The user typically requests a worktree alongside a task description. Before creating the worktree:

1. **Brainstorm and plan the task** — survey the codebase, ask clarifying questions, discuss tradeoffs
2. **Include the worktree proposal in your planning response** — present the parent branch, new branch name, and worktree directory as part of the plan for the user to confirm
3. **Wait for user confirmation** of both the plan and the worktree details before proceeding

Present the worktree details like:

```
Git worktree:
  Parent branch: main
  New branch:    feature/<descriptive-name>
  Directory:     .worktrees/feature/<descriptive-name>
```

**Important:** Once the worktree is created, all subsequent code modifications, experiment runs, and file operations must happen inside the worktree directory — not in the main worktree.

### 2. Verify `.worktrees/` is Git ignored

```bash
git check-ignore -q .worktrees/ 2>/dev/null
```

**If NOT ignored:** Add `.worktrees/` to `.gitignore` and commit the change before proceeding.

### 3. Create Worktree

```bash
git worktree add .worktrees/<branch-name> -b <branch-name>
cd .worktrees/<branch-name>
```

### 4. Install Dependencies

```bash
uv sync --all-extras
```

Each worktree gets its own `.venv/` for package isolation. `uv`'s global cache means downloads are near-zero — it mostly symlinks, so the overhead is minimal. This ensures each worktree can independently `uv add <package>` without affecting others.

### 5. Repo-Specific Post-Checkout Setup

Check `copilot-instructions.md` (or equivalent project docs) for repo-specific setup steps. Common examples:

- **Submodule initialization:** `git submodule update --init --recursive`
- **Symlinks:** Shared directories (storage, caches) that should be symlinked from the main worktree
- **Environment files:** `.env` or credentials that need copying

Apply all repo-specific steps before running baseline verification.

### 5a. Verify Repo-Specific Files are Gitignored

After post-checkout setup, verify that any symlinks or generated files (e.g., a symlinked `storage/` directory) are gitignored in the worktree. This prevents `git add -A` from accidentally staging them.

```bash
# Check each repo-specific path created during setup
git check-ignore -q <symlink-or-generated-path> 2>/dev/null || echo "WARNING: not ignored"
```

**If not ignored:** Use selective `git add <files>` instead of `git add -A` during commits.

### 6. Verify Clean Baseline

Run the project's validation command (check `Makefile` or project docs):

```bash
make all    # or whatever the project uses
```

**If tests fail:** Report failures and ask whether to proceed or investigate.
**If tests pass:** Report ready.

### 7. Report

```
Worktree ready at <full-path>
Branch: <branch-name> (from <parent-branch>)
Validation passing
Ready to work on <task-description>
```

## Cleanup / Teardown

When work in a worktree is complete:

### 1. Ensure Changes are Committed

```bash
cd .worktrees/<branch-name>
git add -u && git commit -m "<message>"
```

**Always use `git add -u`** (stages only already-tracked files). Never use `git add -A` or `git add .` in worktrees — they can stage symlinks (e.g., `storage`) created during post-checkout setup, causing data loss on merge.

**Optional — push to remote** if collaboration or backup is needed:

```bash
git push -u origin <branch-name>
```

For local-only workflows, skip the push.

### 2. Pre-Merge Safety Checks

Before merging, verify the branch doesn't accidentally track paths created during post-checkout setup (symlinks, generated files). Check `copilot-instructions.md` for the list of repo-specific paths to verify.

```bash
# For each symlink/path created during post-checkout setup:
git ls-tree -r --name-only <branch-name> | grep -q '^<path>$' && echo "DANGER: branch tracks <path>" || echo "OK"
```

**If DANGER:** Do NOT merge. Remove the tracked path from the branch first:

```bash
cd .worktrees/<branch-name>
git rm --cached <path>
git commit -m "fix: remove accidentally tracked <path>"
```

### 3. Merge into Parent Branch

Explicitly `cd` to the main worktree root (not the feature worktree):

```bash
cd <main-worktree-root>   # e.g., /path/to/repo (NOT .worktrees/...)
git checkout <parent-branch>
git merge <branch-name>
```

If the merge succeeds cleanly, proceed to step 3.

**If merge conflicts arise:** follow the project's merge conflict resolution guidelines in `copilot-instructions.md`. General approach:

1. Assess conflict complexity — simple (one side added, other didn't touch) vs. ambiguous (both sides modified same region)
2. For simple conflicts: resolve automatically
3. For ambiguous conflicts: present the conflicting hunks to the user, explain options (take ours / take theirs / manual blend), and ask for approval before resolving
4. If the user prefers to handle it manually, pause and let them resolve, then continue cleanup when asked

### 4. Remove Worktree

Remove the worktree **before** deleting the branch. If the worktree contains submodules, deinit them first and use `--force`:

```bash
# If worktree has submodules:
cd .worktrees/<branch-name>
git submodule deinit -f --all
cd <main-worktree-root>

# --force is required after submodule deinit (git considers the tree "dirty")
git worktree remove --force .worktrees/<branch-name>
```

If there are no submodules:

```bash
git worktree remove .worktrees/<branch-name>
```

### 5. Delete Branch (Local + Remote)

```bash
git branch -d <branch-name>
```

**Optional — delete remote branch** if it was pushed:

```bash
git push origin --delete <branch-name>
```

## Quick Reference

| Situation | Action |
|-----------|--------|
| `.worktrees/` exists | Use it (verify ignored) |
| `.worktrees/` not ignored | Add to `.gitignore` + commit |
| User wants different parent branch | Checkout that branch first, then create worktree |
| Tests fail during baseline | Report failures + ask user |
| Merge conflicts on cleanup | Follow repo-specific conflict resolution guidelines |
| Worktree branch already pushed | Delete remote branch during cleanup |
| Worktree has submodules | `git submodule deinit -f --all` then `git worktree remove --force` |
| Repo-specific symlinks not gitignored | Use `git add -u` instead of `git add -A` |
| Branch tracks repo-specific symlinks | Remove with `git rm --cached <path>` before merging |

## Common Mistakes

### Skipping ignore verification

- **Problem:** Worktree contents get tracked, pollute git status
- **Fix:** Always use `git check-ignore` before creating project-local worktree

### Skipping user confirmation

- **Problem:** Wrong parent branch or branch name
- **Fix:** Include worktree details in the planning response; user confirms both the plan and the worktree setup together

### Skipping repo-specific setup

- **Problem:** Missing submodules, symlinks, or shared caches cause runtime failures
- **Fix:** Always check `copilot-instructions.md` for post-checkout steps

### Proceeding with failing tests

- **Problem:** Can't distinguish new bugs from pre-existing issues
- **Fix:** Report failures, get explicit permission to proceed

### Leaving orphan branches after cleanup

- **Problem:** Branch pollution in local and remote
- **Fix:** Always delete local branch after merge; delete remote branch if it was pushed

### Staging repo-specific symlinks

- **Problem:** `git add -A` stages symlinks/generated files created during post-checkout setup, causing data loss on merge
- **Fix:** Always use `git add -u` (stages only tracked files). Never use `git add -A` or `git add .` in worktrees

### Merging without checking for tracked symlinks

- **Problem:** Branch accidentally tracks a symlink created during post-checkout setup; merge replaces the real directory with the symlink, potentially destroying data
- **Fix:** Run pre-merge safety check (`git ls-tree -r --name-only <branch> | grep '<path>'`) before every merge. See `copilot-instructions.md` for repo-specific paths to check

### Worktree removal fails after submodule deinit

- **Problem:** `git worktree remove` errors with "working trees containing submodules cannot be moved" even after `git submodule deinit -f --all`
- **Fix:** Use `git worktree remove --force` — deinit leaves `.git/modules/` metadata that git considers "dirty"

## Red Flags

**Never:**
- Create worktree without verifying it's ignored
- Skip baseline validation
- Proceed with failing tests without asking
- Skip user confirmation of branch names
- Delete a worktree without merging first
- Force-delete branches without confirming merge status
- Use `git add -A` or `git add .` in worktrees (use `git add -u` instead)
- Merge a branch without running pre-merge safety checks

**Always:**
- Include worktree details in planning response for user confirmation
- Verify `.worktrees/` is in `.gitignore`
- Run repo-specific post-checkout setup from `copilot-instructions.md`
- Verify repo-specific files (symlinks, generated paths) are gitignored after setup
- Run pre-merge safety check before merging (verify no repo-specific symlinks are tracked)
- Run project validation before starting work
- Do all code modifications inside the worktree directory, not the main worktree
- Remove worktree before deleting branch during cleanup
- Deinit submodules before removing worktree
- Merge before deleting branches

## Integration

**Called by:**
- Any workflow needing isolated workspace for parallel agent work
- Feature development requiring isolation from main branch
- Experiment runs that modify tracked source files

**Pairs with:**
- Repo-specific `copilot-instructions.md` for post-checkout setup and merge conflict guidelines
