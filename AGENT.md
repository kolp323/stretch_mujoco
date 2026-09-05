# NPC Independent Development and Git Protocol

This file supplements `AGENTS.md`. It defines the mandatory Git workflow for
agents working on the NPC migration. The goal is to keep NPC work reviewable,
reversible, and isolated from the original project and from parallel NPC work.

## Branch Roles

- `main` is the original-project baseline. Never commit or push directly to it.
- `feat/npc-system` is the NPC integration branch. Its eventual pull request
  targets `main`; agents do not place unrelated or multi-task commits directly
  on it.
- Each independently reviewable task uses `feat/npc/<scope>`, based on
  `origin/feat/npc-system`, and opens a pull request back to
  `feat/npc-system`. Examples: `feat/npc/recording-v2`,
  `fix/npc/handover-timeout`, and `docs/npc/asset-manifest`.

If there is no separate NPC integration branch for a solo effort, a single
`feat/npc-system` branch may target `main`. Do not use that exception when
parallel work is active.

## Mandatory Preflight

Before editing code, every agent must run:

```bash
git status --short
git diff --name-only
git branch --show-current
git fetch origin --prune
```

Rules:

- Treat all pre-existing modified or untracked files as user-owned.
- Do not switch branches, rebase, reset, clean, stash, or create a worktree
  from a dirty working directory.
- If the current branch is `main` or `feat/npc-system`, create a dedicated
  task worktree before implementation. If existing changes cannot be safely
  separated, leave them untouched and report the exact conflict.
- Read the relevant sections of `aaa_workspace/docs/current.md` before any
  NPC schema, asset, animation, interaction, transport, recording, or
  compatibility change. Update that record in the same task when implemented
  behavior or migration status changes.

## Isolated Task Workspace

Prefer a separate Git worktree for every agent task. From a clean repository
or another clean worktree:

```bash
git fetch origin --prune
git worktree add ../stretch_mujoco-npc-<scope> \
  -b feat/npc/<scope> origin/feat/npc-system
cd ../stretch_mujoco-npc-<scope>
```

One agent owns one task branch and worktree. Do not share a worktree with a
different agent. Do not edit another task's files merely to resolve an
unrelated conflict; record the dependency instead.

## Change and Commit Rules

Agents are authorized to create a local commit for a completed, focused code
task without asking again, provided all of the following are true:

1. The task is on a dedicated `feat/npc/<scope>`, `fix/npc/<scope>`,
   `test/npc/<scope>`, or `docs/npc/<scope>` branch.
1. Only files required by that task are staged.
1. Relevant focused checks have run, or a concrete environment/dependency
   blocker is recorded in the handoff and PR description.
1. The staged diff was reviewed and contains no credential, private asset,
   generated recording, local configuration, or unrelated user change.

Use this sequence:

```bash
git status --short
git diff -- <relevant-paths>
git add <explicit-path-1> <explicit-path-2>
git diff --cached --check
git diff --cached
git commit -m "feat(npc): concise task summary"
```

Never use `git add .`, `git add -A`, or `git commit -a` for NPC work. Do not
stage `*.local.json`, `.env`, private SMPL-X inputs, generated assets without
their approved manifest, recordings, `outputs/`, `aaa_workspace/task/`, or
diagnostic dumps. The presence of an ignored path is not permission to delete
or overwrite it.

Use conventional, single-purpose commits. Split code, tests, docs, and XML or
asset changes when they can be reviewed independently; keep them together only
when separating them would make the behavior unverifiable.

Before committing, verify the local repository identity is exactly:

```bash
git config user.name
git config user.email
```

Expected values are `kolp323` and `1317416016@qq.com`. If they differ, set
them only in this repository with `git config user.name ...` and
`git config user.email ...`; never change global Git identity.

## Verification and Implementation Record

For each behavior change, run the narrowest meaningful tests first. Run
`uv run pytest -q` and `uv run pre-commit run --all-files` when practical. If
the full suite is blocked, record the exact command, result, and blocker;
never claim it passed.

For NPC migration work, update `aaa_workspace/docs/current.md` in the same
commit with the actual contract, data flow, compatibility effect, limitations,
and observed verification result. This document is an implementation record,
not a plan or a place for generated output.

## Synchronization, Push, and Pull Requests

Before opening or updating a task PR, synchronize only from a clean task
worktree:

```bash
git fetch origin --prune
git rebase origin/feat/npc-system
<run relevant checks again>
git push -u origin HEAD
```

After rebasing an already-pushed task branch, use:

```bash
git push --force-with-lease
```

Never use an unguarded force push. Task agents never push directly to `main`
or `feat/npc-system`; only the designated integration owner may update
`feat/npc-system` when publishing reviewed task merges or its final rebase. An
agent may create or update the task PR when GitHub access is available;
otherwise it must leave the verified local commit intact and report the exact
branch and push/PR commands needed.

Each task PR targets `feat/npc-system` and states:

- scope and explicit non-goals;
- base branch and compatibility impact;
- tests run, observed results, and any blocked broader checks;
- asset/XML/schema/semantic-ID or public API changes;
- reproduction steps or video evidence for scene and visual changes.

When the NPC integration is ready, its owner rebases `feat/npc-system` onto
`origin/main`, runs the agreed integration checks, publishes that rebase with
`git push --force-with-lease`, and opens the single final PR from
`feat/npc-system` to `main`.

## Handoff

At the end of every task, an agent reports the branch, commit SHA (if created),
staged versus unstaged state, checks run, and remaining risks. It must not hide
uncommitted work, failed checks, conflicts, or an inability to push.
