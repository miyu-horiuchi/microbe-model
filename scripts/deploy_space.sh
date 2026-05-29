#!/usr/bin/env bash
#
# Deploy the app to the Hugging Face Space.
#
# Why this script exists
# ----------------------
# The HF Space's pre-receive hook rejects raw (non-LFS) binary files. `main`
# carries paper/*.pdf and paper/figures/*.png that the GitHub repo wants but the
# deployed app never needs — so a plain `git push space main` is rejected.
#
# This script ships the current source tree to the Space *minus* paper/, as a
# single commit on top of the Space's own history. The only diff each deploy is
# real app changes, the push stays fast-forward, and the main working tree is
# never touched (all work happens in a throwaway worktree).
#
# Usage
# -----
#   scripts/deploy_space.sh [SOURCE_REF]      # default SOURCE_REF = main
#   SPACE_REMOTE=space scripts/deploy_space.sh
#
# After pushing, the Space rebuilds the Docker image automatically (the web
# frontend is compiled in stage 1, so no local `npm run build` is needed).

set -euo pipefail

REMOTE="${SPACE_REMOTE:-space}"
SOURCE_REF="${1:-main}"
EXCLUDE_DIR="paper"

cd "$(git rev-parse --show-toplevel)"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "error: git remote '$REMOTE' not found. Set SPACE_REMOTE or add the remote." >&2
  exit 1
fi

echo "→ Fetching $REMOTE ..."
git fetch --quiet "$REMOTE"

SRC_SHA="$(git rev-parse --short "$SOURCE_REF")"
WORKTREE="$(mktemp -d)"
cleanup() { git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true; rm -rf "$WORKTREE"; }
trap cleanup EXIT

echo "→ Building deploy commit from $SOURCE_REF@$SRC_SHA (excluding $EXCLUDE_DIR/) ..."
# Base on the Space's existing history so the push is a fast-forward and we do
# not re-introduce rejected binaries from main's history.
git worktree add --quiet --detach "$WORKTREE" "$REMOTE/main"

(
  cd "$WORKTREE"
  # Make the tree exactly match SOURCE_REF, then drop the excluded dir.
  git rm -rfq --ignore-unmatch . >/dev/null
  git checkout "$SOURCE_REF" -- .
  git rm -rfq --cached --ignore-unmatch "$EXCLUDE_DIR" >/dev/null 2>&1 || true
  rm -rf "$EXCLUDE_DIR"
  git add -A

  if git diff --cached --quiet; then
    echo "✓ Space already matches $SOURCE_REF (minus $EXCLUDE_DIR/) — nothing to deploy."
    exit 0
  fi

  git commit --quiet -m "Deploy app from ${SOURCE_REF}@${SRC_SHA} (no ${EXCLUDE_DIR}/ binaries)"
  echo "→ Pushing to $REMOTE/main (triggers HF Docker rebuild) ..."
  git push "$REMOTE" HEAD:main
  echo "✓ Deployed. Watch the build at the Space's logs; runtime sha will match $(git rev-parse --short HEAD)."
)
