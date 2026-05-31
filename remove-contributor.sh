#!/bin/bash

# Script to remove Kaushalsaathi from commit history
# This script rewrites the initial commit to use only your author information

set -e

echo "=========================================="
echo "Removing Kaushalsaathi from commit history"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d .git ]; then
    echo "ERROR: Not a git repository. Please run this from your repository root."
    exit 1
fi

# Get the current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "ERROR: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

echo "This script will rewrite the initial commit."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read -r

# Rewrite the history using git filter-branch
echo "Rewriting commit history..."
git filter-branch --env-filter '
if [ "$GIT_COMMITTER_NAME" = "kaushalsaathi" ]; then
    export GIT_COMMITTER_NAME="Aneek Ghosh"
    export GIT_COMMITTER_EMAIL="ghoshaneek02@gmail.com"
fi
if [ "$GIT_AUTHOR_NAME" = "kaushalsaathi" ]; then
    export GIT_AUTHOR_NAME="Aneek Ghosh"
    export GIT_AUTHOR_EMAIL="ghoshaneek02@gmail.com"
fi
' -- --all

echo ""
echo "=========================================="
echo "History rewritten successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the changes with: git log --oneline"
echo "2. Force push to GitHub: git push origin $CURRENT_BRANCH --force-with-lease"
echo ""
echo "WARNING: Force pushing rewrites history on GitHub."
echo "Make sure no one else is working on this branch!"
