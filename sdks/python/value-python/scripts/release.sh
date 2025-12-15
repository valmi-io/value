#!/bin/bash
# Usage: ./scripts/release.sh 0.1.2 "Release notes"

set -e

VERSION=$1
NOTES=${2:-"Release v$VERSION"}

if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/release.sh <version> [notes]"
    echo "Example: ./scripts/release.sh 0.1.2 'Bug fixes and improvements'"
    exit 1
fi

echo "Releasing version $VERSION..."

# Update version in pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml && rm pyproject.toml.bak

# Update version in __init__.py
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" src/value/__init__.py && rm src/value/__init__.py.bak

# Commit version bump
git add pyproject.toml src/value/__init__.py
git commit -m "Bump version to $VERSION"
git push

# Create GitHub release (triggers publish workflow)
gh release create "v$VERSION" --title "v$VERSION" --notes "$NOTES"

echo "Released v$VERSION!"
echo "PyPI publish will be triggered automatically."
