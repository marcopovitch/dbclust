
#!/bin/bash

set -e  # Exit on any error

NLL_PATH="$HOME/github/nll"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_PATH="$SCRIPT_DIR/patch/nll/NLLocLib.patch"

echo "Installing NonLinLoc to $NLL_PATH..."

# Check if patch file exists
if [ ! -f "$PATCH_PATH" ]; then
    echo "Error: Patch file not found at $PATCH_PATH"
    exit 1
fi

# Clone NonLinLoc if directory doesn't exist
if [ ! -d "$NLL_PATH" ]; then
    echo "Cloning NonLinLoc repository..."
    git clone --depth=1 https://github.com/ut-beg-texnet/NonLinLoc "$NLL_PATH"
else
    echo "NonLinLoc directory already exists, updating..."
    cd "$NLL_PATH"
    git pull --depth=1
fi

# Navigate to src directory
cd "$NLL_PATH/src"

# Clean previous build
echo "Cleaning previous build..."
rm -rf bin
mkdir bin
rm -f CMakeCache.txt

# Apply patch and build NonLinLoc
echo "Applying NLLocLib patch..."
cp "$PATCH_PATH" "$NLL_PATH/src"
patch -p0 < NLLocLib.patch

echo "Building NonLinLoc..."
cmake .
make

# Create symlink
echo "Creating symlink..."
ln -sf "$NLL_PATH/src/bin" "$NLL_PATH/bin"

# Cleanup
echo "Cleaning up..."
rm "$NLL_PATH/src/NLLocLib.patch"

echo "NonLinLoc installation completed successfully!"
echo "Binary location: $NLL_PATH/bin"
