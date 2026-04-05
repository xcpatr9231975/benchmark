#!/bin/bash
# GitHub Push Script - Replace YOUR_TOKEN with your Personal Access Token

echo "=== GitHub Repository Setup ==="
echo "Repository: xcpatr9231975/benchmark"
echo ""

echo "To push your benchmark code to GitHub:"
echo ""
echo "1. Create a Personal Access Token at: https://github.com/settings/tokens"
echo "   - Name: benchmark-push"
echo "   - Expires: 90 days (recommended)"
echo "   - Repository access: Only this repository (xcpatr9231975/benchmark)"
echo "   - Scopes: Check 'repo' (full control of private repositories)"
echo "   - Click 'Generate token'"
echo ""
echo "2. Copy the generated token (starts with ghp_...)"
echo ""
echo "3. Run this command (replace YOUR_TOKEN):"
echo ""
echo "   cd /home/cpatriaw/benchmark"
echo "   git push https://xcpatr9231975:YOUR_TOKEN@github.com/xcpatr9231975/benchmark.git main"
echo ""
echo "Or run: git push origin main (will prompt for password = your token)"
echo ""

cd /home/cpatriaw/benchmark

# Test if we can push
if git remote get-url origin 2>/dev/null | grep -q github.com; then
    echo "Origin is already set. Trying to push..."
    echo "Note: You'll be prompted for username/password."
    echo "Username: xcpatr9231975"
    echo "Password: (paste your token here - you won't see it)"
    git push origin main
else
    echo "No origin remote configured."
fi
