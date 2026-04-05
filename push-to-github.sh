#!/bin/bash
# Interactive GitHub Push Script

echo "======================================"
echo "  GitHub Push Helper"
echo "======================================"
echo ""

# Check if token file exists
if [ -f ~/.github_token ]; then
    echo "Found saved token (last modified: $(stat -c %y ~/.github_token | cut -d' ' -f1))"
    if [ -z "${TOKEN}" ]; then
        read -p "Use saved token? (y/n): " use_saved
        if [ "$use_saved" == "y" ]; then
            TOKEN=$(cat ~/.github_token)
        fi
    fi
fi

if [ -z "$TOKEN" ]; then
    echo ""
    echo "GitHub Personal Access Token required"
    echo ""
    echo "1. Visit: https://github.com/settings/tokens/new"
    echo "2. Set: Token name='benchmark-push', Expiration='90 days'"
    echo "3. Scope: Check 'repo' (full control of private repos)"
    echo "4. Click 'Generate token'"
    echo "5. Copy the generated token"
    echo ""
    read -s -p "Enter your token: " TOKEN
    echo ""
    
    # Save token securely
    read -p "Save token for future use? (y/n): " save_token
    if [ "$save_token" == "y" ]; then
        echo "$TOKEN" > ~/.github_token
        chmod 600 ~/.github_token
        echo "Token saved to ~/.github_token"
    fi
fi

echo ""
echo "Attempting to push to GitHub..."

cd /home/cpatriaw/benchmark

if [ -z "$TOKEN" ]; then
    echo "No token provided. Exiting."
    exit 1
fi

git push https://xcpatr9231975:$TOKEN@github.com/xcpatr9231975/benchmark.git main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "Repository: https://github.com/xcpatr9231975/benchmark"
else
    echo ""
    echo "❌ Push failed. Possible reasons:"
    echo "   - Token is invalid or expired"
    echo "   - Repository doesn't exist"
    echo "   - Token doesn't have 'repo' scope"
    echo ""
    echo "To create the repository on GitHub first:"
    echo "   1. Go to: https://github.com/new"
    echo "   2. Repository name: benchmark"
    echo "   3. Owner: xcpatr9231975"
    echo "   4. Select 'Private'"
    echo "   5. Click 'Create repository'"
    echo ""
    echo "Then run this script again."
fi
