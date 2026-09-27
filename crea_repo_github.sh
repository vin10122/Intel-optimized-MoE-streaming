#!/bin/bash

# Script per creare repo GitHub privato via API
# Uso: ./crea_repo_github.sh TUO_TOKEN

if [ -z "$1" ]; then
    echo "Usage: $0 GITHUB_PERSONAL_ACCESS_TOKEN"
    echo ""
    echo "1. Create a PAT at: https://github.com/settings/tokens"
    echo "2. Select scopes: repo (full control of private repositories)"
    echo "3. Run: $0 YOUR_TOKEN"
    exit 1
fi

TOKEN=$1
REPO_NAME="flash-moe-q8"
DESCRIPTION="FLASH-MOE: Q8_0 to BF16 Optimized Inference for Intel Lunar Lake"
PRIVATE=true

echo "Creazione repo $REPO_NAME su GitHub..."

RESPONSE=$(curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"$REPO_NAME\",\"private\":$PRIVATE,\"description\":\"$DESCRIPTION\"}")

echo ""

# Check if repo was created
REPO_URL=$(echo "$RESPONSE" | grep -o '"html_url": *"[^"]*"' | cut -d'"' -f4)

if [ -n "$REPO_URL" ]; then
    echo "✓ Repo creato: $REPO_URL"
    echo ""
    echo "Ora esegui questi comandi:"
    echo "  cd /c/Users/valer/Desktop/moe-client-prot/Nuova cartella"
    echo "  git remote add origin $REPO_URL.git"
    echo "  git push -u origin master"
else
    echo "✗ Errore nella creazione del repo"
    echo "Risposta: $RESPONSE"
    exit 1
fi
