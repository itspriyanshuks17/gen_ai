#!/usr/bin/env bash
# Exit on first failure (`-e`), undefined variable (`-u`), and pipeline errors (`-o pipefail`).
set -euo pipefail

# Always run script from project directory, regardless of current shell path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Validate SAM CLI availability.
if ! command -v sam >/dev/null 2>&1; then
  echo "Error: SAM CLI is not installed or not in PATH."
  exit 1
fi

# Load local environment variables if `.env` exists.
if [ -f .env ]; then
  # Export key(s) from .env for this script execution.
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Abort early if key is missing.
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "Error: GEMINI_API_KEY is not set. Add it to .env or export it in your shell."
  exit 1
fi

# Optional override support:
# STACK_NAME=my-stack AWS_REGION=us-east-1 bash deploy.sh
STACK_NAME="${STACK_NAME:-wweather}"
AWS_REGION="${AWS_REGION:-ap-south-1}"

# Build deployment artifacts and package Lambda function.
sam build -t template.yaml
# Deploy/update stack, passing Gemini key as CloudFormation parameter.
sam deploy \
  -t template.yaml \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides "GeminiApiKey=$GEMINI_API_KEY"

echo "Deployment complete for stack: $STACK_NAME (region: $AWS_REGION)"
