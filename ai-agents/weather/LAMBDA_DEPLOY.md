# Deploy Weather App to AWS Lambda (SAM)

## 1) Prerequisites
- AWS account + IAM user with permissions for Lambda, API Gateway, CloudFormation, and IAM role creation.
- AWS CLI installed and configured (`aws configure`).
- AWS SAM CLI installed.
- Node.js 20+ installed.

## 2) Build and deploy
From `ai-agents/weather`:

```bash
npm install
npm run sam:build
npm run sam:deploy
```

During `sam deploy --guided`, provide:
- Stack Name: `weather-agent-app`
- AWS Region: your preferred region
- Parameter `GeminiApiKey`: your Gemini API key
- Confirm changeset: `Y`
- Allow SAM role creation: `Y`
- Save arguments to `samconfig.toml`: `Y`

## 3) Get app URL
After deployment, SAM prints output `ApiUrl`.
Open that URL in browser.

## 4) Update Gemini key later
```bash
sam deploy --guided -t template.yaml
```
Set new value for `GeminiApiKey` when prompted.

## 5) Remove deployment
```bash
aws cloudformation delete-stack --stack-name weather-agent-app
```

## Notes
- `index.js` runs in two modes:
  - Local dev: `npm start`
  - Lambda runtime: exports `handler` used by AWS Lambda.
- `.env` is for local only. Lambda uses environment variables from SAM template.
