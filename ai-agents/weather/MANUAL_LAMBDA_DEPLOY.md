# Manual AWS Lambda Deployment Guide

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Node.js 20.x installed
- Gemini API Key

## Step 1: Prepare the Application

```bash
cd /home/virtualuser/gen_ai/ai-agents/weather
npm install --production
```

## Step 2: Create Deployment Package

```bash
# Create a deployment directory
mkdir -p lambda-deploy
cp -r node_modules lambda-deploy/
cp index.js lambda-deploy/
cp -r public lambda-deploy/

# Create ZIP file
cd lambda-deploy
zip -r ../weather-agent.zip .
cd ..
```

## Step 3: Create IAM Role for Lambda

```bash
# Create trust policy file
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create IAM role
aws iam create-role \
  --role-name weather-agent-lambda-role \
  --assume-role-policy-document file://trust-policy.json

# Attach basic execution policy
aws iam attach-role-policy \
  --role-name weather-agent-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

## Step 4: Create Lambda Function

```bash
# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create Lambda function
aws lambda create-function \
  --function-name weather-agent-app \
  --runtime nodejs20.x \
  --role arn:aws:iam::${ACCOUNT_ID}:role/weather-agent-lambda-role \
  --handler index.handler \
  --zip-file fileb://weather-agent.zip \
  --timeout 30 \
  --memory-size 256 \
  --environment Variables={GEMINI_API_KEY=your_gemini_api_key_here}
```

**Replace `your_gemini_api_key_here` with your actual Gemini API key**

## Step 5: Create API Gateway HTTP API

```bash
# Create HTTP API
API_ID=$(aws apigatewayv2 create-api \
  --name weather-agent-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:us-east-1:${ACCOUNT_ID}:function:weather-agent-app \
  --query ApiId --output text)

echo "API ID: $API_ID"

# Grant API Gateway permission to invoke Lambda
aws lambda add-permission \
  --function-name weather-agent-app \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-1:${ACCOUNT_ID}:${API_ID}/*/*"

# Get API endpoint
API_ENDPOINT=$(aws apigatewayv2 get-api --api-id $API_ID --query ApiEndpoint --output text)
echo "Your API is available at: $API_ENDPOINT"
```

## Step 6: Test the Deployment

```bash
# Test the root endpoint
curl $API_ENDPOINT/

# Test the weather API
curl -X POST $API_ENDPOINT/api/weather \
  -H "Content-Type: application/json" \
  -d '{"city":"Pune"}'
```

## Step 7: Update Lambda Function (For Future Updates)

```bash
# Recreate ZIP with changes
cd lambda-deploy
zip -r ../weather-agent.zip .
cd ..

# Update function code
aws lambda update-function-code \
  --function-name weather-agent-app \
  --zip-file fileb://weather-agent.zip
```

## Step 8: Update Environment Variables

```bash
aws lambda update-function-configuration \
  --function-name weather-agent-app \
  --environment Variables={GEMINI_API_KEY=new_api_key_here}
```

## Cleanup (Optional)

```bash
# Delete API Gateway
aws apigatewayv2 delete-api --api-id $API_ID

# Delete Lambda function
aws lambda delete-function --function-name weather-agent-app

# Detach and delete IAM role
aws iam detach-role-policy \
  --role-name weather-agent-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam delete-role --role-name weather-agent-lambda-role

# Clean up local files
rm -rf lambda-deploy weather-agent.zip trust-policy.json
```

## Notes

- Default region is `us-east-1`. Change if needed using `--region` flag
- Lambda timeout is set to 30 seconds
- Memory is set to 256 MB
- The API Gateway automatically creates a default stage
- CORS is already configured in the application code

## Troubleshooting

**Check Lambda logs:**
```bash
aws logs tail /aws/lambda/weather-agent-app --follow
```

**Test Lambda directly:**
```bash
aws lambda invoke \
  --function-name weather-agent-app \
  --payload '{"rawPath":"/","requestContext":{"http":{"method":"GET"}}}' \
  response.json

cat response.json
```
