# AWS Lambda Deployment via Console

## Step 1: Prepare Deployment Package

```bash
cd /home/virtualuser/gen_ai/ai-agents/weather
npm install --production
zip -r weather-agent.zip node_modules index.js public
```

## Step 2: Create Lambda Function

1. Go to **AWS Console** → **Lambda** → **Create function**
2. Choose **Author from scratch**
3. Configure:
   - **Function name:** `weather-agent-app`
   - **Runtime:** Node.js 20.x
   - **Architecture:** x86_64
4. Click **Create function**

## Step 3: Upload Code

1. In the function page, go to **Code** tab
2. Click **Upload from** → **.zip file**
3. Upload `weather-agent.zip`
4. Click **Save**

## Step 4: Configure Function

1. Go to **Configuration** tab → **General configuration** → **Edit**
   - **Timeout:** 30 seconds
   - **Memory:** 256 MB
   - Click **Save**

2. Go to **Configuration** tab → **Environment variables** → **Edit**
   - Click **Add environment variable**
   - **Key:** `GEMINI_API_KEY`
   - **Value:** Your Gemini API key
   - Click **Save**

## Step 5: Create API Gateway

1. Go to **API Gateway** console
2. Click **Create API**
3. Choose **HTTP API** → **Build**
4. Configure:
   - **API name:** `weather-agent-api`
   - Click **Add integration** → **Lambda**
   - Select region and function: `weather-agent-app`
   - Click **Next**

5. Configure routes:
   - **Method:** ANY
   - **Resource path:** `/{proxy+}`
   - Click **Next**

6. Configure stages:
   - **Stage name:** `$default` (auto-configured)
   - Click **Next**

7. Review and click **Create**

## Step 6: Get API URL

1. In API Gateway console, select your API
2. Copy the **Invoke URL** (e.g., `https://abc123.execute-api.us-east-1.amazonaws.com`)
3. Your app is live at this URL

## Step 7: Test

Open browser and visit:
- `https://YOUR_API_URL/` - Frontend
- Test weather API with curl:

```bash
curl -X POST https://YOUR_API_URL/api/weather \
  -H "Content-Type: application/json" \
  -d '{"city":"Pune"}'
```

## Update Code (Future Changes)

1. Create new ZIP file with changes
2. Go to Lambda function → **Code** tab
3. **Upload from** → **.zip file**
4. Upload new ZIP
5. Click **Save**

## Monitor Logs

1. Go to Lambda function → **Monitor** tab
2. Click **View CloudWatch logs**
3. Select latest log stream to see execution logs

## Cleanup

1. Delete API Gateway: API Gateway console → Select API → **Actions** → **Delete**
2. Delete Lambda: Lambda console → Select function → **Actions** → **Delete**

## Notes

- API Gateway automatically handles CORS (configured in code)
- Lambda cold starts may cause first request to be slow
- Free tier: 1M requests/month + 400,000 GB-seconds compute
