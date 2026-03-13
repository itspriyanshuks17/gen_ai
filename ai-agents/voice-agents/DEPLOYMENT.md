# Deploying AI Telephony Agent to Cloud

## Why Deploy?

WSL has networking limitations that prevent WebRTC audio streaming. For production telephony, deploy to a cloud server with a public IP.

## Option 1: AWS EC2 (Recommended)

### 1. Launch EC2 Instance
```bash
# Instance type: t3.small or larger
# OS: Ubuntu 22.04 LTS
# Security Group: Allow inbound TCP 8081
```

### 2. Connect and Setup
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo apt install python3.12 python3.12-venv python3-pip -y

# Clone or upload your project
mkdir voice-agent && cd voice-agent
```

### 3. Upload Files
```bash
# From your local machine
scp -i your-key.pem -r ~/gen_ai/ai-agents/voice-agents/* ubuntu@your-ec2-ip:~/voice-agent/
```

### 4. Install and Run
```bash
cd ~/voice-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run agent
python3 main.py
```

### 5. Run as Service (Production)
```bash
sudo nano /etc/systemd/system/voice-agent.service
```

Add:
```ini
[Unit]
Description=VideoSDK Voice Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/voice-agent
Environment="PATH=/home/ubuntu/voice-agent/.venv/bin"
ExecStart=/home/ubuntu/voice-agent/.venv/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-agent
sudo systemctl start voice-agent
sudo systemctl status voice-agent
```

## Option 2: DigitalOcean Droplet

### 1. Create Droplet
- Choose Ubuntu 22.04
- Basic plan ($6/month)
- Add SSH key

### 2. Follow same steps as AWS EC2

## Option 3: Google Cloud VM

### 1. Create VM Instance
```bash
gcloud compute instances create voice-agent \
  --machine-type=e2-small \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB
```

### 2. Follow same setup steps

## Option 4: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t voice-agent .
docker run -d --env-file .env -p 8081:8081 voice-agent
```

## Testing After Deployment

1. Verify agent is running:
```bash
curl http://your-server-ip:8081/health
```

2. Check logs:
```bash
# If using systemd
sudo journalctl -u voice-agent -f

# If running directly
tail -f logs/agent.log
```

3. Make test call:
```bash
python3 make_call.py
```

## Troubleshooting

### Agent not reachable
- Check firewall: `sudo ufw allow 8081`
- Verify security group allows TCP 8081
- Check agent is listening on 0.0.0.0, not localhost

### Audio issues
- Ensure server has good network connectivity
- Check CPU/memory usage
- Verify Google API key is valid

### Call not connecting
- Verify agent_id matches routing rule
- Check VideoSDK dashboard for agent status
- Review agent logs for errors

## Cost Estimates

- **AWS EC2 t3.small**: ~$15/month
- **DigitalOcean Basic**: $6/month
- **Google Cloud e2-small**: ~$13/month

## Next Steps

After deployment:
1. Update DNS if needed
2. Set up monitoring (CloudWatch, Datadog)
3. Configure auto-restart on failure
4. Set up log rotation
5. Add SSL/TLS if exposing HTTP endpoints
