# 🚀 Deployment Guide

## Quick Start

### Local Testing (Ollama)
```bash
# No setup needed! Just run:
python unified_server.py

# Visit: http://localhost:8000
```

### Local Testing with Groq (Optional)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get Groq API key from: https://console.groq.com
# 3. Set environment variable
set GROQ_API_KEY=your_api_key_here  # Windows
# OR
export GROQ_API_KEY=your_api_key_here  # Mac/Linux

# 4. Run server
python unified_server.py

# Visit: http://localhost:8000
```

---

## 🌐 Cloud Deployment Options

### Option 1: Railway (Recommended)

**Step 1: Get Groq API Key**
1. Visit: https://console.groq.com
2. Sign up (free)
3. Create API Key
4. Copy the key (starts with `gsk_...`)

**Step 2: Deploy to Railway**

**Method A - GitHub (Easier):**
```bash
# 1. Push code to GitHub
git add .
git commit -m "Add Railway deployment support"
git push origin main

# 2. Railway Setup:
# - Visit: https://railway.app
# - Login with GitHub
# - New Project → Deploy from GitHub
# - Select your repository
```

**Method B - Railway CLI:**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up
```

**Step 3: Set Environment Variables**
```
In Railway Dashboard → Variables:

GROQ_API_KEY = gsk_your_api_key_here
```

**Step 4: Done!** 🎉
Railway will provide a public URL like: `https://your-app.railway.app`

---

### Option 2: Render.com (Forever Free)

**Step 1: Get Groq API Key** (same as Railway)

**Step 2: Deploy**
1. Visit: https://render.com
2. Sign up with GitHub (no CC required!)
3. New → Web Service
4. Connect your GitHub repository
5. Settings:
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python unified_server.py`
6. Add Environment Variable:
   - Key: `GROQ_API_KEY`
   - Value: `gsk_your_api_key_here`
7. Deploy!

---

## 📊 Comparison

| Platform | Free? | Setup Time | CC Required? |
|----------|-------|------------|--------------|
| **Local** | ✅ | 1 min | ❌ |
| **Railway** | Trial | 10 min | Eventually |
| **Render** | ✅ Forever | 15 min | ❌ |

---

## 🔧 How It Works

### Automatic Model Selection

The app automatically detects the environment:

**Local Development:**
```
No GROQ_API_KEY → Uses Ollama (localhost:11434)
✓ Ollama must be running
✓ Model: llama2:latest
```

**Cloud Deployment:**
```
GROQ_API_KEY present → Uses Groq API
✓ No local resources needed
✓ Model: llama-3.1-70b-versatile (better & faster!)
```

---

## 🗄️ Database Options

### Local Development
```python
# Uses localhost MySQL automatically
# No configuration needed
```

### Cloud with External MySQL
```
Set these environment variables:

DB_HOST = your_mysql_host
DB_USER = your_mysql_user
DB_PASSWORD = your_mysql_password
DB_NAME = case_studies_db
```

---

## ❓ Troubleshooting

### Groq API Error
```
Error: Groq client not initialized
```
**Solution:** Make sure `GROQ_API_KEY` environment variable is set

### Ollama Error (Local)
```
Error: Cannot connect to Ollama service
```
**Solution:** Start Ollama: `ollama serve`

### Database Error
```
Error: Can't connect to MySQL server
```
**Solution:**
- Local: Make sure MySQL is running
- Cloud: Check DB_* environment variables

---

## 📞 Support

- Groq API Docs: https://console.groq.com/docs
- Railway Docs: https://docs.railway.app
- Render Docs: https://render.com/docs

---

## 🎯 Performance Comparison

| Metric | Ollama (Local) | Groq (Cloud) |
|--------|---------------|--------------|
| **Speed** | 10-15s | 1-2s ⚡ |
| **Model** | llama2:7B | llama-3.1:70B |
| **Quality** | Good | Excellent ⭐ |
| **Cost** | Free | Free (trial) |
| **24/7** | No (PC must be on) | Yes ✅ |

---

Made with ❤️ for Dr. Robert Young's Semantic Search
