# Streamlit Cloud Deployment Guide

## Quick Setup (3 steps)

### Step 1: Push Code to GitHub
1. Create a GitHub account (free): https://github.com
2. Create a new repository called `blood-flow-simulator`
3. Upload these files to the repo:
   - `streamlit_app.py` (rename from `streamlit_app.py`)
   - `streamlit_requirements.txt` (rename to `requirements.txt`)

### Step 2: Deploy to Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click **"New App"**
3. Select:
   - Repository: `blood-flow-simulator`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Click **"Deploy"**

### Step 3: Share Your App
- Your app will be live at: `https://your-username-blood-flow-simulator.streamlit.app`
- Users can access it directly in their browser
- No installation needed on their end

---

## That's It! ✅

Your simulator is now:
- **Live on the internet**
- **Free hosting**
- **No server management needed**
- **Auto-updates when you push to GitHub**

---

## Run Locally (For Testing)

If you want to test before deploying:

```bash
pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py
```

Then visit: `http://localhost:8501`

---

## Features

✅ Interactive parameter sliders
✅ Real-time progress tracking
✅ Beautiful charts and visualizations
✅ Summary statistics
✅ Runs on free tier (generous limits)
✅ Mobile-friendly

---

## Limitations (Streamlit Cloud Free)

- App will sleep after 1 hour of inactivity (wakes up on next access)
- Limited computational resources (OK for this simulation)
- 1 GB storage, 3 apps per GitHub account

For more power, upgrade to paid tier ($5-$25/month)

---

## Troubleshooting

**"Module not found" error?**
- Streamlit Cloud auto-installs from requirements.txt
- Make sure filename is exactly `requirements.txt`

**App is slow?**
- First run takes ~30 seconds
- Streamlit caches results automatically
- Adjust grid resolution if needed

**Need custom domain?**
- Upgrade to paid Streamlit Cloud tier
- Or use: https://www.namecheap.com or similar to add custom domain pointing to Streamlit URL
