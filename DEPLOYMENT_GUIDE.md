# Blood Flow Simulation - Deployment Guide for cPanel

## Overview
This is a web-based blood flow simulator that runs on Flask and displays interactive cardiovascular hemodynamics simulations.

## Prerequisites
- Python 3.7+ installed on your hosting server
- cPanel access with Python application support
- pip (Python package manager)

---

## Deployment Steps for cPanel

### Step 1: Upload Files to cPanel
1. Open **File Manager** in cPanel
2. Navigate to `/public_html` directory
3. Create a new folder: `blood_simulation`
4. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `templates/` folder (with `index.html`)

### Step 2: Set Up Python Environment in cPanel

#### Option A: Using cPanel's "Setup Python App"
1. Go to cPanel → **Software** → **Setup Python App**
2. Click **Create Application**
3. Select Python version (3.9+ recommended)
4. Set:
   - **Application Path**: `/blood_simulation`
   - **Application Root**: `/public_html/blood_simulation`
   - **Application URL**: Your domain URL
   - **Application Startup File**: `app.py`
   - **Application Entry Point**: `app`
5. Click **Create**

#### Option B: Manual Setup with Passenger
1. SSH into your server (cPanel → **Terminal**)
2. Navigate to your app directory:
   ```bash
   cd /home/username/public_html/blood_simulation
   ```
3. Create virtual environment:
   ```bash
   python3 -m venv venv
   ```
4. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Create `passenger_wsgi.py`:
   ```python
   import sys
   from app import app
   application = app
   ```

### Step 3: Configure cPanel Settings
1. Go to **Setup Python App** → Your app
2. Click **Edit**
3. Set **Passenger Log** to debug any issues
4. Restart application

### Step 4: Verify Deployment
- Visit your app URL in a browser
- Try running a simulation with default parameters
- Check logs if there are errors

---

## Alternative: Quick CGI Deployment (No Framework)

If cPanel doesn't support Python apps well, create a simple CGI script instead:

1. Upload scripts to `/public_html/simulations/`
2. Make them executable: `chmod 755 simulation_t.py`
3. Create an HTML form that calls the scripts via CGI
4. Results saved as image files users can download

---

## Troubleshooting

### "Module not found" errors
- SSH and run: `pip install -r requirements.txt` in the venv

### "Permission denied" errors
- Run: `chmod 755 app.py` and `chmod 755 -R templates/`

### Port already in use
- cPanel automatically manages ports, ensure setup is complete

### Matplotlib not working
- Install: `pip install python-dateutil pytz`

---

## Performance Notes
- First simulation may take 10-30 seconds
- Results are cached on the server temporarily
- For high traffic, consider adding result caching or queue system

---

## Support
For cPanel-specific issues, contact your hosting provider's support team.
