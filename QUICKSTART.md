# Quick Start Guide - LSI Dashboard

## 🚀 Launch in 30 Seconds

### Windows Users - Easiest Way:
```
Double-click: run_lsi_dashboard.bat
```
The dashboard will automatically open in your browser!

---

## Alternative: Command Line

### Option A - Using PowerShell:
```powershell
cd "C:\Users\UM_AS\OneDrive\Documents"
streamlit run lsi_streamlit_app.py
```

### Option B - Using Command Prompt:
```cmd
cd C:\Users\UM_AS\OneDrive\Documents
streamlit run lsi_streamlit_app.py
```

---

## 📍 Access Points

- **Local Computer**: http://localhost:8501
- **Mobile/Other Device**: http://<your-computer-ip>:8501

---

## 📄 Dashboard Pages

### 1️⃣ Dashboard (Overview)
- See system metrics
- View LSI trends
- Check data distribution

### 2️⃣ Predictions (Main Tool)
- Input water parameters
- Get instant LSI prediction
- See risk status (Scaling/Corrosion/Balanced)

### 3️⃣ Model Analysis
- Compare 3 AI models
- View accuracy metrics
- See feature importance (pH dominates!)

### 4️⃣ Historical Data
- Browse all records
- Filter by date range
- Download as CSV

---

## ⚡ Quick Test

1. Launch the app
2. Go to **Predictions** tab
3. Click **Predict LSI** button (use default values)
4. You'll see 3 predictions with confidence levels

---

## ❓ Troubleshooting

**Port in use?**
```
streamlit run lsi_streamlit_app.py --server.port 8502
```

**Data not found?**
Ensure this file exists:
```
C:\Users\UM_AS\Downloads\AI_LSI_Demo_Historical_Data.csv
```

**Slow loading?**
- Refresh browser (F5)
- Reduce data range slider

---

## 📊 What You'll See

✅ Real-time model predictions
✅ Interactive charts and graphs
✅ Performance metrics (R² = 0.9999)
✅ Feature importance analysis
✅ Historical trends
✅ Risk indicators

---

## 🛑 To Stop the App

Press `Ctrl+C` in the terminal window

---

**Ready? Let's go!** 🎉
