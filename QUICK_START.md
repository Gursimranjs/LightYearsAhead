# ⚡ Quick Start Guide

Get LightYears Ahead running in 3 minutes!

---

## 🚀 Step 1: Start Backend

**Terminal 1:**
```bash
cd "A world beyond/starsifter"
python backend_api.py
```

✅ Wait for: `Server running on: http://localhost:5001`

---

## 🎨 Step 2: Start Frontend

**Terminal 2:**
```bash
cd "A world beyond/frontend-react"
npm install  # First time only
npm run dev
```

✅ Wait for: `Local: http://localhost:3000/`

---

## 🌐 Step 3: Open Browser

Navigate to: **http://localhost:3000**

---

## 🎯 How to Use

1. **Home Page** - Click "Start Analysis"
2. **Upload CSV** - Drag & drop or browse for transit data
3. **Click "Analyze Data"** - AI processes your file
4. **View Results** - Interactive charts and detailed classifications
5. **Download Reports** - Individual PDFs or bulk ZIP

---

## 📊 Test Data

Example CSV files included:
```
starsifter/test_data/messy_csv_aliases.csv
working_test.csv
```

---

## 🆘 Troubleshooting

**Backend not starting?**
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost lightgbm imbalanced-learn pennylane fpdf
```

**Frontend not starting?**
```bash
cd frontend-react
rm -rf node_modules package-lock.json
npm install
```

**Ports in use?**
```bash
# Backend (5001)
lsof -ti:5001 | xargs kill -9

# Frontend (3000)
lsof -ti:3000 | xargs kill -9
```

---

## 📋 CSV Format

**Required columns (at least 7):**
- period, depth, duration, snr
- prad, teq, steff
- stellar_radius, stellar_mass, ror, insolation

**Example:**
```csv
target_name,period,depth,duration,snr,prad,teq,steff
Kepler-10b,0.84,0.015,2.1,25.3,1.4,1800,5600
```

System handles:
- ✅ Missing values
- ✅ Different column names
- ✅ Multiple targets
- ✅ Extra columns

---

## 🎉 That's It!

You're now analyzing exoplanets with AI! 🌌

For detailed documentation, see [README.md](README.md)

---

*LightYears Ahead - NASA Space Apps Challenge 2025*
