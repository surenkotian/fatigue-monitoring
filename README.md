## 🧠 Advanced Employee Fatigue Monitoring System

A Streamlit-based dashboard and detection system for monitoring employee fatigue using facial metrics like **EAR (Eye Aspect Ratio)** and **MAR (Mouth Aspect Ratio)**. Designed for remote work environments to boost health awareness, productivity, and wellness.

---

### 📦 Features

- ✅ Real-time fatigue detection using EAR & MAR
- 📊 Streamlit dashboard for live and historical analytics
- 📈 Fatigue trends, efficiency breakdowns, heatmaps
- 💡 Personalized recommendations and alerts
- 🔐 User authentication and admin panel
- 🧪 Dummy data fallback if dataset is missing
- 🐳 Docker-compatible deployment
- 💾 Data export in CSV and Excel formats

---

### 🗂️ Project Structure

```
fatigue_monitoring_package/
├── client/                  # Eye/mouth detection script (EXE build ready)
│   └── fatigue_detector.py
├── server/                  # Streamlit dashboard
│   ├── dashboard.py
│   ├── Dockerfile
│   └── requirements.txt
├── model/
│   └── fatigue_model.pkl
├── config.yaml              # Credentials and thresholds
└── wfh_fatigue_data.csv     # Sample or real fatigue dataset
```

---

### 🚀 How to Run the Dashboard

#### ▶️ Option 1: Run Locally

```bash
cd server
pip install -r requirements.txt
streamlit run dashboard.py
```

> Make sure `wfh_fatigue_data.csv` is in the same folder as `dashboard.py`.

---

#### 🐳 Option 2: Run with Docker

```bash
cd server
docker build -t fatigue-dashboard .
docker run -p 8501:8501 fatigue-dashboard
```

Then open: [http://localhost:8501](http://localhost:8501)

---

### 📈 Key Metrics Tracked

- EAR (Eye Aspect Ratio) – low values suggest eye closure
- MAR (Mouth Aspect Ratio) – high values suggest yawning
- Blink & yawn trends
- Efficiency score (calculated using EAR/MAR)
- State classification: Alert, Yawning, Drowsy
- Recovery time and fatigue transition

---

### 🛡️ Authentication

- Admin & user roles managed via `config.yaml`
- Supports registration (admin-approved)

---

### 📤 Export Features

- 📁 CSV and Excel downloads
- 📊 Embedded charts in Excel reports
- 📄 Summary and analytics report generation

---

### 📌 TODOs / Future Enhancements

- [ ] Deploy to Streamlit Cloud (or use Docker on cloud)
- [ ] Live webcam input integration
- [ ] Notification/Slack alert integration
- [ ] Scheduling & session tracking

---

### 🙌 Author

**Suren Kotian**  
Fatigue detection system powered by facial aspect ratios and Streamlit analytics.
