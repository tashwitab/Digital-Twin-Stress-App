# Digital Twin - Stress Monitor

![Render](https://img.shields.io/badge/Render-deployed-46E3B7?style=for-the-badge&logo=render)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

A Flask and ML web application for predicting, tracking, and managing personal stress levels.

## 🚀 Live Demo

**View the live application here:**

**https://digital-twin-stress-app.onrender.com/#**



---

## 📸 Dashboard Preview

![A preview of the Digital Twin dashboard, showing stress level, history chart, and analytics.]([...PASTE YOUR SCREENSHOT URL HERE...])

*(**How to add a screenshot:** 1. Take a screenshot of your live app. 2. On your GitHub repo, click the "Issues" tab, then "New Issue". 3. Drag and drop your screenshot into the comment box. 4. Copy the URL it generates and paste it above, then cancel the issue.)*

---

## 📖 About The Project

This project is a web-based "digital twin" designed to help users monitor their mental well-being. By inputting 20 behavioral and environmental factors (based on the [Stress Level Dataset from Kaggle](https://www.kaggle.com/datasets/rxanthony/stress-level-dataset)), a pre-trained machine learning model predicts the user's current stress level (Low, Medium, or High).

The dashboard provides a comprehensive suite of tools to visualize, track, and manage stress over time, all linked to a persistent cloud database.

## ✨ Features

* **ML-Powered Prediction:** Predicts stress levels using a pre-trained Random Forest model.
* **Secure Authentication:** User registration and login system with persistent session data.
* **Interactive Dashboard:** A single-page application to visualize all personal data.
* **Stress History:** A dynamic Chart.js line graph showing the user's last 30 stress predictions.
* **Sentiment Journal:** Log daily thoughts and receive simple sentiment analysis (Positive, Neutral, Negative).
* **Habit & Goal Tracking:** Users can log daily habits (e.g., "Exercise") and track progress toward a personal goal.
* **"What-If" Simulation:** A tool to change input variables (e.g., "What if I get more sleep?") to see the predicted impact on stress.
* **Personalized Interventions:** Receive actionable advice based on the current stress level.
* **Behavioral Analytics:** View a summary of average cognitive load and sentiment trends.

## 🛠️ Tech Stack

* **Backend:** **Flask** (Python)
* **Database:** **PostgreSQL** (Managed by **SQLAlchemy** on Render's free tier)
* **Machine Learning:** **Scikit-learn** (RandomForestClassifier), **Joblib**, **Pandas**, **Numpy**
* **Frontend:** **HTML**, **Tailwind CSS**, **Chart.js**
* **Deployment:** **Render** (Free Web Service + Free Postgres)

---

## 📦 Running the Project Locally

To run this project on your local machine, follow these steps.

**1. Clone the Repository**
```bash
git clone [https://github.com/](https://github.com/)[...PASTE YOUR GITHUB USERNAME/REPONAME...].git
cd digital-twin-stress-app

2. Create and Activate a Virtual Environment

Bash

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies

Bash

pip install -r requirements.txt
4. Set Environment Variables This project requires a PostgreSQL database and a secret key. You can use the same free database you created on Render.

Find your External Connection String from your Render database.

Find your FLASK_SECRET_KEY from your Render Web Service environment.

Set these in your terminal:

Bash

# Windows (PowerShell)
$env:FLASK_SECRET_KEY = "your_secret_key_from_render"
$env:DATABASE_URL = "postgres://user:pass@host.com/db_name"

# macOS / Linux
export FLASK_SECRET_KEY="your_secret_key_from_render"
export DATABASE_URL="postgres://user:pass@host.com/db_name"
(Remember to replace the postgres:// prefix with postgresql:// in the DATABASE_URL)

5. Initialize the Database (You only need to run this once for a new database)

Bash

python -c "from app import init_db; init_db()"
Output: Initializing database... Database tables created or already exist.

6. Run the Application

Bash

flask run
Your app will be running at http://127.0.0.1:5000.
