import sqlite3
import os
import time
from flask import Flask, request, jsonify, session, render_template # Added render_template
from flask_cors import CORS
import joblib
import numpy as np
import random
from collections import Counter

app = Flask(__name__)
# A secret key is still needed for basic session management, but we won't use it for auth
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_development_key_12345')
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# --- MODIFICATION 1: Use persistent disk path on Render ---
# Use persistent disk path on Render, or local file for development
DB_PATH = os.environ.get('DB_PATH', 'digital_twin.db')
DB_NAME = DB_PATH
# --- End Modification 1 ---

# --- 1. Database Setup ---
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False) # Added check_same_thread=False for stability
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if os.path.exists(DB_NAME):
        print(f"Database '{DB_NAME}' already exists.")
        return # Database is ready

    print("First run: Creating database schema...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # User table
    cursor.execute('''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    
    # Predictions table
    cursor.execute('''
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        stress_level INTEGER NOT NULL,
        timestamp INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    # Habits table
    cursor.execute('''
    CREATE TABLE habits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        habit TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        stress_level INTEGER NOT NULL
    )''')
    
    # Goals table
    cursor.execute('''
    CREATE TABLE goals (
        user_id INTEGER PRIMARY KEY,
        stress_goal INTEGER NOT NULL,
        habit_goal TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    # Journal table
    cursor.execute('''
    CREATE TABLE journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    conn.commit()
    conn.close()
    print("Database schema created successfully.")

# --- MODIFICATION 2: Add route to serve the main HTML page ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')
# --- End Modification 2 ---


# --- 2. Load Model and Define Feature Order ---
try:
    model = joblib.load('stress_model.pkl')
except FileNotFoundError:
    print("\nError: 'stress_model.pkl' not found. Please run 'model_training.py' first.\n")
    exit()

FEATURE_ORDER = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]
stress_levels_map = {0: 'Low', 1: 'Medium', 2: 'High'}

# --- 3. Authentication Endpoints ---

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        user_id = cursor.lastrowid
        print(f"User {username} (ID: {user_id}) registered.")
        return jsonify({'success': True, 'message': 'Registration successful', 'userId': user_id}), 201
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Username already exists'}), 409
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if user and user['password'] == password: # Plain text password check
        print(f"User {username} (ID: {user['id']}) logged in.")
        return jsonify({'success': True, 'message': 'Login successful', 'userId': user['id']}), 200
    else:
        print(f"Failed login attempt for username: {username}")
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

# --- 4. Main API Endpoints (Now require userId) ---

def get_user_id_from_request(data):
    """Helper to get user_id from request JSON."""
    user_id = data.get('userId')
    if not user_id:
        # This is the "unauthorized" error, but it's explicit
        raise ValueError("Missing userId") 
    return int(user_id)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        features = [data['inputs'][feature] for feature in FEATURE_ORDER]
        prediction_array = np.array(features).reshape(1, -1)
        prediction = model.predict(prediction_array)
        stress_level = int(prediction[0])
        stress_text = stress_levels_map[stress_level]
        
        # Log prediction to database
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO predictions (user_id, stress_level, timestamp) VALUES (?, ?, ?)",
            (user_id, stress_level, int(time.time()))
        )
        conn.commit()
        conn.close()

        return jsonify({'stress_level': stress_level, 'stress_text': stress_text})
    except ValueError as e:
        print(f"Auth Error in /predict: {e}")
        return jsonify({'error': str(e)}), 401 # Send 401 if userId is missing
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': f'Prediction error: {e}'}), 500


@app.route('/history', methods=['POST'])
def get_history():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT stress_level, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 30",
            (user_id,)
        )
        history_rows = cursor.fetchall()
        conn.close()
        
        history = [{'stress': row['stress_level'], 'time': row['timestamp']} for row in history_rows]
        # Return reversed so chart plots oldest-to-newest
        return jsonify({'history': list(reversed(history))})
    except ValueError as e:
        print(f"Auth Error in /history: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /history: {e}")
        return jsonify({'error': f'History error: {e}'}), 500


@app.route('/log_habit', methods=['POST'])
def log_habit():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        habit = data.get('habit')
        stress_level = data.get('stress_level') # Get current stress level

        if not habit or stress_level is None:
            return jsonify({'error': 'Habit and stress_level are required'}), 400
            
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO habits (user_id, habit, timestamp, stress_level) VALUES (?, ?, ?, ?)",
            (user_id, habit, int(time.time()), stress_level)
        )
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Habit logged.'})
    except ValueError as e:
        print(f"Auth Error in /log_habit: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /log_habit: {e}")
        return jsonify({'error': f'Habit logging error: {e}'}), 500


@app.route('/goals', methods=['POST'])
def manage_goals():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT stress_goal, habit_goal FROM goals WHERE user_id = ?", (user_id,))
        goal_row = cursor.fetchone()
        conn.close()

        if goal_row:
            return jsonify({
                'stress_goal': goal_row['stress_goal'],
                'habit_goal': goal_row['habit_goal']
            })
        else:
            # Return defaults if no goals are set
            return jsonify({
                'stress_goal': 1, # Default
                'habit_goal': 'Exercise' # Default
            })
    except ValueError as e:
        print(f"Auth Error in /goals: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /goals: {e}")
        return jsonify({'error': f'Goal error: {e}'}), 500


@app.route('/set_goals', methods=['POST'])
def set_goals():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        stress_goal = data.get('stress_goal')
        habit_goal = data.get('habit_goal')

        if stress_goal is None or habit_goal is None:
             return jsonify({'error': 'stress_goal and habit_goal are required'}), 400

        conn = get_db_connection()
        conn.execute(
            "INSERT OR REPLACE INTO goals (user_id, stress_goal, habit_goal) VALUES (?, ?, ?)",
            (user_id, stress_goal, habit_goal)
        )
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Goals updated.'})
    except ValueError as e:
        print(f"Auth Error in /set_goals: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /set_goals: {e}")
        return jsonify({'error': f'Set goal error: {e}'}), 500


@app.route('/report', methods=['POST'])
def get_report():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get habit data
        cursor.execute("SELECT habit, stress_level FROM habits WHERE user_id = ?", (user_id,))
        habits_rows = cursor.fetchall()
        
        # Get goal data
        cursor.execute("SELECT habit_goal FROM goals WHERE user_id = ?", (user_id,))
        goal_row = cursor.fetchone()
        habit_goal = goal_row['habit_goal'] if goal_row else 'None'
        conn.close()

        if not habits_rows:
            return jsonify({'report': 'Not enough data to generate a report. Start by logging habits.'})

        # Process report data
        habit_stress = {} # { 'habit': [stress1, stress2, ...] }
        habit_counts = Counter()
        
        for row in habits_rows:
            habit = row['habit']
            stress = row['stress_level']
            habit_counts[habit] += 1
            if habit not in habit_stress:
                habit_stress[habit] = []
            habit_stress[habit].append(stress)

        report_lines = ["--- Stress & Habit Analysis ---"]
        for habit, levels in habit_stress.items():
            if levels:
                avg_stress = sum(levels) / len(levels)
                avg_stress_text = stress_levels_map.get(round(avg_stress), "N/A")
                report_lines.append(f"Habit: '{habit}' (Logged {habit_counts[habit]} times)")
                report_lines.append(f"  - Avg. Stress Level: {avg_stress:.2f} ({avg_stress_text})")
        
        # Goal progress
        goal_count = habit_counts.get(habit_goal, 0)
        report_lines.append("\n--- Wellness Goal Progress ---")
        report_lines.append(f"Your Goal: '{habit_goal}'")
        report_lines.append(f"Progress: You have logged this habit {goal_count} times.")

        return jsonify({'report': '\n'.join(report_lines)})
    except ValueError as e:
        print(f"Auth Error in /report: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /report: {e}")
        return jsonify({'error': f'Report error: {e}'}), 500


@app.route('/journal', methods=['POST'])
def save_journal():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Journal text is required'}), 400

        # Simple sentiment analysis placeholder
        text_lower = text.lower()
        if any(word in text_lower for word in ['sad', 'angry', 'worried', 'stressed', 'anxious']):
            sentiment = 'Negative'
        elif any(word in text_lower for word in ['happy', 'great', 'good', 'calm', 'relaxed']):
            sentiment = 'Positive'
        else:
            sentiment = 'Neutral'
            
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO journal (user_id, text, sentiment, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, text, sentiment, int(time.time()))
        )
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'sentiment': sentiment})
    except ValueError as e:
        print(f"Auth Error in /journal: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /journal: {e}")
        return jsonify({'error': f'Journal error: {e}'}), 500


@app.route('/analytics', methods=['POST'])
def get_analytics():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Cognitive Load (based on prediction history)
        cursor.execute("SELECT stress_level FROM predictions WHERE user_id = ?", (user_id,))
        stress_levels = [row['stress_level'] for row in cursor.fetchall()]
        if stress_levels:
            avg_stress = sum(stress_levels) / len(stress_levels)
            if avg_stress > 1.5:
                cognitive_load = "High"
            elif avg_stress > 0.8:
                cognitive_load = "Medium"
            else:
                cognitive_load = "Low"
        else:
            cognitive_load = "N/A"

        # 2. Sentiment Trend (from journal)
        cursor.execute("SELECT sentiment, COUNT(*) as count FROM journal WHERE user_id = ? GROUP BY sentiment", (user_id,))
        sentiment_rows = cursor.fetchall()
        conn.close()
        
        sentiment_trend = {row['sentiment']: row['count'] for row in sentiment_rows}

        return jsonify({
            'cognitive_load': cognitive_load,
            'sentiment_trend': sentiment_trend
        })
    except ValueError as e:
        print(f"Auth Error in /analytics: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /analytics: {e}")
        return jsonify({'error': f'Analytics error: {e}'}), 500


@app.route('/interventions', methods=['POST'])
def get_interventions():
    data = request.json
    try:
        # This one doesn't strictly need a userId, but we check for it
        # to ensure the frontend is sending it correctly.
        get_user_id_from_request(data) 
        stress_level = data.get('stress_level') # Current stress level
        
        interventions = {
            0: [ # Low
                "Great job managing your stress!",
                "Try a 5-minute gratitude journal.",
                "Plan a social activity with friends."
            ],
            1: [ # Medium
                "Take a 10-minute walk outside.",
                "Practice 5 minutes of box breathing (inhale 4s, hold 4s, exhale 4s, hold 4s).",
                "Listen to a calming music playlist."
            ],
            2: [ # High
                "Stop and take 5 deep, slow breaths immediately.",
                "Try a 10-minute guided meditation for stress.",
                "Consider taking a short break from your current task."
            ]
        }
        
        suggestions = interventions.get(stress_level, interventions[1]) # Default to medium
        return jsonify({'suggestions': random.sample(suggestions, 2)}) # Give 2 random suggestions
    except ValueError as e:
        print(f"Auth Error in /interventions: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /interventions: {e}")
        return jsonify({'error': f'Intervention error: {e}'}), 500


@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    try:
        # Check for user ID
        get_user_id_from_request(data)
        
        inputs = data.get('inputs')
        change = data.get('change') # e.g., {'study_load': 5}

        if not inputs or not change:
            return jsonify({'error': 'inputs and change are required'}), 400

        # Create a copy of current inputs and apply the change
        simulated_inputs = inputs.copy()
        simulated_inputs.update(change)
        
        # Get original prediction
        original_features = [inputs[feature] for feature in FEATURE_ORDER]
        original_pred = model.predict(np.array(original_features).reshape(1, -1))[0]
        original_text = stress_levels_map[int(original_pred)]
        
        # Get simulated prediction
        simulated_features = [simulated_inputs[feature] for feature in FEATURE_ORDER]
        simulated_pred = model.predict(np.array(simulated_features).reshape(1, -1))[0]
        simulated_text = stress_levels_map[int(simulated_pred)]

        change_key = list(change.keys())[0]
        change_val = list(change.values())[0]

        return jsonify({
            'original_stress': original_text,
            'simulated_stress': simulated_text,
            'simulation_summary': f"If '{change_key}' changes to '{change_val}', your stress may change from {original_text} to {simulated_text}."
        })
    except ValueError as e:
        print(f"Auth Error in /simulate: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        print(f"Error in /simulate: {e}")
        return jsonify({'error': f'Simulation error: {e}'}), 500


# --- 5. Run Application ---
# --- MODIFICATION 3: This block MUST be here for Render ---
# It ensures app.run() only starts in local development
# and not when Render's Gunicorn server imports the file.
if __name__ == '__main__':
    init_db() # Ensure database exists and is set up
    app.run(debug=True, port=5000)
# --- End Modification 3 ---