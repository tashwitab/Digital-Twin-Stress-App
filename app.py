import os
import time
from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
import joblib
import numpy as np
import random
from collections import Counter
from flask_sqlalchemy import SQLAlchemy  # <-- NEW: Replaces sqlite3
from sqlalchemy.sql import func       # <-- NEW: For analytics query

app = Flask(__name__)
# A secret key is still needed for basic session management
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_development_key_12345')
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# --- MODIFICATION 1: Setup Postgres Database ---
# Get the database URL from the environment variable (which you will set in Render)
db_uri = os.environ.get('DATABASE_URL')

# SQLAlchemy needs 'postgresql://' but Render's free tier gives 'postgres://'
# This simple line fixes it.
if db_uri and db_uri.startswith("postgres://"):
    db_uri = db_uri.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# --- End Modification 1 ---

# --- MODIFICATION 2: Define Database Models ---
# These classes replace your "CREATE TABLE" commands.
# SQLAlchemy uses them to understand your database structure.

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    # Define relationships for easy access
    predictions = db.relationship('Predictions', backref='user', lazy=True)
    habits = db.relationship('Habits', backref='user', lazy=True)
    journal_entries = db.relationship('Journal', backref='user', lazy=True)
    goal = db.relationship('Goals', backref='user', uselist=False, lazy=True)

class Predictions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stress_level = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)

class Habits(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    habit = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)
    stress_level = db.Column(db.Integer, nullable=False)

class Goals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    stress_goal = db.Column(db.Integer, nullable=False)
    habit_goal = db.Column(db.String(255), nullable=False)

class Journal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)

# This function creates the tables if they don't exist
# We will call this using the Render Build Command
def init_db():
    print("Initializing database...")
    with app.app_context():
        db.create_all()
    print("Database tables created or already exist.")
# --- End Modification 2 ---


# --- HTML Route (Unchanged) ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')


# --- 2. Load Model (Unchanged) ---
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

# --- 3. Authentication Endpoints (MODIFIED FOR SQLALCHEMY) ---

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400

    # Check if username already exists
    existing_user = Users.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'success': False, 'message': 'Username already exists'}), 409

    # Create new user
    new_user = Users(username=username, password=password)
    db.session.add(new_user)
    try:
        db.session.commit()
        print(f"User {username} (ID: {new_user.id}) registered.")
        return jsonify({'success': True, 'message': 'Registration successful', 'userId': new_user.id}), 201
    except Exception as e:
        db.session.rollback() # Roll back changes on error
        print(f"Error in /register: {e}")
        return jsonify({'success': False, 'message': 'Database error'}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400

    # Find user by username
    user = Users.query.filter_by(username=username).first()

    if user and user.password == password: # Plain text password check
        print(f"User {username} (ID: {user.id}) logged in.")
        return jsonify({'success': True, 'message': 'Login successful', 'userId': user.id}), 200
    else:
        print(f"Failed login attempt for username: {username}")
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

# --- 4. Main API Endpoints (MODIFIED FOR SQLALCHEMY) ---

def get_user_id_from_request(data):
    """Helper to get user_id from request JSON."""
    user_id = data.get('userId')
    if not user_id:
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
        new_prediction = Predictions(
            user_id=user_id, 
            stress_level=stress_level, 
            timestamp=int(time.time())
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({'stress_level': stress_level, 'stress_text': stress_text})
    except ValueError as e:
        print(f"Auth Error in /predict: {e}")
        return jsonify({'error': str(e)}), 401 # Send 401 if userId is missing
    except Exception as e:
        db.session.rollback()
        print(f"Error in /predict: {e}")
        return jsonify({'error': f'Prediction error: {e}'}), 500


@app.route('/history', methods=['POST'])
def get_history():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        
        # Query for history
        history_rows = Predictions.query.filter_by(user_id=user_id)\
            .order_by(Predictions.timestamp.desc())\
            .limit(30)\
            .all()
        
        history = [{'stress': row.stress_level, 'time': row.timestamp} for row in history_rows]
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
            
        # Log habit
        new_habit = Habits(
            user_id=user_id, 
            habit=habit, 
            timestamp=int(time.time()), 
            stress_level=stress_level
        )
        db.session.add(new_habit)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Habit logged.'})
    except ValueError as e:
        print(f"Auth Error in /log_habit: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        db.session.rollback()
        print(f"Error in /log_habit: {e}")
        return jsonify({'error': f'Habit logging error: {e}'}), 500


@app.route('/goals', methods=['POST'])
def manage_goals():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        
        # Find user's goal
        goal_row = Goals.query.filter_by(user_id=user_id).first()

        if goal_row:
            return jsonify({
                'stress_goal': goal_row.stress_goal,
                'habit_goal': goal_row.habit_goal
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

        # This is an "upsert" (Update or Insert)
        # Find existing goal
        goal = Goals.query.filter_by(user_id=user_id).first()
        
        if not goal:
            # Create new one if it doesn't exist
            goal = Goals(user_id=user_id)
            db.session.add(goal)
        
        # Update fields
        goal.stress_goal = stress_goal
        goal.habit_goal = habit_goal
        
        db.session.commit()
        return jsonify({'success': True, 'message': 'Goals updated.'})
    except ValueError as e:
        print(f"Auth Error in /set_goals: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        db.session.rollback()
        print(f"Error in /set_goals: {e}")
        return jsonify({'error': f'Set goal error: {e}'}), 500


@app.route('/report', methods=['POST'])
def get_report():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        
        # Get habit data
        habits_rows = Habits.query.filter_by(user_id=user_id).all()
        
        # Get goal data
        goal_row = Goals.query.filter_by(user_id=user_id).first()
        habit_goal = goal_row.habit_goal if goal_row else 'None'

        if not habits_rows:
            return jsonify({'report': 'Not enough data to generate a report. Start by logging habits.'})

        # Process report data (This logic is unchanged)
        habit_stress = {} # { 'habit': [stress1, stress2, ...] }
        habit_counts = Counter()
        
        for row in habits_rows:
            habit = row.habit
            stress = row.stress_level
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

        # Simple sentiment analysis placeholder (Unchanged)
        text_lower = text.lower()
        if any(word in text_lower for word in ['sad', 'angry', 'worried', 'stressed', 'anxious']):
            sentiment = 'Negative'
        elif any(word in text_lower for word in ['happy', 'great', 'good', 'calm', 'relaxed']):
            sentiment = 'Positive'
        else:
            sentiment = 'Neutral'
            
        # Save to database
        new_entry = Journal(
            user_id=user_id,
            text=text,
            sentiment=sentiment,
            timestamp=int(time.time())
        )
        db.session.add(new_entry)
        db.session.commit()

        return jsonify({'success': True, 'sentiment': sentiment})
    except ValueError as e:
        print(f"Auth Error in /journal: {e}")
        return jsonify({'error': str(e)}), 401
    except Exception as e:
        db.session.rollback()
        print(f"Error in /journal: {e}")
        return jsonify({'error': f'Journal error: {e}'}), 500


@app.route('/analytics', methods=['POST'])
def get_analytics():
    data = request.json
    try:
        user_id = get_user_id_from_request(data)
        
        # 1. Cognitive Load (based on prediction history)
        stress_levels = [row.stress_level for row in Predictions.query.filter_by(user_id=user_id).all()]
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
        # This is how you do a "GROUP BY" and "COUNT" in SQLAlchemy
        sentiment_rows = db.session.query(Journal.sentiment, func.count(Journal.sentiment))\
            .filter_by(user_id=user_id)\
            .group_by(Journal.sentiment)\
            .all()
        
        sentiment_trend = {sentiment: count for sentiment, count in sentiment_rows}

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
    # This function has no database logic, so it is unchanged.
    data = request.json
    try:
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
    # This function has no database logic, so it is unchanged.
    data = request.json
    try:
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


# --- 5. Run Application (Unchanged) ---
if __name__ == '__main__':
    # This init_db() will now call the new SQLAlchemy function
    init_db() 
    app.run(debug=True, port=5000)