import matplotlib.pyplot as plt
import numpy as np

print("--- Generating Simulated Time-Series Plot (Figure 5.3) ---")

# --- 1. Create Simulated Data ---

# Create a 4-hour time axis (4 * 60 = 240 minutes)
time_minutes = np.linspace(0, 240, 480)

# Generate baseline "predicted stress" (low-level noise)
baseline = np.random.rand(480) * 0.15 + 0.1
predicted_stress = baseline

# --- 2. Create the Predictive Spike ---
# This is a helper function to create a smooth rise
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Event 1: Logged at 11:00 AM (1 hour in, or 60 minutes)
# The prediction will start rising *before* this, at minute 45
event_time_1 = 60
prediction_rise_time_1 = 45
spike_1 = gaussian(time_minutes, event_time_1, 10) * 0.7
predicted_stress[prediction_rise_time_1:event_time_1 + 30] += spike_1[prediction_rise_time_1:event_time_1 + 30]

# Event 2: Logged at 1:30 PM (3.5 hours in, or 210 minutes)
# The prediction will start rising *before* this, at minute 195
event_time_2 = 210
prediction_rise_time_2 = 195
spike_2 = gaussian(time_minutes, event_time_2, 12) * 0.8
predicted_stress[prediction_rise_time_2:event_time_2 + 30] += spike_2[prediction_rise_time_2:event_time_2 + 30]

# Clip values to be between 0.0 and 1.0
predicted_stress = np.clip(predicted_stress, 0, 1)

# Define the actual stress log events (the red dots)
actual_stress_times = [event_time_1, event_time_2]

# --- START OF FIX ---
# Find the *closest* indices in the time_minutes array
# We can't use == for floating point numbers
index_1 = np.argmin(np.abs(time_minutes - event_time_1))
index_2 = np.argmin(np.abs(time_minutes - event_time_2))

actual_stress_values = [
    predicted_stress[index_1],
    predicted_stress[index_2]
]
# --- END OF FIX ---

# --- 3. Plot the Figure ---
plt.figure(figsize=(12, 6))
plt.plot(time_minutes, predicted_stress, label='Predicted Stress (Model)', color='blue', linewidth=2)
plt.scatter(actual_stress_times, actual_stress_values, color='red', s=100, zorder=5, label='Actual Stress Log (User)')

# --- 4. Styling (to match your description) ---
plt.title('Model Prediction vs. Actual Stress Logs (Figure 5.3)', fontsize=16)
plt.ylabel('Stress Probability', fontsize=12)
plt.ylim(0, 1.1)
plt.xlabel('Time', fontsize=12)
plt.xlim(0, 240)
plt.xticks(
    ticks=[0, 60, 120, 180, 240],
    labels=['10:00 AM', '11:00 AM', '12:00 PM', '1:00 PM', '2:00 PM']
)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figure
plt.savefig('timeseries_plot.png')
print("Plot 'timeseries_plot.png' (Figure 5.3) saved successfully.")
plt.close()

print("--- Plot Generation Complete ---")

