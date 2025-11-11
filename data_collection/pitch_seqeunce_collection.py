from pybaseball import statcast
import pandas as pd

# Define date range for 2025 season
START_DATE = "2025-03-17"
END_DATE = "2025-09-28"

print("Downloading Statcast pitch-by-pitch data...")
data = statcast(start_dt=START_DATE, end_dt=END_DATE)
print(f"Retrieved {len(data):,} pitches.")

# Columns relevant to sequencing and context
cols = [
    "game_date", "game_pk", "at_bat_number", "pitch_number",
    "pitcher", "batter",
    "balls", "strikes",                     # <-- count context
    "pitch_type", "release_speed",
    "release_spin_rate", "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "events", "description"
]

# Keep only columns that exist in the data
cols = [c for c in cols if c in data.columns]
pitch_data = data[cols].copy()

# Sort by order of play
pitch_data = pitch_data.sort_values(
    ["game_pk", "at_bat_number", "pitch_number"]
).reset_index(drop=True)

# Create next pitch label
group_cols = ["game_pk", "pitcher", "batter"]
pitch_data["next_pitch_type"] = (
    pitch_data.groupby(group_cols)["pitch_type"].shift(-1)
)

# Drop final pitches of each at-bat (no next pitch)
pitch_data = pitch_data.dropna(subset=["next_pitch_type"])

# Create numeric count feature
pitch_data["count_state"] = pitch_data["balls"] * 10 + pitch_data["strikes"]

# Save to file
pitch_data.to_csv("data/pitch_sequence_2025.csv", index=False)

print(f"Saved {len(pitch_data):,} rows × {len(pitch_data.columns)} columns to data/pitch_sequence_2025.csv")
