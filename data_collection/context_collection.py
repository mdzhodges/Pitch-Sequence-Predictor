from pybaseball import statcast
import pandas as pd

# Define the date range for the 2025 season
START_DATE = "2025-03-17"
END_DATE = "2025-09-28"

print("Downloading Statcast data (this may take a while)...")
data = statcast(start_dt=START_DATE, end_dt=END_DATE)

print(f"Retrieved {len(data):,} pitches with {len(data.columns)} total columns.")

# Columns to extract (based on 2025 Statcast schema)
wanted_cols = [
    'age_bat', 'age_bat_legacy', 'age_pit', 'age_pit_legacy',
    'api_break_x_arm', 'api_break_x_batter_in', 'api_break_z_with_gravity',
    'arm_angle', 'at_bat_number', 'attack_angle', 'attack_direction',
    'away_score', 'ax', 'ay', 'az', 'babip_value', 'balls', 'bat_score',
    'bat_score_diff', 'bat_speed', 'bat_win_exp', 'batter', 'bb_type',
    'break_angle_deprecated', 'break_length_deprecated',
    'delta_home_win_exp', 'delta_pitcher_run_exp', 'delta_run_exp', 'des',
    'effective_speed', 'estimated_ba_using_speedangle',
    'estimated_slg_using_speedangle', 'estimated_woba_using_speedangle',
    'events', 'fld_score', 'hc_x', 'hc_y', 'hit_distance_sc',
    'hit_location', 'home_score', 'home_score_diff', 'home_team',
    'home_win_exp', 'hyper_speed', 'if_fielding_alignment', 'inning',
    'inning_topbot', 'intercept_ball_minus_batter_pos_x_inches',
    'intercept_ball_minus_batter_pos_y_inches', 'iso_value', 'launch_angle',
    'launch_speed', 'launch_speed_angle', 'n_priorpa_thisgame_player_at_bat',
    'n_thruorder_pitcher', 'of_fielding_alignment', 'on_1b', 'on_2b', 'on_3b',
    'outs_when_up', 'p_throws', 'pfx_x', 'pfx_z', 'pitch_name',
    'pitch_number', 'pitch_type', 'pitcher', 'pitcher_days_since_prev_game',
    'pitcher_days_until_next_game', 'plate_x', 'plate_z', 'player_name',
    'post_away_score', 'post_bat_score', 'post_fld_score', 'post_home_score',
    'release_extension', 'release_pos_x', 'release_pos_y', 'release_pos_z',
    'release_speed', 'release_spin_rate', 'spin_axis', 'spin_dir', 'stand',
    'strikes', 'sv_id', 'swing_length', 'swing_path_tilt', 'sz_bot', 'sz_top',
    'type', 'vx0', 'vy0', 'vz0', 'woba_denom', 'woba_value', 'zone'
]

# Keep only columns that exist in the current dataset
available_cols = [c for c in wanted_cols if c in data.columns]
missing_cols = set(wanted_cols) - set(available_cols)

if missing_cols:
    print(f"Missing columns skipped ({len(missing_cols)}): {sorted(list(missing_cols))}")

# Copy the subset of available columns
context = data[available_cols].copy()

# Add base runner presence columns if missing
for base_col in ["on_1b", "on_2b", "on_3b"]:
    if base_col not in context.columns:
        context[base_col] = pd.NA

# Compute total runners on base
context["runners_on_base"] = context[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)

# Add convenience aliases for modeling
if "estimated_ba_using_speedangle" in context.columns:
    context["xba"] = context["estimated_ba_using_speedangle"]
if "estimated_woba_using_speedangle" in context.columns:
    context["xwoba"] = context["estimated_woba_using_speedangle"]
if "estimated_slg_using_speedangle" in context.columns:
    context["xslg"] = context["estimated_slg_using_speedangle"]

# Save the processed context dataset
context.to_csv("data/context_2025_full.csv", index=False)

print(f"Saved {len(context):,} rows × {len(context.columns)} columns to data/context_2025_full.csv")
