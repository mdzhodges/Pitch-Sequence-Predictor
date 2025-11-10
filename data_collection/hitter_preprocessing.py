import warnings
import pandas as pd
import pybaseball
from pybaseball import statcast, playerid_reverse_lookup


# Constants
START_DT = "2025-03-17"
END_DT = "2025-09-28"
OUTPUT_PATH = "data/statcast_hitters_2025.csv"

# disable warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")
pybaseball.cache.enable()

print(f"Fetching Statcast data for {START_DT} → {END_DT} ...")

# Get data
data = statcast(start_dt=START_DT, end_dt=END_DT)

# Raname to pitcher name
if "player_name" in data.columns:
    data = data.rename(columns={"player_name": "pitcher_name"})

# The ID given in the batter ID, but we need the batter name, hence the remap
batter_ids = data["batter"].dropna().unique().astype(int).tolist()
id_map = playerid_reverse_lookup(batter_ids, key_type="mlbam")

# Get the batter name
id_map["batter_name"] = id_map["name_first"].fillna("") + " " + id_map["name_last"].fillna("")
id_map = id_map[["key_mlbam", "batter_name"]]


# merge the column of the two data, to get one complete set
data = data.merge(id_map, left_on="batter", right_on="key_mlbam", how="left")
data.drop(columns=["key_mlbam"], inplace=True)

# these are the hitter stats we are interested in
hitter_cols = [
    "game_date", "game_pk", "batter", "batter_name", "stand",
    "pitch_type", "pitcher", "pitcher_name", "events", "description",
    "type", "bb_type", "launch_speed", "launch_angle", "hit_distance_sc",
    "hc_x", "hc_y", "balls", "strikes", "outs_when_up", "inning",
    "inning_topbot", "on_1b", "on_2b", "on_3b", "home_team", "away_team",
    "plate_x", "plate_z", "zone", "sz_top", "sz_bot"
]

# do some data manipulation to get the batter data we want from the data variables
available_cols = [c for c in hitter_cols if c in data.columns]
hitter_data = data[available_cols].copy()

# print to the csv file (is git ignored)
hitter_data.to_csv(OUTPUT_PATH, index=False)
