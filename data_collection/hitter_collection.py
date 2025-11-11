from pybaseball import batting_stats
import pandas as pd

# Get every single stat possible from every hitter possiible
batters_2025 = batting_stats(2025, qual=0)

# Remove hitters with no PA
batters_2025 = batters_2025[batters_2025["PA"] > 0].reset_index(drop=True)

# Save everything
batters_2025.to_csv("batters_2025_full.csv", index=False)

print(batters_2025.head(5))
print("Retrieved")