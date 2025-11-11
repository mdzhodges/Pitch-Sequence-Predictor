from pybaseball import pitching_stats
import pandas as pd

# Get every single stat possible from every hitter possiible
pitching_stats = pitching_stats(2025, qual=0)

# Remove hitters with no PA
pitching_stats = pitching_stats[pitching_stats["IP"] > 0].reset_index(drop=True)

# Save everything
pitching_stats.to_csv("data/pitchers_2025_full.csv", index=False)

print(pitching_stats.head(5))
print("Retrieved")