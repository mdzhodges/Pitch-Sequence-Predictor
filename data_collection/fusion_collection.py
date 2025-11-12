import pandas as pd
from pybaseball import playerid_reverse_lookup
from controller.config import Config
from utils.logger import Logger

def preprocess_unified(debug: bool = True):
    """
    Creates a unified dataset combining context, hitter, and pitcher information.
    Generates a `next_pitch_type` and corresponding integer label `next_pitch_idx`
    for next-pitch prediction. Keeps only rows with valid next-pitch targets.
    """

    config = Config()

    # ------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------
    context_df = pd.read_parquet(config.CONTEXT_PARQUET_FILE_PATH)
    hitter_df = pd.read_parquet(config.HITTER_PARQUET_FILE_PATH)
    pitcher_df = pd.read_parquet(config.PITCHER_PARQUET_FILE_PATH)

    # ------------------------------------------------------------
    # 2. Map FanGraphs -> MLBAM IDs
    # ------------------------------------------------------------
    fg_ids = pd.concat([hitter_df["IDfg"], pitcher_df["IDfg"]]).dropna().unique().tolist()
    id_map = playerid_reverse_lookup(fg_ids, key_type="fangraphs")[["key_fangraphs", "key_mlbam"]].dropna()
    id_map = id_map.rename(columns={"key_fangraphs": "IDfg", "key_mlbam": "mlbam"})

    hitter_df = hitter_df.merge(id_map, on="IDfg", how="left")
    pitcher_df = pitcher_df.merge(id_map, on="IDfg", how="left")

    # ------------------------------------------------------------
    # 3. Identify sequencing columns
    # ------------------------------------------------------------
    possible_ab_cols = [c for c in context_df.columns if "at_bat" in c.lower()]
    possible_pitch_cols = [c for c in context_df.columns if "pitch_number" in c.lower() or "pitch_no" in c.lower()]
    possible_game_cols = [c for c in context_df.columns if "game_pk" in c.lower() or "game_id" in c.lower()]

    if not possible_ab_cols or not possible_pitch_cols:
        raise KeyError(
            f"Missing sequencing columns. Found: at_bat={possible_ab_cols}, pitch={possible_pitch_cols}"
        )

    ab_col = possible_ab_cols[0]
    pitch_col = possible_pitch_cols[0]
    game_col = possible_game_cols[0] if possible_game_cols else None

    # ------------------------------------------------------------
    # 4. Sort dataset by pitch order
    # ------------------------------------------------------------
    sort_cols = [col for col in [game_col, ab_col, pitch_col] if col is not None]
    context_df = context_df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------
    # 5. Generate next-pitch labels
    # ------------------------------------------------------------
    group_cols = [col for col in [game_col, ab_col] if col is not None]
    context_df["next_pitch_type"] = (
        context_df.groupby(group_cols)["pitch_type"].shift(-1)
    )

    context_df = context_df.dropna(subset=["next_pitch_type"]).reset_index(drop=True)
    # ------------------------------------------------------------
    # 6. Encode pitch type and next pitch type
    # ------------------------------------------------------------
    pitch_vocab = sorted(context_df["pitch_type"].dropna().unique().tolist())
    pitch_map = {p: i for i, p in enumerate(pitch_vocab)}

    context_df["pitch_type_idx"] = context_df["pitch_type"].map(pitch_map)
    context_df["next_pitch_idx"] = context_df["next_pitch_type"].map(pitch_map)

    # ------------------------------------------------------------
    # 7. Check column overlaps before merging hitter/pitcher
    # ------------------------------------------------------------
    context_cols = set(context_df.columns)
    hitter_cols = set(hitter_df.columns)
    pitcher_cols = set(pitcher_df.columns)

    _overlap_hitter = context_cols & hitter_cols
    _overlap_pitcher = context_cols & pitcher_cols

    # ------------------------------------------------------------
    # 8. Merge hitter and pitcher info (collision-safe)
    # ------------------------------------------------------------
    context_df = (
        context_df.merge(
            hitter_df.add_suffix("_hitter"),
            left_on="batter",
            right_on="mlbam_hitter",
            how="left",
            suffixes=("", "_dup_hitter")
        )
        .merge(
            pitcher_df.add_suffix("_pitcher"),
            left_on="pitcher",
            right_on="mlbam_pitcher",
            how="left",
            suffixes=("", "_dup_pitcher")
        )
    )

    # ------------------------------------------------------------
    # 9. Save unified dataset
    # ------------------------------------------------------------
    save_path = "data/unified_context.parquet"
    context_df.to_parquet(save_path, index=False)
    
    logger = Logger("Generation of Fusion")
    
    logger.info(f"Generated data of size: {context_df.shape} <= Should be (732976, 820)")
    


    return context_df
