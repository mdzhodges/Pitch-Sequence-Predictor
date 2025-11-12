class Constants:
    COLOR_INFO: str = "\033[60m"
    COLOR_WARNING: str = "\033[93m"
    COLOR_ERROR: str = "\033[91m"
    COLOR_RESET: str = "\033[0m"

    START_DATE_STR: str = "2025-03-17"
    END_DATE_STR: str = "2025-09-28"

    FEATURE_EXCLUSION_SET: set[str] = {"IDfg", "Season", "Team", "Age"}

    COLUMN_EXCLUSION_SET: set[str] = {
        "game_date",
        "description",
        "des",
        "umpire",
        "sv_id",
        "pitcher_name",
        "batter_name",
        "home_team",
        "away_team",
    }

    PITCH_SEQUENCE_COLUMNS_LIST: list[str] = [
        "game_date", "game_pk", "at_bat_number", "pitch_number",
        "pitcher", "batter",
        "balls", "strikes",  # <-- count context
        "pitch_type", "release_speed",
        "release_spin_rate", "pfx_x", "pfx_z",
        "plate_x", "plate_z",
        "events"
    ]

    CONTEXT_COLUMNS_LIST: list[str] = [
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
