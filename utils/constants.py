class Constants:
    COLOR_INFO: str = "\033[60m"
    COLOR_WARNING: str = "\033[93m"
    COLOR_ERROR: str = "\033[91m"
    COLOR_RESET: str = "\033[0m"

    START_DATE_STR:str = "2025-03-17"
    END_DATE_STR:str = "2025-09-28"

    PITCH_SEQUENCE_COLUMNS_LIST: list[str] = [
        "game_date", "game_pk", "at_bat_number", "pitch_number",
        "pitcher", "batter",
        "balls", "strikes",  # <-- count context
        "pitch_type", "release_speed",
        "release_spin_rate", "pfx_x", "pfx_z",
        "plate_x", "plate_z",
        "events"
    ]
