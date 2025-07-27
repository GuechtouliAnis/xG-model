import pandas as pd
import numpy as np
from statsbombpy import sb
from src.constants import *

# === Matches Data ===
def get_matches(competitions: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches and aggregates match data for each competition-season pair.

    Args:
        competitions (pd.DataFrame): DataFrame with 'competition_id' and 'season_id'.

    Returns:
        pd.DataFrame: Match data with dropped columns defined in MATCHES_COLUMNS_DROP.
    """
    
    required_columns = ['competition_id', 'season_id']
    missing = [col for col in required_columns if col not in competitions.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    
    df_matches = pd.DataFrame()
    
    for i in range(len(competitions)):
        competition_id, season_id = competitions.loc[i, required_columns]
        match_data = sb.matches(competition_id=competition_id,
                                season_id=season_id)
        df_matches = pd.concat([df_matches, match_data],
                               ignore_index=True)
    
    df_matches.drop(MATCHES_COLUMNS_DROP,
                    axis=1,
                    inplace=True)
    
    return df_matches

# === Lineups Data ===
def concat_lineups(lineups_dict: dict,
                   match_id: int) -> pd.DataFrame:
    """
    Flattens a nested lineups dictionary into a single DataFrame.

    Args:
        lineups_dict (dict): Output of sb.lineups(match_id), team_name -> lineup DataFrame.
        match_id (int): Match ID to assign to all rows.

    Returns:
        pd.DataFrame: Combined lineups with 'match_id' column added.
    """
    
    df = pd.DataFrame()
    for team_df in lineups_dict.values():
        df = pd.concat([df, team_df])
    
    df['match_id'] = match_id
    return df

def get_lineups(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches and aggregates player lineups for all matches.

    Args:
        df_matches (pd.DataFrame): DataFrame with 'match_id' column.

    Returns:
        pd.DataFrame: All player lineups with associated match IDs.
    """
    
    # Error handling: check required column
    required_column = 'match_id'
    if required_column not in df_matches.columns:
        raise ValueError(f"Missing required column: '{required_column}'")
    
    df_lineups = pd.DataFrame()

    for i in range(len(df_matches)):
        match_id = df_matches.loc[i, required_column]
        lineups_dict = sb.lineups(match_id)
        df = concat_lineups(lineups_dict,
                            match_id)
        df_lineups = pd.concat([df_lineups, df],
                               ignore_index=True)

        if i % 50 == 0:
            print(f"[{i}] Processed match_id {match_id}")

    return df_lineups

# === Events Data ===
def get_events(df_matches: pd.DataFrame,
    std_cols: list[str],
    start_i: int,
    end_i: int) -> pd.DataFrame:
    """
    Fetches and standardizes event data for a given range of matches.

    Args:
        df_matches (pd.DataFrame): DataFrame with 'match_id', 'competition', and 'season'.
        std_cols (list[str]): List of columns to enforce on the output DataFrame.
        start_i (int): Start index (inclusive) from df_matches.
        end_i (int): End index (exclusive) from df_matches.

    Returns:
        pd.DataFrame: Standardized event data for all processed matches.
    """
    
    required_columns = ['match_id','competition','season']
    missing = [col for col in required_columns if col not in df_matches.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    
    events = pd.DataFrame()

    for i in range(start_i, end_i):
        match_id = df_matches.loc[i, 'match_id']
        competition = df_matches.loc[i, 'competition']
        season = df_matches.loc[i, 'season']

        try:
            current_events = sb.events(match_id=match_id)
            current_events = current_events.reindex(columns=std_cols,
                                                    fill_value=np.nan)
            current_events['competition'] = competition
            current_events['season'] = season

            if events.empty:
                events = current_events
            else:
                events = pd.concat([events[std_cols], current_events[std_cols]],
                                   ignore_index=True)

            if i % 50 == 0:
                print(f"[{i}] Processed match_id={match_id} ({len(events)} rows)")

            if events['match_id'].isna().any():
                print(f"Null match_id at match {i} (match_id={match_id})")

        except Exception as e:
            print(f"Error at match {i} (match_id={match_id}): {str(e)}")
            continue

    # Final stats
    print(f"Completed {i + 1 - start_i} matches.")
    print(f"Total rows: {len(events)}")
    print(f"Columns: {len(events.columns)}")
    print(f"Null match_id values: {events['match_id'].isna().sum()}")

    events.reset_index(drop=True,
                       inplace=True)
    
    return events

# === Frames Data ===
def get_frames(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches freeze-frame data for each match.

    Args:
        df_matches (pd.DataFrame): DataFrame with a 'match_id' column.

    Returns:
        pd.DataFrame: Combined freeze-frame data across all matches.
    """
    
    required_column = 'match_id'
    if required_column not in df_matches.columns:
        raise ValueError(f"Missing required column: '{required_column}'")

    df_frames = pd.DataFrame()

    for i in range(len(df_matches)):
        match_id = df_matches.loc[i, 'match_id']
        try:
            df = sb.frames(match_id)
            df_frames = pd.concat([df_frames, df], ignore_index=True)
        except Exception as e:
            print(f"Skipped match_id={match_id} due to error: {e}")
            continue

        if i % 50 == 0:
            print(f"[{i}] Processed match_id={match_id}")

    df_frames.reset_index(drop=True, inplace=True)
    return df_frames