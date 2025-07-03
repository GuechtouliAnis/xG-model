import pandas as pd
import numpy as np
from statsbombpy import sb
from src.constants import *

# === Matches Data ===
def get_matches(competitions: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches and aggregates match data for each competition-season pair.

    Args:
        competitions (pd.DataFrame): A DataFrame containing at least
            'competition_id' and 'season_id' columns.

    Returns:
        pd.DataFrame: Concatenated match data with unnecessary columns dropped.

    Notes:
        - Uses the `statsbombpy` library to fetch matches.
        - Drops columns defined in `MATCHES_COLUMNS_DROP` constant.
        - Result is reset-indexed and ready for downstream processing.
    """
    df_matches = pd.DataFrame()
    
    for i in range(0, len(competitions)):
        competition_id, season_id = competitions.loc[i, ['competition_id', 'season_id']]
        if df_matches.empty:
            df_matches = sb.matches(competition_id=competition_id,
                                    season_id=season_id)
        else:
            df_matches = pd.concat([df_matches,
                                    sb.matches(competition_id=competition_id, season_id=season_id)])
    df_matches.reset_index(drop=True,
                           inplace=True)
    
    df_matches.drop(MATCHES_COLUMNS_DROP,
                    axis=1,
                    inplace=True)
    
    return df_matches

# === Lineups Data ===
def concat_lineups(lineups_dict: dict, match_id: int) -> pd.DataFrame:
    """
    Flattens the nested lineups dictionary for a single match into a DataFrame.

    Args:
        lineups_dict (dict): Dictionary returned by `sb.lineups(match_id)`,
            where keys are team names and values are player lineups as DataFrames.
        match_id (int): Match ID to associate with all rows.

    Returns:
        pd.DataFrame: Combined lineups for both teams with `match_id` column added.
    """
    df = pd.DataFrame()
    for team_df in lineups_dict.values():
        df = pd.concat([df, team_df])
    
    df['match_id'] = match_id
    return df

def get_lineups(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches and aggregates player lineups for each match in the dataset.

    Args:
        df_matches (pd.DataFrame): DataFrame containing at least a 'match_id' column.

    Returns:
        pd.DataFrame: Concatenated player lineup data across all matches.
    """
    df_lineups = pd.DataFrame()

    for i in range(len(df_matches)):
        match_id = df_matches.loc[i, 'match_id']
        lineups_dict = sb.lineups(match_id)
        df = concat_lineups(lineups_dict, match_id)

        if df_lineups.empty:
            df_lineups = df
        else:
            df_lineups = pd.concat([df_lineups, df])

        if i % 50 == 0:
            print(f"[{i}] Processed match_id {match_id}")

    df_lineups.reset_index(drop=True, inplace=True)
    return df_lineups

# === Events Data ===
def get_events(
    df_matches: pd.DataFrame,
    std_cols: list[str],
    start_i: int,
    end_i: int,
) -> None:
    """
    Fetches event data for a range of matches, standardizes columns, and exports to CSV.

    Args:
        df_matches (pd.DataFrame): Match metadata with at least 'match_id', 'competition', and 'season'.
        std_cols (list[str]): List of standard columns to enforce on the final DataFrame.
        start_i (int): Index (inclusive) to start extracting events from df_matches.
        end_i (int): Index (exclusive) to stop extraction.
        export_path (str): Directory path where the output CSV will be saved.

    Returns:
        None. Saves a CSV file named `events_{start_i}_{end_i}.csv` in `export_path`.

    Notes:
        - The function prints progress every 50 matches.
        - If any `match_id` values are missing, it reports them.
        - Data is saved in chunked form to allow partial processing.
    """
    events = pd.DataFrame()

    for i in range(start_i, end_i):
        match_id = df_matches.loc[i, 'match_id']
        competition = df_matches.loc[i, 'competition']
        season = df_matches.loc[i, 'season']

        try:
            current_events = sb.events(match_id=match_id)
            current_events = current_events.reindex(columns=std_cols, fill_value=np.nan)
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

    events.reset_index(drop=True, inplace=True)
    
    return events

# === Events Data ===
