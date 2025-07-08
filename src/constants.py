"""
constants.py

Defines constants used across notebooks.

Author: Anis Guechtouli
"""

# === Importing Data ===
COMPETITIONS_COLUMNS_DROP = ['competition_youth','match_updated','competition_gender',
                               'match_updated_360','match_available_360','match_available']

MATCHES_COLUMNS_DROP = ['match_status','match_status_360','last_updated','last_updated_360',
                               'shot_fidelity_version','xy_fidelity_version']

EVENT_COLUMNS_SELECT = ['50_50', 'ball_receipt_outcome', 'ball_recovery_recovery_failure', 'carry_end_location',
                        'clearance_aerial_won', 'counterpress', 'dribble_nutmeg', 'dribble_outcome', 'dribble_overrun',
                        'duel_outcome', 'duel_type', 'duration', 'foul_committed_advantage', 'foul_committed_card',
                        'foul_won_advantage', 'foul_won_defensive', 'goalkeeper_body_part', 'goalkeeper_end_location',
                        'goalkeeper_outcome', 'goalkeeper_position', 'goalkeeper_technique', 'goalkeeper_type', 'id', 'index',
                        'interception_outcome', 'location', 'match_id', 'minute', 'pass_aerial_won', 'pass_angle',
                        'pass_assisted_shot_id', 'pass_backheel', 'pass_body_part', 'pass_cross', 'pass_cut_back', 'pass_deflected',
                        'pass_end_location', 'pass_goal_assist', 'pass_height', 'pass_length', 'pass_outcome', 'pass_recipient',
                        'pass_recipient_id', 'pass_shot_assist', 'pass_switch', 'pass_type', 'period', 'play_pattern', 'player',
                        'player_id', 'position', 'possession', 'possession_team', 'possession_team_id', 'related_events', 'second',
                        'shot_aerial_won', 'shot_body_part', 'shot_end_location', 'shot_first_time', 'shot_freeze_frame',
                        'shot_key_pass_id','shot_one_on_one', 'shot_outcome', 'shot_statsbomb_xg', 'shot_technique', 'shot_type',
                        'substitution_outcome', 'substitution_outcome_id', 'substitution_replacement', 'substitution_replacement_id',
                        'tactics', 'team', 'team_id', 'timestamp', 'type', 'under_pressure', 'competition', 'season', 'block_offensive',
                        'foul_committed_offensive', 'foul_committed_type', 'injury_stoppage_in_chain', 'pass_miscommunication',
                        'pass_technique', 'pass_through_ball', 'block_deflection', 'ball_recovery_offensive', 'foul_committed_penalty',
                        'foul_won_penalty', 'shot_open_goal', 'shot_redirect', 'bad_behaviour_card', 'block_save_block',
                        'shot_deflected', 'miscontrol_aerial_won', 'clearance_body_part', 'clearance_head', 'clearance_left_foot',
                        'clearance_right_foot', 'goalkeeper_punched_out', 'off_camera', 'out', 'pass_inswinging', 'pass_outswinging',
                        'pass_straight', 'goalkeeper_shot_saved_to_post', 'pass_no_touch', 'shot_saved_to_post',
                        'goalkeeper_success_in_play', 'clearance_other', 'player_off_permanent', 'goalkeeper_shot_saved_off_target',
                        'shot_saved_off_target', 'shot_follows_dribble', 'dribble_no_touch', 'goalkeeper_lost_out',
                        'half_start_late_video_start', 'goalkeeper_lost_in_play', 'goalkeeper_penalty_saved_to_post',
                        'goalkeeper_saved_to_post', 'goalkeeper_success_out']

# === Data Preprocessing ===
SEASON = '2015/2016'

EVENTS = [
    'id', 'match_id', 'minute', 'second', 'type', 'period', 'location', 'team', 'player_id','player', 'position', # Event info
    'play_pattern','shot_body_part','shot_technique','shot_type', # Shot info
    'shot_freeze_frame', # Complicated info
    'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one',
    'shot_open_goal', 'shot_follows_dribble', # Boolean
    'shot_statsbomb_xg','shot_outcome',# Target
    'pass_body_part','shot_end_location',
    # Pass Data
    'pass_assisted_shot_id', 'pass_height', 'pass_length', 'pass_angle',
    'pass_aerial_won', 'pass_cross', 'pass_cut_back', 'pass_switch', 'pass_through_ball',
    'pass_inswinging', 'pass_outswinging', 'pass_straight', 'pass_no_touch']

DUMMIES = {
    'play_pattern' : {
        'Other': 'other_pp',
        'From Free Kick': 'from_fk',
        'From Throw In': 'from_ti',
        'From Corner': 'from_corner',
        'From Counter': 'from_counter',
        'From Goal Kick': 'from_gk',
        'From Keeper': 'from_keeper',
        'From Kick Off': 'from_ko',
        'Regular Play': 'from_rp'},

    'shot_type': {
        'Free Kick' : 'fk_type',
        'Penalty' : 'pk_type'},

    'shot_technique' : {
        'Half Volley': 'half_volley_technique',
        'Volley': 'volley_technique',
        'Lob': 'lob_technique',
        'Overhead Kick': 'overhead_technique',
        'Backheel': 'backheel_technique',
        'Diving Header': 'diving_h_technique'}
    }

BOOL_TO_INT = ['preferred_foot_shot','under_pressure','shot_aerial_won',
               'shot_first_time','shot_one_on_one',
               'shot_open_goal','shot_follows_dribble','goal',
               'pass_aerial_won', 'pass_cross', 'pass_cut_back', 'pass_switch', 'pass_through_ball',
               'pass_inswinging', 'pass_outswinging', 'pass_straight', 'pass_no_touch']

VARIABLES = ['id', 'player','player_id', 'match_id', 'team', 'period', 'minute', 'second',
    'shot_location_x','shot_location_y','distance_to_goal','shot_angle', # Spatial data
    'preferred_foot_shot', # Boolean
    'from_rp','from_fk','from_ti','from_corner','from_counter','from_gk','from_keeper','from_ko', # Play pattern
    'header','fk_type','pk_type', # Shot type
    'half_volley_technique','volley_technique','lob_technique','overhead_technique','backheel_technique', # Shot technique
    'diving_h_technique',
    'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble', # Boolean
    'players_inside_area', # Spatial data
    
    'assisted', 'pass_height', 'pass_length', 'pass_angle',
    'pass_aerial_won', 'pass_cross', 'pass_cut_back', 'pass_switch', 'pass_through_ball',
    'pass_inswinging', 'pass_outswinging', 'pass_straight', 'pass_no_touch',
    'shot_outcome', 'shot_end_location', 'shot_end_x', 'shot_end_y', 'sb_prediction',
    
    'shot_statsbomb_xg','goal']

FEATURES = [
    'distance_to_goal','shot_angle', # Spatial data
    'preferred_foot_shot', # Boolean
    'from_rp','from_fk','from_ti','from_corner','from_counter','from_gk','from_keeper','from_ko', # Play pattern
    'header','fk_type','pk_type', # Shot type
    'half_volley_technique','volley_technique','lob_technique','overhead_technique','backheel_technique', # Shot technique
    'diving_h_technique',
    'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble', # Boolean
    'players_inside_area', # Spatial data
    # Pass data
    'assisted', 'pass_height', 'pass_length', 'pass_angle',
    'pass_aerial_won', 'pass_cross', 'pass_cut_back', 'pass_switch', 'pass_through_ball',
    'pass_inswinging', 'pass_outswinging', 'pass_straight', 'pass_no_touch'
    ]

# === Model Creation ===
