SEASON = '2015/2016'

MODELS = ['logistic', 'rf', 'mlp', 'gbt', 'dt', 'svm']

EVENTS = [
    'id', 'match_id', 'minute', 'second', 'type', 'period', 'location', 'team', 'player_id', 'position', # Event info
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
        'Corner' : 'corner_type',
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

VARIABLES = ['id', 'player_id', 'match_id', 'team', 'period', 'minute', 'second',
    'shot_location_x','shot_location_y','distance_to_goal','shot_angle', # Spatial data
    'preferred_foot_shot', # Boolean
    'from_rp','from_fk','from_ti','from_corner','from_counter','from_gk','from_keeper','from_ko', # Play pattern
    'header','corner_type','fk_type','pk_type', # Shot type
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
    'header','corner_type','fk_type','pk_type', # Shot type
    'half_volley_technique','volley_technique','lob_technique','overhead_technique','backheel_technique', # Shot technique
    'diving_h_technique',
    'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble', # Boolean
    'players_inside_area', # Spatial data
    # Pass data
    'assisted', 'pass_height', 'pass_length', 'pass_angle',
    'pass_aerial_won', 'pass_cross', 'pass_cut_back', 'pass_switch', 'pass_through_ball',
    'pass_inswinging', 'pass_outswinging', 'pass_straight', 'pass_no_touch'
    ]