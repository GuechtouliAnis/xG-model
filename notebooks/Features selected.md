## Feature Selection

Selecting the correct features is a critical aspect of any data project.<br>
You'll find the features chosen for this project listed below as well as inside `src/constants.py` as a list that will be used for preprocessing.

### 1 - Event and player-related information features
- `id`: Unique identifier for the event  
- `match_id`: Unique identifier for the match  
- `minute`, `second`: Time of the event within the match  
- `period`: Match period (1 = first half, 2 = second half, etc.)  
- `type`: Type of the event (e.g., Pass, Shot)  
- `location`: (x, y) coordinates of the event (0–120 length, 0–80 width)  
- `team`: Team performing the event  
- `player_id`, `player`: Player's unique ID and name
- `position`: Player's position at the time of the event  

---

### 2 - Play and shot context
- `play_pattern`: The game state or sequence pattern (e.g., Regular Play, From Throw In)  
- `shot_body_part`: Body part used for the shot (Right Foot, Left Foot, Head, Other)  
- `shot_technique`: Technical classification of the shot (e.g., Volley, Backheel)
- `shot_type`: Type of shot (e.g., Open Play, Free Kick, Penalty)  
- `shot_freeze_frame`: Frame capturing the positioning of players close to the shooter at the time of shot  

---

### 3 - Boolean shot-related flags
- `under_pressure`: Whether the player was under defensive pressure  
- `shot_aerial_won`: Whether an aerial duel was won before the shot  
- `shot_first_time`: Whether it was a first-time shot  
- `shot_one_on_one`: Whether the shot situation was 1v1 with the keeper  
- `shot_open_goal`: Whether the shot was taken at an open goal  
- `shot_follows_dribble`: Whether the shot followed a dribble  

---

### 4 - Shot outcome and model target
- `shot_statsbomb_xg`: Expected goal value of the shot (between 0 and 1)  
- `shot_outcome`: Result of the shot (e.g., Goal, Missed, Saved, Blocked)  

---

### 5 - Basic pass and shot destination info
- `pass_body_part`: Body part used to make the pass  
- `shot_end_location`: Location where the shot ended (x, y, z if available)  

---

### 6 - Detailed pass characteristics
- `pass_assisted_shot_id`: ID of the shot assisted by this pass (if any)  
- `pass_height`: Height of the pass (e.g., Ground, High)  
- `pass_length`: Distance covered by the pass  
- `pass_angle`: Angle of the pass relative to the pitch axis  

---

### 7 - Boolean pass-related tags
- `pass_aerial_won`: Whether an aerial duel was won during/after the pass  
- `pass_cross`: Whether the pass was a cross  
- `pass_cut_back`: Whether the pass was a cut back  
- `pass_switch`: Whether it was a switch of play  
- `pass_through_ball`: Whether it was a through ball  
- `pass_inswinging`: Whether the pass curled inwards  
- `pass_outswinging`: Whether the pass curled outwards  
- `pass_straight`: Whether the pass was straight  
- `pass_no_touch`: No physical contact before the next action (e.g., dummy or let-run)