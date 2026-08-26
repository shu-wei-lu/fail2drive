export ACTIVATION_POLICY=pdm_oracle
export PDM_ORACLE_ACTION=auto
export PDM_ORACLE_ALPHA=1.0
export PDM_ORACLE_TRIGGER_DISTANCE=20
export PDM_ORACLE_HOLD_FRAMES=5
export PDM_ORACLE_COOLDOWN_FRAMES=10
export PDM_ORACLE_TWO_WAY_CLEAR_DISTANCE=70
export PDM_ORACLE_LANE_KEY_SEARCH_DISTANCE=100
export PDM_ORACLE_SIDE_HAZARD_DISTANCE=25
export PDM_ORACLE_SIDE_HAZARD_TWO_WAY_DISTANCE=10
export PDM_ORACLE_ROADBLOCKED_DISTANCE=40
export PDM_ORACLE_PRIORITY_DISTANCE=25
export PDM_ORACLE_YIELD_EMERGENCY_DISTANCE=50
export PDM_ORACLE_GENERAL_BRAKE=1
export ACTIVATION_VECTOR_PATHS="./steering/transfuser/post_process/brake/steering_vector.pt,./steering/transfuser/post_process/left/steering_vector.pt,./steering/transfuser/post_process/right/steering_vector.pt"

export BRAKE_ACTIVATION_ALPHA_SCALE=1.0
export LEFT_ACTIVATION_ALPHA_SCALE=2.0
export RIGHT_ACTIVATION_ALPHA_SCALE=1.0

# Feature-only scalar-projection gate. The negative_mean.pt files are loaded
# automatically from the directories containing the steering vectors.
export ACTIVATION_PROJECTION_GATE=1
export ACTIVATION_PROJECTION_GATE_LOW=0.05
export ACTIVATION_PROJECTION_GATE_HIGH=0.10
export ACTIVATION_PROJECTION_GATE_VERBOSE=0

# Brake remains ungated by default because its current vector represents the
# stopped/brake-hold state rather than active braking.
export ACTIVATION_PROJECTION_GATE_BRAKE=0
