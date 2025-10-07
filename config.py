# F1 Probability Simulator - Configuration

# F1 Circuit Configurations with Prediction Modifiers
TRACK_CONFIGS = {
    "Bahrain": {
        "name": "Bahrain", "fastf1_name": "Bahrain Grand Prix",
        "circuit_type": "desert", "overtaking_difficulty": "medium",
        "grid_importance": 0.75, "strategy_factor": 0.8, "chaos_factor": 0.3,
        "drs_zones": 3, "total_turns": 15, "lap_length_km": 5.412
    },
    "Saudi Arabia": {
        "name": "Saudi Arabia", "fastf1_name": "Saudi Arabian Grand Prix", 
        "circuit_type": "street", "overtaking_difficulty": "high",
        "grid_importance": 0.85, "strategy_factor": 0.6, "chaos_factor": 0.7,
        "drs_zones": 3, "total_turns": 27, "lap_length_km": 6.174
    },
    "Australia": {
        "name": "Australia", "fastf1_name": "Australian Grand Prix",
        "circuit_type": "street", "overtaking_difficulty": "medium", 
        "grid_importance": 0.8, "strategy_factor": 0.7, "chaos_factor": 0.5,
        "drs_zones": 3, "total_turns": 14, "lap_length_km": 5.278
    },
    "Japan": {
        "name": "Japan", "fastf1_name": "Japanese Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "hard",
        "grid_importance": 0.9, "strategy_factor": 0.4, "chaos_factor": 0.2,
        "drs_zones": 1, "total_turns": 18, "lap_length_km": 5.807
    },
    "China": {
        "name": "China", "fastf1_name": "Chinese Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "easy",
        "grid_importance": 0.6, "strategy_factor": 0.9, "chaos_factor": 0.4,
        "drs_zones": 2, "total_turns": 16, "lap_length_km": 5.451
    },
    "Miami": {
        "name": "Miami", "fastf1_name": "Miami Grand Prix",
        "circuit_type": "street", "overtaking_difficulty": "medium",
        "grid_importance": 0.75, "strategy_factor": 0.7, "chaos_factor": 0.6,
        "drs_zones": 3, "total_turns": 19, "lap_length_km": 5.41
    },
    "Imola": {
        "name": "Imola", "fastf1_name": "Emilia Romagna Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "very_hard",
        "grid_importance": 0.95, "strategy_factor": 0.3, "chaos_factor": 0.1,
        "drs_zones": 2, "total_turns": 19, "lap_length_km": 4.909
    },
    "Monaco": {
        "name": "Monaco", "fastf1_name": "Monaco Grand Prix",
        "circuit_type": "street", "overtaking_difficulty": "impossible",
        "grid_importance": 0.98, "strategy_factor": 0.2, "chaos_factor": 0.8,
        "drs_zones": 1, "total_turns": 19, "lap_length_km": 3.337
    },
    "Canada": {
        "name": "Canada", "fastf1_name": "Canadian Grand Prix",
        "circuit_type": "semi_permanent", "overtaking_difficulty": "easy",
        "grid_importance": 0.5, "strategy_factor": 0.9, "chaos_factor": 0.7,
        "drs_zones": 3, "total_turns": 14, "lap_length_km": 4.361
    },
    "Spain": {
        "name": "Spain", "fastf1_name": "Spanish Grand Prix", 
        "circuit_type": "permanent", "overtaking_difficulty": "hard",
        "grid_importance": 0.85, "strategy_factor": 0.6, "chaos_factor": 0.2,
        "drs_zones": 2, "total_turns": 16, "lap_length_km": 4.675
    },
    "Austria": {
        "name": "Austria", "fastf1_name": "Austrian Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "medium", 
        "grid_importance": 0.7, "strategy_factor": 0.7, "chaos_factor": 0.4,
        "drs_zones": 3, "total_turns": 10, "lap_length_km": 4.318
    },
    "Britain": {
        "name": "Britain", "fastf1_name": "British Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "medium",
        "grid_importance": 0.75, "strategy_factor": 0.8, "chaos_factor": 0.3,
        "drs_zones": 2, "total_turns": 18, "lap_length_km": 5.891
    },
    "Hungary": {
        "name": "Hungary", "fastf1_name": "Hungarian Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "very_hard",
        "grid_importance": 0.9, "strategy_factor": 0.4, "chaos_factor": 0.2,
        "drs_zones": 1, "total_turns": 14, "lap_length_km": 4.381
    },
    "Belgium": {
        "name": "Belgium", "fastf1_name": "Belgian Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "easy",
        "grid_importance": 0.6, "strategy_factor": 0.9, "chaos_factor": 0.5,
        "drs_zones": 2, "total_turns": 19, "lap_length_km": 7.004
    },
    "Netherlands": {
        "name": "Netherlands", "fastf1_name": "Dutch Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "hard",
        "grid_importance": 0.85, "strategy_factor": 0.5, "chaos_factor": 0.3,
        "drs_zones": 1, "total_turns": 14, "lap_length_km": 4.259
    },
    "Italy": {
        "name": "Italy", "fastf1_name": "Italian Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "easy",
        "grid_importance": 0.65, "strategy_factor": 0.8, "chaos_factor": 0.4,
        "drs_zones": 2, "total_turns": 11, "lap_length_km": 5.793
    },
    "Singapore": {
        "name": "Singapore", "fastf1_name": "Singapore Grand Prix",
        "circuit_type": "street", "overtaking_difficulty": "hard", 
        "grid_importance": 0.85, "strategy_factor": 0.7, "chaos_factor": 0.6,
        "drs_zones": 3, "total_turns": 23, "lap_length_km": 5.063
    },
    "United States": {
        "name": "United States", "fastf1_name": "United States Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "medium",
        "grid_importance": 0.7, "strategy_factor": 0.8, "chaos_factor": 0.4,
        "drs_zones": 2, "total_turns": 20, "lap_length_km": 5.513
    },
    "Mexico": {
        "name": "Mexico", "fastf1_name": "Mexico City Grand Prix", 
        "circuit_type": "permanent", "overtaking_difficulty": "medium",
        "grid_importance": 0.75, "strategy_factor": 0.7, "chaos_factor": 0.5,
        "drs_zones": 3, "total_turns": 17, "lap_length_km": 4.304
    },
    "Brazil": {
        "name": "Brazil", "fastf1_name": "Brazilian Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "medium",
        "grid_importance": 0.7, "strategy_factor": 0.8, "chaos_factor": 0.6,
        "drs_zones": 2, "total_turns": 15, "lap_length_km": 4.309
    },
    "Las Vegas": {
        "name": "Las Vegas", "fastf1_name": "Las Vegas Grand Prix",
        "circuit_type": "street", "overtaking_difficulty": "easy", 
        "grid_importance": 0.6, "strategy_factor": 0.9, "chaos_factor": 0.7,
        "drs_zones": 3, "total_turns": 14, "lap_length_km": 6.201
    },
    "Qatar": {
        "name": "Qatar", "fastf1_name": "Qatar Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "medium",
        "grid_importance": 0.75, "strategy_factor": 0.7, "chaos_factor": 0.3,
        "drs_zones": 2, "total_turns": 16, "lap_length_km": 5.380
    },
    "Abu Dhabi": {
        "name": "Abu Dhabi", "fastf1_name": "Abu Dhabi Grand Prix",
        "circuit_type": "permanent", "overtaking_difficulty": "medium",
        "grid_importance": 0.75, "strategy_factor": 0.8, "chaos_factor": 0.4,
        "drs_zones": 2, "total_turns": 16, "lap_length_km": 5.281
    }
}

# Universal Data Collection Settings
DATA_CONFIG = {
    "years_to_collect": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "selected_circuit": "Singapore",  # Default circuit - can be changed
    "cache_enabled": True,
    "data_output_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "fallback_years": [2018, 2019],  # Backup years if recent data unavailable
    "min_samples": 40,  # Minimum samples needed for training
    "max_years_back": 6  # Maximum years to look back for data
}

# Universal race selection function
def get_available_circuits():
    """Get list of all available F1 circuits"""
    return list(TRACK_CONFIGS.keys())

def set_target_circuit(circuit_name):
    """Set the target circuit for data collection and modeling"""
    if circuit_name in TRACK_CONFIGS:
        DATA_CONFIG["selected_circuit"] = circuit_name
        return True
    else:
        available = get_available_circuits()
        raise ValueError(f"Circuit '{circuit_name}' not found. Available: {available}")

def get_circuit_config(circuit_name=None):
    """Get configuration for specified circuit or currently selected circuit"""
    circuit = circuit_name or DATA_CONFIG["selected_circuit"]
    return TRACK_CONFIGS.get(circuit, TRACK_CONFIGS["Singapore"])

# Feature groups for easier management
FEATURE_GROUPS = {
    "core_race": [
        "driver_name", "team_name", "year", "starting_position", 
        "finishing_position", "points_scored", "dnf_status", 
        "total_race_time", "positions_gained_lost"
    ],
    "qualifying": [
        "qualifying_position", "q1_time", "q2_time", "q3_time",
        "qualifying_gap_to_pole", "qualifying_gap_to_next_car"
    ],
    "lap_performance": [
        "total_laps_completed", "avg_lap_time", "fastest_lap_time",
        "lap_time_consistency", "sector_1_avg", "sector_2_avg", 
        "sector_3_avg", "gap_to_leader_avg", "gap_to_car_ahead_avg"
    ],
    "tyre_strategy": [
        "tyre_compounds_used", "number_of_pit_stops", "stint_lengths",
        "pit_stop_durations", "tyre_age_at_race_start"
    ],
    "weather_track": [
        "track_temperature", "air_temperature", "humidity", 
        "rainfall", "session_type"
    ],
    "telemetry": [
        "avg_speed_per_lap", "max_speed_achieved", "min_speed_corners",
        "throttle_avg_per_lap", "throttle_100_percent_time", 
        "brake_avg_per_lap", "brake_applications_count",
        "avg_gear_per_lap", "gear_changes_per_lap",
        "avg_rpm_per_lap", "max_rpm_per_lap", "drs_usage_percentage"
    ],
    "race_context": [
        "safety_car_periods_count", "virtual_safety_car_periods",
        "overtakes_made", "overtakes_received"
    ]
}

# Feature importance weights (0.0 to 1.0) based on correlation analysis
FEATURE_WEIGHTS = {
    # High importance features (>0.7 correlation with finish position)
    "high_importance": {
        "grid_pos": 0.95,           # 0.791 correlation - starting position critical
        "quali_pos": 0.90,          # 0.778 correlation - qualifying performance
        "team_strength": 0.85,      # -0.824 correlation - team capability decisive
        "driver_skill": 0.80,       # -0.733 correlation - driver ability
        "gap_to_pole": 0.75,        # Qualifying pace relative to best
    },
    
    # Medium importance features (0.3-0.7 correlation)
    "medium_importance": {
        "pit_stops": 0.60,          # 0.348 correlation - strategy impact
        "q3_time": 0.65,            # Raw qualifying pace
        "q2_time": 0.55,            # Backup qualifying time
        "driver_experience": 0.50,   # Experience factor
        "year_normalized": 0.45,     # Temporal effects
    },
    
    # Supporting features (context and derived)
    "supporting": {
        "q1_time": 0.35,            # Basic qualifying measure
        "total_laps": 0.30,         # Race completion
        "tyres_used": 0.25,         # Strategy context
        "pos_change": 0.20,         # Historical position change
        "points": 0.15,             # Result indicator (not predictor)
    }
}

# Model-specific feature priorities
MODEL_FEATURE_CONFIG = {
    "VAE": {
        "primary_features": [
            "grid_pos", "quali_pos", "driver_skill", "team_strength", 
            "pit_stops", "gap_to_pole"
        ],
        "feature_weights": "high_importance + medium_importance",
        "normalization": "standard_scaler",
        "latent_dimensions": 4
    },
    
    "BayesianNetwork": {
        "discrete_nodes": [
            "grid_pos_binned", "finish_pos_binned", "pit_stops", 
            "driver_category", "team_tier", "year"
        ],
        "continuous_nodes": [
            "driver_skill", "team_strength", "gap_to_pole", "q3_time"
        ],
        "evidence_nodes": ["grid_pos_binned", "driver_skill", "team_strength"],
        "target_node": "finish_pos_binned"
    },
    
    "ensemble": {
        "feature_selection": "top_10_by_weight",
        "weight_threshold": 0.4,
        "include_engineered": True
    }
}

# Position binning for discrete models
POSITION_BINS = {
    "podium_contenders": [1, 2, 3],      # Top 3 finishers
    "points_scorers": [4, 5, 6, 7, 8, 9, 10],  # Points positions
    "midfield": [11, 12, 13, 14, 15],    # Competitive midfield
    "backmarkers": [16, 17, 18, 19, 20]  # Bottom positions
}

# Driver/Team categorization for modeling
DRIVER_CATEGORIES = {
    "elite": ["Hamilton", "Verstappen", "Leclerc", "Russell", "Norris"],
    "experienced": ["Alonso", "Perez", "Sainz", "Bottas", "Stroll"],
    "developing": ["Piastri", "Gasly", "Ocon", "Albon", "Zhou"],
    "rookie": ["Sargeant", "Lawson", "Bearman"]  # Updated for current grid
}

TEAM_TIERS = {
    "top_tier": ["Red Bull Racing", "Mercedes", "Ferrari"],
    "midfield": ["McLaren", "Aston Martin", "Alpine", "Williams"],
    "backmarker": ["Haas", "AlphaTauri", "Alfa Romeo"]
}

# Utility functions for feature weighting
def get_weighted_features(importance_level="all", threshold=0.0):
    """
    Get features based on importance level and weight threshold.
    
    Args:
        importance_level: 'high', 'medium', 'supporting', or 'all'
        threshold: Minimum weight threshold (0.0 to 1.0)
    
    Returns:
        dict: {feature_name: weight} for features meeting criteria
    """
    weighted_features = {}
    
    if importance_level == "all" or importance_level == "high":
        weighted_features.update(FEATURE_WEIGHTS["high_importance"])
    if importance_level == "all" or importance_level == "medium":
        weighted_features.update(FEATURE_WEIGHTS["medium_importance"])
    if importance_level == "all" or importance_level == "supporting":
        weighted_features.update(FEATURE_WEIGHTS["supporting"])
    
    # Filter by threshold
    return {k: v for k, v in weighted_features.items() if v >= threshold}

def get_model_config(model_type):
    """Get configuration for specific model type."""
    return MODEL_FEATURE_CONFIG.get(model_type, {})

def get_top_features(n=10):
    """Get top N features by weight."""
    all_features = get_weighted_features("all")
    sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_features[:n])

def get_circuit_prediction_modifiers(circuit_name=None):
    """Get circuit-specific prediction modifiers for realistic F1 simulation"""
    circuit = circuit_name or DATA_CONFIG["selected_circuit"]
    config = get_circuit_config(circuit)
    
    return {
        "grid_importance": config.get("grid_importance", 0.75),
        "strategy_factor": config.get("strategy_factor", 0.7), 
        "chaos_factor": config.get("chaos_factor", 0.4),
        "overtaking_difficulty": config.get("overtaking_difficulty", "medium"),
        "circuit_type": config.get("circuit_type", "permanent")
    }

def get_circuit_fastf1_name(circuit_name=None):
    """Get FastF1-compatible race name for data collection"""
    circuit = circuit_name or DATA_CONFIG["selected_circuit"]
    config = get_circuit_config(circuit)
    return config.get("fastf1_name", f"{circuit} Grand Prix")