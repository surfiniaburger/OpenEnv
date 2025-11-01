"""
Data models for Pokemon battle environment.

Action encoding follows Gymnasium-compatible integer system:
- -2: Default action (let server decide)
- -1: Forfeit
- 0-3: Use move at index 0-3
- 4-9: Switch to Pokemon at index 0-5
- 10-13: Use move 0-3 with Mega Evolution
- 14-17: Use move 0-3 with Z-Move
- 18-21: Use move 0-3 with Dynamax
- 22-25: Use move 0-3 with Terastallize

For doubles battles, action contains two sub-actions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.env_server import Action, Observation, State


@dataclass(kw_only=True)
class PokemonAction(Action):
    """
    Action for Pokemon battles.

    Supports both integer encoding (for RL) and structured format (for clarity).
    """
    # Integer encoding (primary - used for RL agents)
    action_id: int = -2  # Default action

    # Structured format (optional - for interpretability)
    action_type: str = "default"  # "move", "switch", "forfeit", "default"
    move_index: Optional[int] = None  # 0-3
    switch_target: Optional[int] = None  # 0-5 (team index)

    # Battle modifiers
    mega: bool = False
    z_move: bool = False
    dynamax: bool = False
    terastallize: bool = False

    # For doubles (if needed)
    move_target: int = 0  # Target position in doubles


@dataclass(kw_only=True)
class PokemonObservation(Observation):
    """
    Observation of Pokemon battle state.

    Contains full battle state including:
    - Active Pokemon on both sides
    - Team information
    - Field conditions (weather, terrain)
    - Legal actions
    - Battle metadata
    """
    # Turn information
    turn: int = 0

    # Active Pokemon state
    active_pokemon: Optional[Dict[str, Any]] = None
    opponent_active_pokemon: Optional[Dict[str, Any]] = None

    # Team state (your team)
    team: List[Dict[str, Any]] = field(default_factory=list)

    # Opponent team (visible info only)
    opponent_team: List[Dict[str, Any]] = field(default_factory=list)

    # Field conditions
    weather: Optional[Dict[str, Any]] = None
    terrain: Optional[Dict[str, Any]] = None
    side_conditions: Dict[str, Any] = field(default_factory=dict)
    opponent_side_conditions: Dict[str, Any] = field(default_factory=dict)

    # Legal actions this turn
    legal_actions: List[int] = field(default_factory=list)
    available_moves: List[Dict[str, Any]] = field(default_factory=list)
    available_switches: List[int] = field(default_factory=list)

    # Battle modifiers available
    can_mega_evolve: bool = False
    can_z_move: bool = False
    can_dynamax: bool = False
    can_terastallize: bool = False
    force_switch: bool = False
    trapped: bool = False

    # Battle status
    battle_finished: bool = False
    battle_won: Optional[bool] = None

    # Team preview
    in_team_preview: bool = False

    # Error handling
    error: Optional[str] = None
    last_action_valid: bool = True


@dataclass
class PokemonState(State):
    """
    Extended state for Pokemon battles.

    Tracks battle-specific information beyond the base State.
    """
    # Battle identification
    battle_tag: str = ""
    format: str = "gen9randombattle"

    # Team configuration
    team_size: int = 6
    team_preview_required: bool = False

    # Battle progress
    total_turns: int = 0
    actions_taken: int = 0

    # Outcome tracking
    pokemon_fainted: int = 0
    opponent_pokemon_fainted: int = 0

    # Server connection
    server_url: str = "localhost"
    server_port: int = 8000
    connected: bool = False


@dataclass(kw_only=True)
class PokemonConfig:
    """
    Configuration for Pokemon environment.

    Used to customize environment behavior.
    """
    # Battle format
    format: str = "gen9randombattle"  # Random battles by default

    # Team (optional - for non-random formats)
    team: Optional[str] = None  # Packed team string

    # Server configuration
    server_url: str = "ws://localhost:8000/showdown/"

    # Battle settings
    max_turns: int = 1000  # Safety limit

    # Reward shaping
    reward_mode: str = "sparse"  # "sparse", "dense", "custom"
    reward_for_faint: float = 0.1  # Reward for fainting opponent Pokemon
    reward_for_damage: float = 0.0  # Reward per damage dealt (if dense)
    penalty_for_damage: float = 0.0  # Penalty per damage taken (if dense)
    reward_for_win: float = 1.0
    reward_for_loss: float = -1.0
    reward_for_tie: float = 0.0
    reward_for_illegal_action: float = -0.1

    # Auto-handling
    auto_team_preview: bool = True  # Use default team ordering
    auto_forfeit_on_timeout: bool = False

    # Account configuration
    username: Optional[str] = None
    password: Optional[str] = None

    # Opponent configuration
    opponent_mode: str = "self"  # "self", "random", "fixed"
    opponent_name: Optional[str] = None
