import pygame
from typing import Dict, List
from src.game import core
from lane_planner import LanePlannerAgent, LanePlannerParams

# Single agent instance (persists memory across calls)
AGENT = LanePlannerAgent(LanePlannerParams())

def _collect_request_dict() -> Dict:
    sensors_map = {}
    for s in core.STATE.sensors:
        sensors_map[s.name] = None if s.reading is None else float(s.reading)
    return {
        "did_crash": core.STATE.crashed,
        "elapsed_ticks": core.STATE.ticks,
        "distance": core.STATE.distance,
        "velocity": {"x": float(core.STATE.ego.velocity.x), "y": float(core.STATE.ego.velocity.y)},
        "sensors": sensors_map,
    }

# --- For FastAPI server usage ---
def return_action(state: Dict) -> Dict[str, List[str]]:
    """Used by api.py to serve actions to the competition server."""
    return AGENT.act_batch(state)

# --- For local visual run (monkeypatch core.get_action) ---
def agent_get_action():
    """core.game_loop expects a list of actions; feed a batch from the agent."""
    req = _collect_request_dict()
    return AGENT.act_batch(req)["actions"]

if __name__ == '__main__':
    seed_value = 565318  # or None for random
    pygame.init()
    # Plug our agent into the core loop
    core.get_action = agent_get_action
    core.initialize_game_state("http://example.com/api/predict", seed_value)
    core.game_loop(verbose=True)  # Opens a window and renders
    pygame.quit()