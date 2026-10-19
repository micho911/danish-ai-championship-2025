import os
import time
import glob
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import uvicorn
from fastapi import Body, FastAPI
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

# ----------------------------
# Logging
# Use uvicorn's error logger channel so messages always show up under uvicorn.
# Also attach a fallback handler if uvicorn did not configure handlers yet.
# ----------------------------
log = logging.getLogger("uvicorn.error")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[api] %(message)s'))
    log.addHandler(_h)
    log.propagate = False

# ----------------------------
# Config
# ----------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9052"))
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "models/ppo_racecar.zip")  # file or directory
V_CAP = float(os.getenv("V_CAP", "8.0"))
BATCH_LEN = 5  # <-- fixed batch length per request (no hot reload)

# ----------------------------
# PPO loader + safety filter
# ----------------------------
ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT"]
SENSOR_NAMES = [
    "front","right_front","right_side","right_back","back","left_back","left_side","left_front",
    "left_side_front","front_left_front","front_right_front","right_side_front","right_side_back",
    "back_right_back","back_left_back","left_side_back"
]
MAX_SENSOR = 1000.0

def resolve_model(path_like: str) -> str:
    """
    If a directory is passed, pick the newest .zip.
    If a file is passed, return it as-is.
    """
    p = Path(path_like)
    if p.is_file():
        return str(p)
    # directory
    zips = sorted(glob.glob(str(p / "*.zip")), key=os.path.getmtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No .zip models in {p}")
    return zips[0]

def per_model_vecnorm_path(model_path: str) -> List[Path]:
    """
    Return candidate VecNormalize paths in priority order:
    1) <stem>.vecnorm.pkl (preferred)
    2) vecnormalize.pkl (legacy)
    """
    mp = Path(model_path)
    stem = mp.with_suffix("").name
    dirp = mp.parent
    return [dirp / f"{stem}.vecnorm.pkl", dirp / "vecnormalize.pkl"]

def safety_filter(action_idx: int, sensors: Dict[str, Optional[float]], vx: float, v_cap: float = V_CAP) -> int:
    """
    Match evaluate.py's shield-lite: avoid accel into close front, cap speed, steer off walls.
    """
    front = sensors.get("front") if sensors.get("front") is not None else MAX_SENSOR
    top   = sensors.get("left_side") if sensors.get("left_side") is not None else MAX_SENSOR
    bot   = sensors.get("right_side") if sensors.get("right_side") is not None else MAX_SENSOR
    # cap speed against over-accel
    if vx > v_cap and action_idx == 1:  # ACCELERATE
        action_idx = 0
    # don't accelerate if front is tight
    if front < 250 and action_idx == 1:
        action_idx = 2  # DECELERATE
    # nudge away from walls
    if top < 160:
        action_idx = 4  # steer down
    if bot < 160:
        action_idx = 3  # steer up
    return action_idx

class PPOPolicyServerAgent:
    """
    Loads a PPO model saved with Stable-Baselines3 and (optionally) VecNormalize stats.
    Builds observations DIRECTLY from the incoming DTO (no pygame STATE dependency).
    """
    def __init__(self, model_path_like: str):
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import VecNormalize
        except Exception as e:
            raise RuntimeError("stable-baselines3 not installed. pip install stable-baselines3[extra]") from e

        resolved = resolve_model(model_path_like)
        self.model_path = resolved
        self.PPO = PPO
        self.model = self.PPO.load(self.model_path, device="cpu")

        # load vecnormalize if present
        self.vecnorm = None
        for cand in per_model_vecnorm_path(self.model_path):
            if cand.exists():
                try:
                    self.vecnorm = VecNormalize.load(str(cand))
                    self.vecnorm.training = False
                    self.vecnorm.norm_reward = False
                    log.info(f"Loaded VecNormalize stats: {cand}")
                    break
                except Exception:
                    self.vecnorm = None
        log.info(f"Loaded PPO model: {self.model_path}")

        self.batch_len = BATCH_LEN   # fixed at 5 per request
        self.v_cap = V_CAP
        log.info(f"Configured batch_len={self.batch_len} v_cap={self.v_cap}")

    @staticmethod
    def _obs_from_request(req: Dict) -> np.ndarray:
        """
        Build observation vector to match training pipeline:
        - 16 sensors (missing -> MAX, then /1000)
        - speed_x normalized (/20, clipped 0..1)
        - lane ratio r ~ left_side / (left_side + right_side) (falls back to 0.5)
        """
        sensors = req.get("sensors", {}) or {}
        vec = []
        for name in SENSOR_NAMES:
            v = sensors.get(name)
            if v is None:
                v = MAX_SENSOR
            v = min(max(float(v), 0.0), MAX_SENSOR)
            vec.append(v / MAX_SENSOR)

        vx = float(req.get("velocity", {}).get("x", 0.0))
        left = sensors.get("left_side")
        right = sensors.get("right_side")
        left = MAX_SENSOR if left is None else float(left)
        right = MAX_SENSOR if right is None else float(right)
        r = left / (left + right) if (left + right) > 1e-6 else 0.5

        vec.append(min(max(vx / 20.0, 0.0), 1.0))
        vec.append(min(max(r, 0.0), 1.0))
        return np.asarray(vec, dtype=np.float32)

    def act_batch(self, req: Dict) -> Dict[str, List[str]]:
        obs = self._obs_from_request(req)[None, :]  # (1, obs_dim)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)

        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        # apply safety filter on top of PPO
        vx = float(req.get("velocity", {}).get("x", 0.0))
        sensors = req.get("sensors", {}) or {}
        action = safety_filter(action, sensors, vx, v_cap=self.v_cap)

        act_str = ACTIONS[action]
        actions = [act_str] * self.batch_len
        # log the action + batch once per request
        front_val = sensors.get('front') if sensors.get('front') is not None else MAX_SENSOR
        log.info(f"predict -> action={act_str} batch_len={self.batch_len} vx={vx:.2f} front={front_val:.1f}")
        return {"actions": actions}

# FastAPI app
app = FastAPI()
start_time = time.time()
_AGENT: Optional[PPOPolicyServerAgent] = None

def _load_agent():
    global _AGENT
    _AGENT = PPOPolicyServerAgent(RL_MODEL_PATH)

@app.on_event("startup")
def on_start():
    _load_agent()
    log.info(f"startup -> model_path={getattr(_AGENT, 'model_path', None)} batch_len={getattr(_AGENT, 'batch_len', None)} v_cap={getattr(_AGENT, 'v_cap', None)}")

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    if _AGENT is None:
        _load_agent()
    action = _AGENT.act_batch(request.dict())
    log.info(f"/predict returned {len(action['actions'])} actions")
    return RaceCarPredictResponseDto(actions=action['actions'])

@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "model_path": getattr(_AGENT, "model_path", None),
        "batch_len": getattr(_AGENT, "batch_len", None),
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your RL endpoint is running!"

if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT, log_level="info")