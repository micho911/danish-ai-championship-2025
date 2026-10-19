# scripts/evaluate.py
import os, argparse, itertools, statistics, json, time
import pygame
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

# Use dummy video driver so Pygame doesn't open a window
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT"]
SENSOR_NAMES = [
    "front","right_front","right_side","right_back","back","left_back","left_side","left_front",
    "left_side_front","front_left_front","front_right_front","right_side_front","right_side_back",
    "back_right_back","back_left_back","left_side_back"
]
MAX_SENSOR = 1000.0
# Lane geometry (normalized vertical ratio r: 0=top, 1=bottom)
LANE_COUNT = 5
LANE_CENTERS = [(2*k - 1)/(2*LANE_COUNT) for k in range(1, LANE_COUNT+1)]  # [0.1, 0.3, 0.5, 0.7, 0.9]

from lane_planner import LanePlannerAgent, LanePlannerParams
from src.game import core

def safety_filter(action_idx, sensors, vx, v_cap=12.0):
    # Balanced shield: avoid accelerating when too close ahead, cap speed, steer away from walls
    front = sensors.get("front") or MAX_SENSOR
    top   = sensors.get("left_side") or MAX_SENSOR
    bot   = sensors.get("right_side") or MAX_SENSOR
    if vx > v_cap and action_idx == 1:  # ACCELERATE
        action_idx = 0
    if front < 250 and action_idx == 1:
        action_idx = 2  # DECELERATE
    if top < 160:
        action_idx = 4  # steer down
    if bot < 160:
        action_idx = 3  # steer up
    return action_idx


# --- SafeCruiseAgent: very conservative survival baseline ---
class SafeCruiseAgent:
    """
    Extremely conservative baseline to verify the pipeline.
    Goal: survive full 60s. Caps speed and recenters gently.
    """
    def __init__(self, v_cap=10.0, ttc_brake=140):
        self.v_cap = v_cap
        self.ttc_brake = ttc_brake

    def _obs_from_state(self):
        sensors = {}
        for s in core.STATE.sensors:
            sensors[s.name] = None if s.reading is None else float(s.reading)
        vx = float(core.STATE.ego.velocity.x)
        top = sensors.get("left_side") or MAX_SENSOR
        bot = sensors.get("right_side") or MAX_SENSOR
        r = top / (top + bot) if (top + bot) > 1e-6 else 0.5
        front = sensors.get("front") or MAX_SENSOR
        return sensors, vx, r, front

    def act_single(self, _request_like_dict):
        sensors, vx, r, front = self._obs_from_state()
        # steer to middle lane (0.5 ratio)
        steer = "NOTHING"
        if r < 0.48:
            steer = "STEER_RIGHT"
        elif r > 0.52:
            steer = "STEER_LEFT"
        # speed control
        if vx < self.v_cap and front > 300:
            speed = "ACCELERATE"
        elif front < 200 or (front / max(vx, 1e-3)) < self.ttc_brake:
            speed = "DECELERATE"
        else:
            speed = "NOTHING"
        # prioritize safety actions over centering
        return speed if speed != "NOTHING" else steer

    def act_batch(self, _request_like_dict):
        a = self.act_single(_request_like_dict)
        return {"actions": [a] * 30}

class ShieldAgent:
    """
    Hard safety controller:
    - Estimates closing rates (px/tick) from sensor deltas.
    - Computes TTC for front/back and predicted clearances for left/right lanes.
    - Prefers SAFE LANE CHANGE over braking; accelerates to avoid rear-end; decelerates for front threat.
    - Keeps a commit window for lane changes (no flapping).
    Goal: survive 60s; distance is secondary.
    """
    def __init__(self, ttc_min_front=100, ttc_min_back=100, wall_margin=180.0,
                 lane_deadband=0.02, commit_ticks=28, v_min=7.0, v_cap=12.0):
        self.ttc_min_front = ttc_min_front
        self.ttc_min_back = ttc_min_back
        self.wall_margin = wall_margin
        self.lane_deadband = lane_deadband
        self.commit_ticks = commit_ticks
        self.v_min = v_min
        self.v_cap = v_cap
        self.prev = {"front": None, "left": None, "right": None, "back": None}
        self.rate = {"front": 0.0, "left": 0.0, "right": 0.0, "back": 0.0}
        self.mode = "CRUISE"; self.target_lane = None; self._commit = 0

    def _closest_lane(self, r):
        return min(range(LANE_COUNT), key=lambda i: abs(r - LANE_CENTERS[i]))

    def _sense(self):
        sensors = {s.name: (None if s.reading is None else float(s.reading)) for s in core.STATE.sensors}
        def get(k, d=MAX_SENSOR): 
            v = sensors.get(k); return d if v is None else v
        front = min(get("front"), get("front_left_front"), get("front_right_front"))
        left  = min(get("left_front"), get("left_side_front"))
        right = min(get("right_front"), get("right_side_front"))
        back  = min(get("back"), get("back_left_back"), get("back_right_back"))
        top, bot = get("left_side"), get("right_side")
        r = top/(top+bot) if (top+bot)>1e-6 else 0.5
        vx = float(core.STATE.ego.velocity.x)
        return sensors, front, left, right, back, r, vx, top, bot

    def _upd_rate(self, front,left,right,back, alpha=0.4):
        for k,c in [("front",front),("left",left),("right",right),("back",back)]:
            p = self.prev[k]
            if p is not None and c < MAX_SENSOR/2 and p < MAX_SENSOR/2:
                inst = max(min(p - c, 200.0), -200.0)
                self.rate[k] = alpha*inst + (1-alpha)*self.rate[k]
            self.prev[k] = c

    def _ttc(self, d, rel): 
        rel = max(rel, 1e-6); return d/rel

    def _steer_to(self, r, lane):
        target = LANE_CENTERS[lane]
        if r < target - self.lane_deadband: return "STEER_RIGHT"
        if r > target + self.lane_deadband: return "STEER_LEFT"
        return "NOTHING"

    def act_single(self, _req):
        sensors, front, left, right, back, r, vx, top, bot = self._sense()
        if top < self.wall_margin:  return "STEER_RIGHT"
        if bot < self.wall_margin:  return "STEER_LEFT"

        self._upd_rate(front,left,right,back)
        ttc_front = self._ttc(front, abs(self.rate["front"])+1.0)
        ttc_back  = self._ttc(back,  abs(self.rate["back"])+1.0)
        curr = self._closest_lane(r)

        left_ok  = curr>0 and ((sensors.get("back_left_back") or MAX_SENSOR) > 180)
        right_ok = curr<LANE_COUNT-1 and ((sensors.get("back_right_back") or MAX_SENSOR) > 180)

        if self._commit>0:
            self._commit -= 1
            step = self._steer_to(r, self.target_lane)
            if step == "NOTHING": self.mode="CRUISE"
            return step

        front_threat = ttc_front < self.ttc_min_front
        back_threat  = ttc_back  < self.ttc_min_back

        H=30
        util_stay  = front - abs(self.rate["front"])*H
        util_left  = (left  - abs(self.rate["left"])*H)  - (0 if left_ok else 1e5)
        util_right = (right - abs(self.rate["right"])*H) - (0 if right_ok else 1e5)
        best = max([("STAY",curr,util_stay),
                    ("LEFT",curr-1,util_left),
                    ("RIGHT",curr+1,util_right)], key=lambda x: x[2])

        if front_threat and best[0] == "LEFT":
            self.mode="CHANGE_LEFT"; self.target_lane=best[1]; self._commit=self.commit_ticks
            return self._steer_to(r, self.target_lane)
        if front_threat and best[0] == "RIGHT":
            self.mode="CHANGE_RIGHT"; self.target_lane=best[1]; self._commit=self.commit_ticks
            return self._steer_to(r, self.target_lane)

        if back_threat and not front_threat and vx < self.v_cap:
            return "ACCELERATE"
        if front_threat:
            return "DECELERATE"

        if vx < self.v_min and front > 250: 
            return "ACCELERATE"
        return self._steer_to(r, curr)

    def act_batch(self, _req): 
        a=self.act_single(_req); return {"actions":[a]*30}
def _collect_request_dict():
    # Build the dto-like dict the agent expects
    sensors_map = {}
    for s in core.STATE.sensors:
        sensors_map[s.name] = None if s.reading is None else float(s.reading)

    return {
        "did_crash": core.STATE.crashed,
        "elapsed_ticks": core.STATE.ticks,   # core.game_loop normally increments; we will keep our own counter
        "distance": core.STATE.distance,
        "velocity": {"x": float(core.STATE.ego.velocity.x), "y": float(core.STATE.ego.velocity.y)},
        "sensors": sensors_map,
    }

def _check_collisions():
    # Mirror the checks in core.game_loop
    for car in core.STATE.cars:
        if car != core.STATE.ego and core.intersects(core.STATE.ego.rect, car.rect):
            core.STATE.crashed = True
            return
    for wall in core.STATE.road.walls:
        if core.intersects(core.STATE.ego.rect, wall.rect):
            core.STATE.crashed = True
            return
# ---------- Gymnasium-like environment for RL ----------
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None

class RaceCarEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}
    def __init__(self, seed_value=None, sensor_removal=0, safety_shield=True):
        if gym is None:
            raise RuntimeError("gymnasium not installed. pip install gymnasium")
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(SENSOR_NAMES)+2,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._seed_value = seed_value
        self._sensor_removal = sensor_removal
        self._safety_shield = safety_shield
        self._ticks = 0

    def _obs(self):
        sensors = {}
        for s in core.STATE.sensors:
            sensors[s.name] = None if s.reading is None else float(s.reading)
        vec = []
        for name in SENSOR_NAMES:
            v = sensors.get(name)
            v = MAX_SENSOR if v is None else v
            vec.append(min(v, MAX_SENSOR)/MAX_SENSOR)
        vx = float(core.STATE.ego.velocity.x)
        # normalized lane ratio r
        top = sensors.get("left_side") or MAX_SENSOR
        bot = sensors.get("right_side") or MAX_SENSOR
        r = top / (top + bot) if (top + bot) > 1e-6 else 0.5
        vec.append(min(max(vx/20.0, 0.0), 1.0))
        vec.append(min(max(r, 0.0), 1.0))
        return np.array(vec, dtype=np.float32), sensors, vx

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed_value = seed
        pygame.init()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        core.initialize_game_state("http://localhost/mock", self._seed_value, sensor_removal=self._sensor_removal)
        core.update_game("NOTHING")
        self._ticks = 0
        obs, _, _ = self._obs()
        return obs, {}

    def step(self, action):
        # optional safety shield
        sensors_map = {s.name: (None if s.reading is None else float(s.reading)) for s in core.STATE.sensors}
        vx = float(core.STATE.ego.velocity.x)
        if self._safety_shield:
            action = safety_filter(action, sensors_map, vx)
        act_str = ACTIONS[action]
        core.update_game(act_str)
        # collisions
        for car in core.STATE.cars:
            if car != core.STATE.ego and core.intersects(core.STATE.ego.rect, car.rect):
                core.STATE.crashed = True
                break
        if not core.STATE.crashed:
            for wall in core.STATE.road.walls:
                if core.intersects(core.STATE.ego.rect, wall.rect):
                    core.STATE.crashed = True
                    break
        self._ticks += 1
        done = core.STATE.crashed or (self._ticks >= 3600)
        # reward shaping
        obs, sensors, vx = self._obs()
        front = sensors.get("front") or MAX_SENSOR
        wall_top = sensors.get("left_side") or MAX_SENSOR
        wall_bot = sensors.get("right_side") or MAX_SENSOR
        # Base reward: scaled forward speed
        rew = vx/11.0
        # Encourage clearance ahead
        rew += min(front, MAX_SENSOR)/MAX_SENSOR * 0.5
        # Small survival bonus
        rew += 0.1
        # Penalties near walls
        if wall_top < 120 or wall_bot < 120:
            rew -= 0.3
        # Crash penalty and success bonus
        if core.STATE.crashed:
            rew -= 1000.0
        elif done:
            # survived full horizon
            rew += 100.0
        return obs, rew, done, False, {}

class InlineRLPolicyAgent:
    def __init__(self, model_path, device="cpu"):
        try:
            from stable_baselines3 import PPO
        except Exception as e:
            raise RuntimeError("stable-baselines3 not installed. pip install stable-baselines3[extra]") from e
        self.model_path = model_path
        self.PPO = PPO
        self.model = self.PPO.load(model_path, device=device)
        self.vecnorm = None
        try:
            vecnorm_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
            if os.path.exists(vecnorm_path):
              self.vecnorm = VecNormalize.load(vecnorm_path)
              self.vecnorm.training = False
              self.vecnorm.norm_reward = False
        except Exception:
            self.vecnorm = None
        self.batch_len = 30
        self.step_repeat = 4
        self.v_cap = 10.0

    def _obs_from_state(self):
        sensors = {}
        for s in core.STATE.sensors:
            sensors[s.name] = None if s.reading is None else float(s.reading)
        vec = []
        for name in SENSOR_NAMES:
            v = sensors.get(name)
            v = MAX_SENSOR if v is None else v
            vec.append(min(v, MAX_SENSOR)/MAX_SENSOR)
        vx = float(core.STATE.ego.velocity.x)
        top = sensors.get("left_side") or MAX_SENSOR
        bot = sensors.get("right_side") or MAX_SENSOR
        r = top / (top + bot) if (top + bot) > 1e-6 else 0.5
        vec.append(min(max(vx/20.0, 0.0), 1.0))
        vec.append(min(max(r, 0.0), 1.0))
        return np.array(vec, dtype=np.float32), sensors, vx

    def act_single(self, _request_like_dict):
        obs, sensors, vx = self._obs_from_state()
        obs_in = obs[None, :]
        if getattr(self, "vecnorm", None) is not None:
            obs_in = self.vecnorm.normalize_obs(obs_in)
        action, _ = self.model.predict(obs_in, deterministic=True)
        action = int(action)
        action = safety_filter(action, sensors, vx, v_cap=self.v_cap)
        return ACTIONS[action]

    def act_batch(self, _request_like_dict):
        # Build a small batch for server compatibility
        first = self.act_single(_request_like_dict)
        return {"actions": [first] * (self.batch_len)}

def run_episode(seed_value, params: LanePlannerParams, sensor_removal=0, max_ticks=3600, render=False, agent=None):  
    if render:
        os.environ.pop("SDL_VIDEODRIVER", None)  # allow window if requested
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    core.initialize_game_state("http://localhost/mock", seed_value, sensor_removal=sensor_removal)

    # one pre-update to populate initial sensor readings
    _ = core.update_game("NOTHING")
    _check_collisions()

    agent = agent or LanePlannerAgent(params)
    ticks = 0
    while not core.STATE.crashed and ticks < max_ticks:
        req = _collect_request_dict()
        action = agent.act_single(req)
        _ = core.update_game(action)
        _check_collisions()
        # advance a pseudo tick counter (core.game_loop does this usually)
        core.STATE.ticks += 1
        ticks += 1

    result = {
        "seed": seed_value,
        "distance": float(core.STATE.distance),
        "ticks": ticks,
        "crashed": bool(core.STATE.crashed),
    }
    pygame.quit()
    return result

def grid_search(seeds, base_params: LanePlannerParams, search_space, sensor_removal=0):
    keys = list(search_space.keys())
    best = None
    all_results = []

    for values in itertools.product(*[search_space[k] for k in keys]):
        params = LanePlannerParams(**base_params.__dict__)  # copy
        for k, v in zip(keys, values):
            setattr(params, k, v)

        distances, crashes = [], 0
        for sd in seeds:
            res = run_episode(sd, params, sensor_removal=sensor_removal)
            distances.append(res["distance"])
            crashes += int(res["crashed"])

        avg = statistics.mean(distances)
        p95 = statistics.quantiles(distances, n=20)[18] if len(distances) >= 20 else max(distances)
        cand = {
            "params": {k: getattr(params, k) for k in params.__dict__.keys()},
            "avg_distance": avg,
            "p95_distance": p95,
            "crash_rate": crashes / len(seeds),
            "seeds": len(seeds),
        }
        all_results.append(cand)
        if (best is None or cand["avg_distance"] > best["avg_distance"]
            or (cand["avg_distance"] == best["avg_distance"] and cand["crash_rate"] < best["crash_rate"])):
            best = cand

        print(f"[GS] {cand}")

    all_results.sort(key=lambda x: (x["avg_distance"], -x["crash_rate"]), reverse=True)
    return best, all_results

def train_rl(total_timesteps=500_000, n_envs=8, sensor_removal=0, save_path="models/ppo_racecar.zip"):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as e:
        raise RuntimeError("stable-baselines3 not installed. pip install stable-baselines3[extra]") from e

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def make_env(rank):
        def _thunk():
            env = RaceCarEnv(seed_value=12345 + rank*997, sensor_removal=sensor_removal, safety_shield=True)
            return env
        return _thunk

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=4096 ,
                gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                policy_kwargs=policy_kwargs, device="cpu", tensorboard_log="runs/ppo_racecar")
    model.learn(total_timesteps=total_timesteps, tb_log_name="PPO")
    model.save(save_path)
    vecnorm_path = os.path.join(os.path.dirname(save_path), "vecnormalize.pkl")
    vec_env.save(vecnorm_path)
    print(f"Saved VecNormalize stats to {vecnorm_path}")
    vec_env.close()
    print(f"Saved PPO model to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20, help="number of random seeds")
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--sensor-removal", type=int, default=0, help="randomly remove N sensors at init")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--grid", action="store_true", help="run a small grid search over key params")
    parser.add_argument("--out", type=str, default=None, help="optional JSON output file")
    parser.add_argument("--policy", type=str, default="planner", choices=["planner","rl","safe", "shield"],)
    parser.add_argument("--rl-model", type=str, default="models/ppo_racecar.zip")
    parser.add_argument("--train", action="store_true", help="Train PPO policy instead of evaluating")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--n-envs", type=int, default=8)
    args = parser.parse_args()
    
    if args.train:
      train_rl(total_timesteps=args.total_timesteps, n_envs=args.n_envs, sensor_removal=args.sensor_removal, save_path=args.rl_model)
      return


    seeds = [args.seed_base + i*7919 for i in range(args.runs)]  # spaced primes-ish

    base = LanePlannerParams()
    t0 = time.time()

    if args.grid:
        if args.policy == "rl":
            print("[WARNING] Grid search is designed for planner policy. Running grid search with planner agent.")
        search_space = {
            "panic_ttc":   [14, 16, 18],
            "brake_ttc":   [32, 36, 40],
            "horizon_ticks": [24, 30, 36],
            "change_cost": [180.0, 220.0, 260.0],
            "behind_safe": [140.0, 160.0, 180.0],
            "ema_alpha":   [0.25, 0.35, 0.5],
        }
        best, all_res = grid_search(seeds, base, search_space, sensor_removal=args.sensor_removal)
        print("\n=== BEST CONFIG ===")
        print(json.dumps(best, indent=2))
        payload = {"mode":"grid", "best": best, "results": all_res, "runtime_sec": time.time()-t0}
    else:
        agent = None
        if args.policy == "rl":
            agent = InlineRLPolicyAgent(args.rl_model)
        elif args.policy == "safe":
            agent = SafeCruiseAgent()
        elif args.policy == "shield":
            agent = ShieldAgent()
        rows = []
        for sd in seeds:
            res = run_episode(sd, base, sensor_removal=args.sensor_removal, render=args.render, agent=agent)            
            print(res)
            rows.append(res)
        dists = [r["distance"] for r in rows]
        payload = {
            "mode":"eval",
            "runs": rows,
            "avg_distance": statistics.mean(dists),
            "p95_distance": statistics.quantiles(dists, n=20)[18] if len(dists) >= 20 else max(dists),
            "crash_rate": sum(int(r["crashed"]) for r in rows)/len(rows),
            "runtime_sec": time.time()-t0
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results -> {args.out}")

if __name__ == "__main__":
    main()