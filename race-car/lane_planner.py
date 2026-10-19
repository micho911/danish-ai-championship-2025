from typing import Dict, Optional, List
import math

# ----- Geometry / lanes -----
LANE_COUNT = 5
LANE_CENTERS = [(2*k - 1)/(2*LANE_COUNT) for k in range(1, LANE_COUNT+1)]  # [0.1,0.3,0.5,0.7,0.9]
FAR = 1e9

def _g(sensors: Dict[str, Optional[float]], name: str, default=FAR) -> float:
    v = sensors.get(name)
    return v if (v is not None) else default

def _norm_y_ratio(sensors) -> Optional[float]:
    # 0° is "left_side" (top), 180° is "right_side" (bottom) per your core wiring
    top_d = _g(sensors, "left_side", None)
    bot_d = _g(sensors, "right_side", None)
    if top_d is None or bot_d is None:
        return None
    denom = top_d + bot_d
    if denom <= 1e-6:
        return None
    return top_d / denom

def _closest_lane(r: float) -> int:
    best_i, best_d = 0, 1e9
    for i, c in enumerate(LANE_CENTERS):
        d = abs(r - c)
        if d < best_d:
            best_d, best_i = d, i
    return best_i

def _front_clearance(s):
    return min(_g(s, "front", FAR),
               _g(s, "front_left_front", FAR),
               _g(s, "front_right_front", FAR))

def _left_ahead(s):
    return min(_g(s, "left_front", FAR),
               _g(s, "left_side_front", FAR))

def _right_ahead(s):
    return min(_g(s, "right_front", FAR),
               _g(s, "right_side_front", FAR))

def _left_behind(s):
    return _g(s, "back_left_back", FAR)

def _right_behind(s):
    return _g(s, "back_right_back", FAR)

# ----- Agent -----
class LanePlannerParams:
    def __init__(
        self,
        panic_ttc=60,           # ticks (panic sooner)
        brake_ttc=110,          # ticks
        comfy_ttc=200,          # ticks (only accelerate when VERY comfy)
        wall_margin=140.0,      # px
        lane_deadband=0.022,    # normalized r
        commit_change_ticks=30,
        panic_brake_ticks=16,
        horizon_ticks=36,
        change_cost=260.0,
        behind_safe=180.0,
        ahead_safe_for_change=240.0,
        ema_alpha=0.35,
        batch_len=30,
        step_repeat=4,
        v_cap=12.0,             # px/tick – conservative max target speed
        accel_ttc=160,          # do not accelerate unless TTC > this
    ):
        self.panic_ttc = panic_ttc
        self.brake_ttc = brake_ttc
        self.comfy_ttc = comfy_ttc
        self.wall_margin = wall_margin
        self.lane_deadband = lane_deadband
        self.commit_change_ticks = commit_change_ticks
        self.panic_brake_ticks = panic_brake_ticks
        self.horizon_ticks = horizon_ticks
        self.change_cost = change_cost
        self.behind_safe = behind_safe
        self.ahead_safe_for_change = ahead_safe_for_change
        self.ema_alpha = ema_alpha
        self.batch_len = batch_len
        self.step_repeat = step_repeat
        self.v_cap = v_cap
        self.accel_ttc = accel_ttc

class LanePlannerMemory:
    def __init__(self):
        self.mode = "CRUISE"         # CRUISE | CHANGE_LEFT | CHANGE_RIGHT | PANIC_BRAKE
        self.target_lane = None
        self.commit_ticks = 0
        # EMA of closing rates (px/tick) per direction key
        self.rate = {"front": 0.0, "left": 0.0, "right": 0.0}
        self.prev = {"front": None, "left": None, "right": None}

class LanePlannerAgent:
    def __init__(self, params: LanePlannerParams = None):
        self.P = params or LanePlannerParams()
        self.M = LanePlannerMemory()

    # ----- helpers -----
    def _ttc(self, clearance_px: float, vx: float) -> float:
        vx = max(vx, 1e-6)
        return clearance_px / vx

    def _steer_toward_lane(self, r, target_lane) -> str:
        """
        Map normalized ratio r (0=top wall, 1=bottom wall) to steering.
        If r < target → we are too close to top → steer DOWN (STEER_RIGHT).
        If r > target → we are too close to bottom → steer UP (STEER_LEFT).
        """
        if r is None:
            return "NOTHING"
        target = LANE_CENTERS[target_lane]
        if r < target - self.P.lane_deadband:
            return "STEER_RIGHT"  # down
        if r > target + self.P.lane_deadband:
            return "STEER_LEFT"   # up
        return "NOTHING"

    def _safe_to_change_left(self, curr_lane, sensors) -> bool:
        if curr_lane <= 0: return False
        return (_left_ahead(sensors) > self.P.ahead_safe_for_change and
                _left_behind(sensors) > self.P.behind_safe)

    def _safe_to_change_right(self, curr_lane, sensors) -> bool:
        if curr_lane >= LANE_COUNT-1: return False
        return (_right_ahead(sensors) > self.P.ahead_safe_for_change and
                _right_behind(sensors) > self.P.behind_safe)

    def _update_rates(self, front_d: float, left_d: float, right_d: float):
        # instant closing rate (prev - curr). Positive => closing.
        alpha = self.P.ema_alpha
        for key, curr in [("front", front_d), ("left", left_d), ("right", right_d)]:
            prev = self.M.prev[key]
            if prev is not None and curr < FAR/2 and prev < FAR/2:
                inst = max(min(prev - curr, 200.0), -200.0)  # clamp (stability)
                self.M.rate[key] = alpha * inst + (1 - alpha) * self.M.rate[key]
            self.M.prev[key] = curr

    # ----- public API -----
    def act_batch(self, state: Dict) -> Dict[str, List[str]]:
        """For server: returns dict with 'actions': List[str]"""
        actions: List[str] = []
        # Collect inputs
        sensors = state.get("sensors", {}) or {}
        vx = float(state.get("velocity", {}).get("x", 10.0))

        r = _norm_y_ratio(sensors)
        curr_lane = _closest_lane(r) if r is not None else 2

        front_d = _front_clearance(sensors)
        left_d  = _left_ahead(sensors)
        right_d = _right_ahead(sensors)
        self._update_rates(front_d, left_d, right_d)

        ttc_front = self._ttc(front_d, vx)

        # Hard ACDA-like guard: if TTC is already under panic threshold, force PANIC_BRAKE
        if ttc_front < self.P.panic_ttc:
            left_ok  = self._safe_to_change_left(curr_lane, sensors)
            right_ok = self._safe_to_change_right(curr_lane, sensors)
            if not left_ok and not right_ok:
                self.M.mode = "PANIC_BRAKE"; self.M.target_lane = curr_lane; self.M.commit_ticks = self.P.panic_brake_ticks

        # wall push
        top_d = _g(sensors, "left_side", FAR)
        bot_d = _g(sensors, "right_side", FAR)
        wall_push = None
        if top_d < self.P.wall_margin:  wall_push = "STEER_RIGHT"
        if bot_d < self.P.wall_margin:  wall_push = "STEER_LEFT"

        # evaluate candidates
        candidates = []
        H = self.P.horizon_ticks
        def utility(lane, d, rate, change_steps, behind_ok):
            pred = d - rate * H
            penalty = self.P.change_cost * change_steps
            if not behind_ok and change_steps > 0:
                penalty += 1e5  # block unsafe changes
            return pred - penalty

        left_ok  = self._safe_to_change_left(curr_lane, sensors)
        right_ok = self._safe_to_change_right(curr_lane, sensors)

        # stay
        candidates.append(("STAY", curr_lane, utility(curr_lane, front_d, self.M.rate["front"], 0, True)))
        # left
        if curr_lane > 0:
            candidates.append(("LEFT", curr_lane-1, utility(curr_lane-1, left_d, self.M.rate["left"], 1, left_ok)))
        # right
        if curr_lane < LANE_COUNT-1:
            candidates.append(("RIGHT", curr_lane+1, utility(curr_lane+1, right_d, self.M.rate["right"], 1, right_ok)))

        best_move, best_lane, _ = max(candidates, key=lambda x: x[2])

        # state transitions
        if ttc_front < self.P.panic_ttc and not left_ok and not right_ok:
            self.M.mode = "PANIC_BRAKE"; self.M.target_lane = curr_lane; self.M.commit_ticks = self.P.panic_brake_ticks
        else:
            if best_move == "LEFT" and self.M.mode not in ("CHANGE_LEFT","CHANGE_RIGHT"):
                self.M.mode = "CHANGE_LEFT"; self.M.target_lane = best_lane; self.M.commit_ticks = self.P.commit_change_ticks
            elif best_move == "RIGHT" and self.M.mode not in ("CHANGE_LEFT","CHANGE_RIGHT"):
                self.M.mode = "CHANGE_RIGHT"; self.M.target_lane = best_lane; self.M.commit_ticks = self.P.commit_change_ticks
            elif self.M.mode not in ("CHANGE_LEFT","CHANGE_RIGHT"):
                self.M.mode = "CRUISE"; self.M.target_lane = curr_lane

        # build batch
        while len(actions) < self.P.batch_len:
            step = "NOTHING"
            if wall_push:
                step = wall_push
            elif self.M.mode == "PANIC_BRAKE":
                step = "DECELERATE"
            elif self.M.mode == "CHANGE_LEFT":
                step = self._steer_toward_lane(r, self.M.target_lane)
                if step == "NOTHING" and ttc_front < self.P.brake_ttc: step = "DECELERATE"
            elif self.M.mode == "CHANGE_RIGHT":
                step = self._steer_toward_lane(r, self.M.target_lane)
                if step == "NOTHING" and ttc_front < self.P.brake_ttc: step = "DECELERATE"
            else:
                # --- SAFETY SHIELD: conservative speed control ---
                # 1) Never accelerate unless TTC is very high AND under v_cap
                if ttc_front > self.P.accel_ttc and vx < self.P.v_cap:
                    step = "ACCELERATE"
                # 2) Brake early if TTC is getting low (but avoid getting rear-ended if something is very close behind)
                elif ttc_front < self.P.brake_ttc:
                    behind_d = _g(sensors, "back", FAR)
                    if behind_d > self.P.behind_safe:
                        step = "DECELERATE"
                    else:
                        # try a gentle re-centering instead of hard brake if someone is tailgating
                        step = self._steer_toward_lane(r, curr_lane)
                else:
                    # keep lane centering when in the "okay" band
                    step = self._steer_toward_lane(r, curr_lane)

            if self.M.commit_ticks > 0:
                self.M.commit_ticks -= 1
                if self.M.commit_ticks == 0 and self.M.mode in ("CHANGE_LEFT","CHANGE_RIGHT"):
                    self.M.mode = "CRUISE"; self.M.target_lane = curr_lane

            actions.extend([step]*self.P.step_repeat)

        return {"actions": actions[:self.P.batch_len]}

    def act_single(self, state: Dict) -> str:
        # For local/eval: just take the first element of batch
        return self.act_batch(state)["actions"][0]