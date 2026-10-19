"""
GPU-parallel genetic trainer for DM-i-AI 2025 race-car
- Evaluates 50 policies in parallel across available CUDA devices.
Run: python trainer_gpu.py
"""

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless video
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # silence ALSA
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")    # satisfy SDL2 on many distros

import copy
import math
import random
from typing import List

import pygame
import torch
import torch.nn as nn
import torch.multiprocessing as torch_mp

# --- local game imports ------------------------------------------------------
import src.game.core as core

# -------------------- Policy / obs ------------------------------------------

SENSOR_NAMES = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front",
    "left_side_front", "front_left_front",
    "front_right_front", "right_side_front",
    "right_side_back", "back_right_back",
    "back_left_back", "left_side_back",
]
N_SENSORS = len(SENSOR_NAMES)

ACTION_LIST = ["ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT", "NOTHING"]
N_ACTIONS = len(ACTION_LIST)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SENSORS + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def get_observation() -> torch.Tensor:
    """Return 18-D observation on CPU (we'll send to GPU inside worker)."""
    max_d = 1000.0
    # Use explicit None handling so 0.0 does not get replaced
        # Prefer STATE.sensors_dict if present; otherwise fall back to scanning STATE.sensors
    def sensor_value(name: str):
        v = None
        if hasattr(core.STATE, "sensors_dict") and isinstance(core.STATE.sensors_dict, dict):
            v = core.STATE.sensors_dict.get(name, None)
        if v is None:
            # try to find the sensor by name
            for s in core.STATE.sensors:
                if getattr(s, "name", None) == name:
                    v = getattr(s, "reading", None)
                    break
        return max_d if v is None else v

    vals = [sensor_value(name) / max_d for name in SENSOR_NAMES]
    vx = core.STATE.ego.velocity.x / 20.0
    vy = core.STATE.ego.velocity.y / 20.0
    return torch.tensor(vals + [vx, vy], dtype=torch.float32)

# -------------------- Single episode eval (runs in worker) -------------------

def run_episode(net_state_dict: dict, seed_value: int, device_id: int) -> float:
    """
    Evaluate one network on one seed; return distance. Runs in its own process.
    device_id: CUDA device to use (>=0). If no CUDA, pass -1 for CPU fallback.
    """
    # Select device
    if device_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    # Rebuild policy and move to device
    policy = PolicyNet().to(device)
    policy.load_state_dict(net_state_dict)
    policy.eval()

    # Headless env
    try:
        os.makedirs(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), exist_ok=True)
        pygame.mixer.quit()
    except Exception:
        pass
    pygame.init()

    core.initialize_game_state("http://placeholder", seed_value, sensor_removal=0)
    # Build name→value mapping (values filled each step from .reading)
    core.STATE.sensors_dict = {s.name: None for s in core.STATE.sensors}

    with torch.inference_mode():
        for _ in range(core.MAX_TICKS):
            # Update sensors -> obs
            for s in core.STATE.sensors:
                core.STATE.sensors_dict[s.name] = s.reading
            obs = get_observation().to(device, non_blocking=True)
            logits = policy(obs)
            action_idx = int(torch.argmax(logits).item())
            action_str = ACTION_LIST[action_idx]

            # Environment step (matches game_loop ordering)
            core.handle_action(action_str)
            core.STATE.distance += core.STATE.ego.velocity.x
            core.update_cars()
            core.remove_passed_cars()
            core.place_car()

            # Sensors update
            for s in core.STATE.sensors:
                s.update()
                if hasattr(core.STATE, "sensors_dict"):
                    core.STATE.sensors_dict[s.name] = s.reading

            # Collision checks (same as game_loop)
            for car in core.STATE.cars:
                if car != core.STATE.ego and core.intersects(core.STATE.ego.rect, car.rect):
                    core.STATE.crashed = True
            for wall in core.STATE.road.walls:
                if core.intersects(core.STATE.ego.rect, wall.rect):
                    core.STATE.crashed = True

            core.STATE.ticks += 1
            if core.STATE.crashed:
                break

    distance = float(core.STATE.distance)
    pygame.quit()
    return distance

# -------------------- GA (selection / mutation / orchestration) --------------

POP_SIZE     = 50
ELITE_KEEP   = 2
MUT_SIGMA    = 0.05
GENERATIONS  = 10

def mutate(parent: PolicyNet, sigma: float = MUT_SIGMA) -> PolicyNet:
    child = copy.deepcopy(parent)
    with torch.no_grad():
        for p in child.parameters():
            p.add_(torch.randn_like(p) * sigma)
    return child

def evaluate_population(pop: List[PolicyNet], seeds: List[int]) -> List[float]:
    """
    Evaluate individuals in parallel across CUDA devices.
    Each job is (state_dict, seed, device_id).
    """
    assert len(pop) == len(seeds)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("⚠️ No CUDA device found – running on CPU.")
    # Round-robin assign a device_id to each job
    device_ids = [(i % num_gpus) if num_gpus > 0 else -1 for i in range(len(pop))]
    jobs = [(net.state_dict(), s, d) for net, s, d in zip(pop, seeds, device_ids)]

    # multiprocessing pool with spawn (required for CUDA safety)
    ctx = torch_mp.get_context("spawn")
    # Cap workers to avoid oversubscription: a few per GPU is fine
    per_gpu_workers = 4 if num_gpus > 0 else 6
    processes = min(len(pop), max(1, per_gpu_workers * max(1, num_gpus)))

    with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        fitness = pool.starmap(run_episode, jobs)
    return fitness

def ga_train():
    population = [PolicyNet() for _ in range(POP_SIZE)]
    best_overall, best_distance = None, -math.inf

    for gen in range(1, GENERATIONS + 1):
        # Fresh seeds each generation (reduces overfitting to one RNG stream)
        seeds = [random.randint(0, 2**31 - 1) for _ in range(POP_SIZE)]
        fitness = evaluate_population(population, seeds)

        ranked = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
        elites = [copy.deepcopy(net) for _, net in ranked[:ELITE_KEEP]]

        gen_best = ranked[0][0]
        gen_avg  = sum(fitness) / len(fitness)
        print(f"Gen {gen:03d} | best = {gen_best} | avg = {gen_avg}")

        if gen_best > best_distance:
            best_distance = gen_best
            best_overall  = copy.deepcopy(ranked[0][1])
            torch.save(best_overall.state_dict(), "best_policy.pt")

        # Next population: keep elites, mutate to refill
        next_pop = elites[:]
        while len(next_pop) < POP_SIZE:
            parent = random.choice(elites)
            next_pop.append(mutate(parent, MUT_SIGMA))
        population = next_pop

    print(f"\nDone. Best distance = {best_distance:.1f} px (saved to best_policy.pt)")

if __name__ == "__main__":
    # Reduce CPU contention a bit when many workers are active
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    ga_train()