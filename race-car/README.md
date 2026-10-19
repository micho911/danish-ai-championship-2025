# Race Car

**Task** ([full description](TASK.md)): drive a car down a five-lane road with traffic, using only 16 distance sensors and the car's velocity. Each tick the API returns a batch of actions (`ACCELERATE`, `DECELERATE`, `STEER_LEFT`, `STEER_RIGHT`, `NOTHING`), and the score is distance covered before crashing or running out of time.

## Approach

I tried three families of controllers:

1. **Rule-based lane planner** ([`lane_planner.py`](lane_planner.py)): estimates the current lane from the left/right wall sensors, measures clearance ahead and behind in neighbouring lanes, and changes lane when the current one is blocked. [`evaluate.py`](evaluate.py) **grid-searches** its parameters over many seeds.
2. **PPO** (Stable-Baselines3, in [`evaluate.py`](evaluate.py)): a Gymnasium wrapper around the game engine, trained with `VecNormalize` across parallel environments. [`api.py`](api.py) serves the trained policy and filters every action through a **safety shield** (speed cap, no accelerating into close traffic, steering away from walls).
3. **Genetic algorithm** ([`trainer.py`](trainer.py)): neuroevolution of a small MLP policy, with populations evaluated in parallel across CUDA devices on headless pygame.

## Status

Experimental. `api.py` expects a trained PPO model at `models/ppo_racecar.zip` (or `$RL_MODEL_PATH`), and no weights are included in this repository.

## What I learned

- Framing a game as an RL environment: observation normalisation, reward shaping and action batching.
- A hand-tuned heuristic is a strong baseline, and a safety layer on top of a learned policy is a cheap way to avoid catastrophic actions.
- Running headless simulations in parallel on GPU nodes.
