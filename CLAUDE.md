# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoxPoser is a research demo using LLMs (GPT-4) to zero-shot synthesize robotic manipulation trajectories by composing 3D value maps in voxel space. Runs in RLBench simulation. No training required.

## Setup & Running

```bash
conda create -n voxposer-env python=3.9
conda activate voxposer-env
# Install PyRep and RLBench per https://github.com/stepjam/RLBench#install
pip install -r requirements.txt
```

Entry point: `src/playground.ipynb` (Jupyter notebook). Requires OpenAI API key set in the first cell. Requires display (or RLBench headless mode).

There is no build system, test suite, or CLI — this is a research demo run interactively.

## Architecture

Pipeline flow:
```
Natural language instruction
  → Composer LMP (decomposes into sub-tasks)
    → Planner LMP (generates code for sub-tasks)
      → Perception LMPs (affordance/avoidance/velocity/rotation/gripper maps)
        → PathPlanner (greedy optimization over value maps)
          → Controller (waypoints → robot actions)
            → RLBench environment
```

### Core Modules (in `src/`)

- **`LMP.py`** — Wraps OpenAI API. Builds prompts from templates, executes LLM-generated Python in restricted namespace. Disk caching via `LLM_cache.py` (→ `./cache/`).
- **`interfaces.py`** — Bridge between LLM code and environment. `detect(obj_name)` for perception, `execute()` orchestrates value maps + planning + control. `setup_LMP()` initializes all 7 LMP instances.
- **`planners.py`** — Greedy path planner: weighted target/obstacle maps → cost map → greedy voxel selection → Savitzky-Golay smoothing.
- **`controllers.py`** — End-effector centric (direct pose) or object centric (MPC with 10k samples).
- **`dynamics_models.py`** — Heuristic pushing model for object-centric MPC.
- **`envs/rlbench_env.py`** — RLBench wrapper: object masks, point clouds, workspace bounds.

### Prompts & Config

- Prompts: `src/prompts/rlbench/` (8 files with few-shot Python examples)
- Config: `src/configs/rlbench_config.yaml` (all 7 LMPs configured here)
- Coordinate frame: x=back-to-front, y=left-to-right, z=bottom-to-up

## Real-World Adaptation

Replace only `rlbench_env.py` with new environment wrapper (same APIs). LMP and planning modules unchanged. Suggested perception: OWL-ViT + SAM 2. Controller: OSC via Deoxys.
