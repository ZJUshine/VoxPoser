# Repository Guidelines

## Project Structure & Module Organization
- `src/` houses the VoxPoser implementation. Key modules include `LMP.py`, `interfaces.py`, `planners.py`, `controllers.py`, `dynamics_models.py`, and `visualizers.py`.
- `src/envs/` contains the RLBench wrapper (`rlbench_env.py`) and task-to-object mapping (`task_object_names.json`).
- `src/configs/` stores environment configs such as `rlbench_config.yaml`.
- `src/prompts/rlbench/` holds prompt templates used by LMPs.
- `src/playground.ipynb` is the primary demo entry point.
- `media/` contains demo assets; `requirements.txt` lists Python deps.

## Build, Test, and Development Commands
- Create a Python 3.9 conda env:
  ```sh
  conda create -n voxposer-env python=3.9
  conda activate voxposer-env
  ```
- Install RLBench + PyRep per the upstream RLBench instructions.
- Install Python deps:
  ```sh
  pip install -r requirements.txt
  ```
- Run the demo notebook (requires a display or RLBench headless setup):
  ```sh
  jupyter notebook src/playground.ipynb
  ```
  Set your OpenAI API key in the first notebook cell.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and PEP 8–style naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep new prompts in `src/prompts/rlbench/` as `.txt` files.
- Update task objects in `src/envs/task_object_names.json` when adding tasks.

## Testing Guidelines
- No automated tests or coverage targets are configured.
- Use the demo notebook as a smoke test (run a simple RLBench task end-to-end).
- If you add tests, place them under `tests/` and document the command (e.g., `pytest`).

## Commit & Pull Request Guidelines
- Git history shows short, direct messages (e.g., “Update README.md”, “Create LICENSE”, “code release”). Keep commits concise and imperative.
- No PR template is present; include a brief summary, test evidence (notebook run + task name), and screenshots/GIFs for visualization changes. Link related issues when applicable.

## Configuration & Secrets
- Do not commit API keys. Keep OpenAI credentials local to your environment or notebook cell.

Use Chinese to answer.