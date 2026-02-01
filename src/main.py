import os
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.real_env import RealRobotEnv
from utils import set_lmp_objects


os.environ["OPENAI_API_KEY"] = ""  # set your API key here
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1" # set your API base here


config = get_config('rlbench')


# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])
env = RealRobotEnv(visualizer=visualizer)

lmps, lmp_env = setup_LMP(env, config, debug=False)
voxposer_ui = lmps['plan_ui']

objects = ["rubbish", "bin"]
instruction = "put the rubbish in the bin"

set_lmp_objects(lmps, objects)

voxposer_ui(instruction)
