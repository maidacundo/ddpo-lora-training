import numpy as np

prompts = [
    "white wall",
    "red bricks wall",
    "wooden wall",
]

def prompt_from_list():
    return  np.random.choice(prompts)