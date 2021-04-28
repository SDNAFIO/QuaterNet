import torch
import numpy as np
import matplotlib.pyplot as plt
from common.visualization import render_animation
from IPython.display import HTML

from short_term.dataset_h36m import dataset as h36m

if torch.cuda.is_available():
  h36m.cuda()
h36m.compute_positions() # Forward kinematics on all animations
render_animation(h36m['S1']['walking_1_d0']['positions_world'], h36m.skeleton(), h36m.fps(), output='interactive')
