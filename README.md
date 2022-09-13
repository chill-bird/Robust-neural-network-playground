# robust-neural-network-playground

### ML

Install requirements.

### Interface

You need the neural_net.py in the same directory.
Install everything from the ML-Part. You need a random mp4-video in the same directory. The video has to be named 'video.mp4'.

### Import render_graphics

To import render_graphics.py, please ensure that you are outside of the ui directory.
Insert these lines at the top of your code
```
import sys
import os
ui_path = os.path.abspath(os.getcwd() + "/ui/") 
sys.path.append(ui_path)
from render_graphics import *
```

Now you can simply call render like this from within your code:
```
render(x_cart, angle, game_over, episode_num)
```
See test_graphics.py as reference.
