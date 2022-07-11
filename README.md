# robust-neural-network-playground

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


### Interface

You need the neural_net.py in the same directory.
Install everything from the ML-Part below. As output you get a list with two numbers. The first one is the leraning rate and the second one is the batch size you called the functions with.

### ML

Install requirements in ML folder.

