# robust-neural-network-playground

### Import render_graphics

To import render_graphics.py, please insert these lines at the top of your code
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

You need the "render_graphics.py" from the "ui" directory. It has to be stored as "render.py", in the same directory than the "interface.py" from the "gradio" directory. 
The "project.py" in the "gradio" directory is the old version of the "interface.py".

### ML

Install requirements in ML folder.

