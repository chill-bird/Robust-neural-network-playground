# robust-neural-network-playground

### ML

Install requirements.

### Interface

You need the neural_net.py in the same directory as the html_interface.py.
Install everything from the ML-Part.
To run it: 
1) You need to have flask installed.
2) Direct to the directory where you have the html_interface.py
3) Type "export FLASK_APP=html_interface.py"
4) Type "flask run"
(Attention: Steps 3) and 4) are working just under Linux! On other distributions it might be different.)
5) If everything works, you will get an URL with a port
6) Copy paste this in a browser-window

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
