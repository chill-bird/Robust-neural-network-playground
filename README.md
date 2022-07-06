# robust-neural-network-playground

### Render Demo

There are two rendering versions available: **render_polygons.py** and **render_graphics.py**.

To test one of them in the python REPL, move to directory ui with ```cd ui``` and move to your python REPL via ```python```.
Inside the REPL, import the python file via ```import render_polygons as render``` or ```import render_graphics as render``` and execute the test function via ```render.test()```.


### Interface

You need the "render_graphics.py" from the "ui" directory. It has to be stored as "render.py", in the same directory than the "interface.py" from the "gradio" directory. 
The "project.py" in the "gradio" directory is the old version of the "interface.py".

### ML

Install requirements in ML folder.
Change path in "sys.path.insert" (ligne 26-27 in neural_network.py) to own path.
If necessary change device from "CUDA" to "CPU" (ligne 29 in neural_network.py).
