import sys
import os
import random
ui_path = os.path.abspath(os.getcwd() + "/ui/") 
sys.path.append(ui_path)
from render_graphics import *

def gif(filenames):
    """ Create GIF from list of images """

    images = []
    for file in filenames:
        frame = Image.open(file)
        images.append(frame)
    images[0].save("cartpole.gif", save_all=True, append_images=images[1:], duration=100)

def test():
    """ Test rendering with random values """

    if not (os.path.isdir("frames/")):
        os.makedirs("frames/")

    frame_files = []

    for i in range(10):
        frame = Image.fromarray(render(random.randint(-5,5), math.radians(random.randint(-360, 360)), False, 1, 1000))
        filename = "frames/" + str(i) + ".png"
        frame.save(filename)
        frame_files.append(filename)

    gif(frame_files)

    # Debugging
    edge1 = Image.fromarray(render(-4.8,float(0), False, 1, 1))
    edge2 = Image.fromarray(render( 4.8,float(0), False, 1, 1))
    gameover_screen = Image.fromarray(render(random.randint(-5,5),math.radians(random.randint(-360, 360)), True, 1, 99))
    edge1.save("frames/edge1.png")
    edge2.save("frames/edge2.png")
    gameover_screen.save("frames/gameover.png")

test()