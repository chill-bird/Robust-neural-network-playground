import numpy as np
from numpy import testing
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image
import os
# Testing purposes
import random

def rot_matrix_2x2(rad_angle):
    """ Receive an angle in radians and create a rotation matrix from said angle """
    return


def render(x_cart, angle, animation_width):
    """ Receive x coordinate of a cart and angle(rad) of the pole and draw a scene """

    # Canvas ---------------------------
    width  = 900
    height = 600
    x_offset = width/2
    y_offset = height*0.3

    fig, ax = plt.subplots()
    ax.set(xlim=[0, width], ylim=[0, height])
    ax.set_aspect("equal")

    scale = 0.7 * width/animation_width

    # Cart ---------------------------
    cart_length = 1 * scale
    cart_height = 0.6 * cart_length

    x_cart = scale * x_cart + x_offset # Scale x_coordinate and add offset

    # Cart coordinates
    l = x_cart   - 0.5*cart_length
    r = x_cart   + 0.5*cart_length
    t = y_offset + 0.5*cart_height
    b = y_offset - 0.5*cart_height

    # Cart polygon
    cart_poly = np.array([[l,b], [l,t], [r,t], [r,b]])
    ax.add_patch(patches.Polygon(cart_poly, color="black"))

    # Cart center
    center_vector = np.array([x_cart,y_offset])
    ax.add_patch(patches.Circle(center_vector, radius=0.05 * scale, color="red"))

    # Pole ---------------------------
    pole_length = 3 * scale
    edge_length = 0.3 * scale

    # Pole vector(s)
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], \
                           [math.sin(angle),  math.cos(angle)]])
    pole_vector = rot_matrix @ np.array([0, pole_length])   # Rotate vector via rotation matrix
    edge_vector = rot_matrix @ np.array([edge_length/2, 0]) # Rotate vector by angle
    np.testing.assert_allclose((pole_vector @ edge_vector), 0, rtol=1e-05, atol=1e-05) # actually, the deviation is kinda randomly chosen :D

    # Pole coordinates
    # A) Depiction as vector
    ax.plot([x_cart, x_cart+pole_vector[0]],
            [y_offset, y_offset+pole_vector[1]], color="orange")

    # B) Depiction as polygon
    lb = center_vector - edge_vector
    lt = center_vector + pole_vector - edge_vector
    rt = center_vector + pole_vector + edge_vector
    rb = center_vector + edge_vector
    pole_poly = np.array([lb, lt, rt, rb])
    ax.add_patch(patches.Polygon(pole_poly, color="brown"))

    # Debugging
    ax.text(800, 500, str(math.degrees(angle)) + "°")
    # plt.show()
    return fig

def gif(filenames):
    """ Create GIF from list of images """

    images = []
    for file in filenames:
        frame = Image.open(file)
        images.append(frame)
    images[0].save("cartpole.gif", save_all=True, append_images=images[1:], duration=100)


def test():
    frame_files = []

    for i in range(10):
        frame = render(random.randint(-5,5),math.radians(random.randint(-360, 360)), 2*4.8)
        filename = "frames/" + str(i) + ".png"
        frame.savefig(filename)
        plt.close(frame)
        frame_files.append(filename)

    gif(frame_files)

    # Debugging
    edge1 = render(-4.8,0,2*4.8)
    edge2 = render(4.8,0,2*4.8)
    edge1.savefig("frames/edge1.png")
    edge2.savefig("frames/edge2.png")
