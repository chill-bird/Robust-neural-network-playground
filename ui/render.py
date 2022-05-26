import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# Scale
scale = 100

# Canvas
width  = 4 * scale
height = 3 * scale

# Offset to x- and y-axis
x_offset = width/2
y_offset = 0.5 * scale

# Cart measurements
cart_length = 0.5 * scale
cart_height = 0.3 * scale

# Pole measurements
pole_length = 1 * scale

def render(x_cart, angle):
    """ Receive x coordinate of a cart and angle(rad) of the pole and draw a scene """

    ######### CANVAS #########

    fig, ax = plt.subplots()
    ax.set(xlim=[0, width], ylim=[0, height])


    ######### CART #########

    # Scale x_coordinate and add offset
    x_cart = scale * x_cart + x_offset

    # Cart coordinates
    l = x_cart   - 0.5*cart_length
    r = x_cart   + 0.5*cart_length
    t = y_offset + 0.5*cart_height
    b = y_offset - 0.5*cart_height

    # Cart polygon
    cart_poly = np.array([[l,b], [l,t], [r,t], [r,b]])
    ax.add_patch(patches.Polygon(cart_poly, color="black"))

    # Cart center
    ax.add_patch(patches.Circle([x_cart,y_offset], radius=0.05 * scale, color="red"))

    ######### POLE #########

    # Pole vector
    angle *= -1
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    pole_vector = np.dot(rot_matrix, np.array([0,pole_length]))

    # Pole coordinates
    ax.plot([x_cart, x_cart+pole_vector[0]],
            [y_offset, y_offset+pole_vector[1]], color="orange")
    
    plt.show()


render(0,math.radians(-30))