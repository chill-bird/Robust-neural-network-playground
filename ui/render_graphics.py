import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import os
# Testing purposes
import random

# Images
background_orig = Image.open("graphics/background.png")
pole_orig = Image.open("graphics/pole.png")
pole_orig_w, pole_orig_h = pole_orig.size
flame_orig = Image.open("graphics/flame.png")
flame_orig_w, flame_orig_h = flame_orig.size

# TODO: Do scaling of images only once for performance

def render(x_cart, angle, animation_width):
    """
    Receive x coordinate of a cart and angle(rad) of the pole and draw a scene.
    Coordinates start with (0.0) at the upper left corner of the background image.
    """

    # Scene ---------------------------
    width  = 900
    height = 600
    x_offset = width/2
    y_offset = height - height*0.3 # reversed y-axis

    # Create a scene
    assert(width/height == background_orig.size[0]/background_orig.size[1]), "Background image must have ratio 3:2"
    scene = background_orig.resize((width, height))

    # Create draw object
    draw = ImageDraw.Draw(scene)

    # Scale
    scale = 0.7 * width/animation_width

    # Cart ---------------------------
    cart_length = 2 * scale
    cart_height = 0.6 * cart_length

    x_cart = scale * x_cart + x_offset # Scale x_coordinate and add offset
    y_cart = y_offset

    # Draw cart
    l = round(x_cart - 0.5*cart_length)
    r = round(x_cart + 0.5*cart_length)
    t = round(y_cart - 0.5*cart_height)
    b = round(y_cart + 0.5*cart_height)
    draw.rectangle((l,t,r,b), fill=(0,0,0))

    # Draw cart center
    radius = scale * 0.05
    draw.ellipse((x_cart-radius, y_offset-radius, x_cart+radius, y_offset+radius), fill=(255,0,0))

    # Pole ---------------------------
    pole_height = 3 * scale
    resize_factor = pole_height / pole_orig_h
    pole_width = pole_orig_w * resize_factor

    # Image resizing & rotating
    pole_im = pole_orig.resize((round(pole_width), round(pole_height)))
    pole_im = pole_im.rotate(math.degrees(angle), expand=1)

    # Pole vector
    # Vector pointing from the cart center to the center of the pole position
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], \
                           [math.sin(angle),  math.cos(angle)]])
    pole_vector = rot_matrix @ np.array([0, - pole_height/2])

    # Pole upper left corner coordinates
    # Starting at the center of the pole, subtract edge length of new, rotated image
    pole_left  = x_cart - pole_vector[0] - (pole_im.size[0] / 2)
    pole_upper = y_cart + pole_vector[1] - (pole_im.size[1] / 2)

    # Placing image
    scene.paste(pole_im, (round(pole_left), round(pole_upper)), mask=pole_im)

    # Flame ---------------------------
    flame_height = 3 * scale
    resize_factor = flame_height / flame_orig_h
    flame_width = flame_orig_w * resize_factor

    # Image resizing & rotating
    flame_im = flame_orig.resize((round(flame_width), round(flame_height)))

    # Flame upper left corner coordinates
    # Starting at the tip of the pole
    flame_x_offset = 0.03 * flame_im.size[0] # flame graphic is slightly off-centered
    flame_left  = x_cart - 2*pole_vector[0] - (flame_im.size[0] / 2) + flame_x_offset
    flame_upper = y_cart + 2*pole_vector[1] - (flame_im.size[0] / 2)

    # Placing image
    scene.paste(flame_im, (round(flame_left), round(flame_upper)), mask=flame_im)

    # Display angle
    # TODO: Replace magic numbers
    l = 780
    r = 880
    t = 50
    b = 80
    draw.rectangle((l,t,r,b), fill=(255,255,255))
    draw.text((800, 50), f"{round(math.degrees(angle), 2)}°", fill=(0,0,0), font=ImageFont.truetype("NotoSans-Regular.ttf", 20), align="right")

    return scene

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
        frame = render(random.randint(-5,5),math.radians(random.randint(-360, 360)), 2*4.8)
        filename = "frames/" + str(i) + ".png"
        frame.save(filename)
        frame_files.append(filename)

    gif(frame_files)

    # Debugging
    edge1 = render(-4.8,0,2*4.8)
    edge2 = render(4.8,0,2*4.8)
    edge1.save("frames/edge1.png")
    edge2.save("frames/edge2.png")
