import numpy as np
import math
from PIL import Image, ImageDraw
import os
# Testing purposes
import random    

# Images
pole_orig = Image.open("graphics/pole.png")
pole_orig_w, pole_orig_h = pole_orig.size
flame_orig = Image.open("graphics/flame.png")
flame_orig_w, flame_orig_h = flame_orig.size

def render(x_cart, angle, animation_width):
    """ 
    Receive x coordinate of a cart and angle(rad) of the pole and draw a scene.
    Coordinates start with (0.0) at the upper left corner of the background image. 
    """

    # Canvas ---------------------------
    width  = 900
    height = 600
    x_offset = width/2
    y_offset = height - height*0.3 # reversed y-axis

    # Create a canvas
    # TODO: Replace Canvas with background image
    canvas = Image.new("RGB", (width, height), color=(255,255,255))

    # Create draw object
    draw = ImageDraw.Draw(canvas)

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

    # Cart center
    radius = scale * 0.05 
    draw.ellipse((x_cart-radius, y_offset-radius, x_cart+radius, y_offset+radius), fill=(255,0,0))

    # Pole ---------------------------
    pole_length = 3 * scale
    pole_resize_factor = pole_length / pole_orig_h
    pole_width = pole_orig_w * pole_resize_factor

    # Image resizing & rotating
    pole_im = pole_orig.resize((round(pole_width), round(pole_length)))
    pole_im = pole_im.rotate(math.degrees(angle), expand=1)

    # Pole vector
    # Vector pointing from the cart center to the center of the pole position
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], \
                           [math.sin(angle),  math.cos(angle)]])
    pole_vector = np.dot(rot_matrix,
                         np.array([0, - pole_length/2])) 

    # Pole upper left corner coordinates
    # Starting at the center of the pole, subtract edge length of new, rotated image
    pole_x_left  = x_cart   - pole_vector[0] - (pole_im.size[0] / 2)
    pole_y_upper = y_cart + pole_vector[1] - (pole_im.size[1] / 2)
    assert(pole_x_left >= 0 and pole_y_upper >= 0), "Pole outside of the canvas"

    # Placing images
    canvas.paste(pole_im, (round(pole_x_left), round(pole_y_upper)), mask=pole_im)

    # Debugging
    # ax.text(800, 500, str(math.degrees(angle)) + "°")
    # plt.show()

    return canvas    

def gif(filenames):
    """ Create GIF from list of images """
    
    images = []
    for file in filenames:
        frame = Image.open(file)
        images.append(frame)
    images[0].save("cartpole.gif", save_all=True, append_images=images[1:], duration=100)

def main():

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

if __name__ == "__main__":
    main()