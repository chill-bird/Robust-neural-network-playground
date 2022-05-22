"""
Tests
"""
import arcade
import math
import numpy as np

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SCREEN_TITLE = "Tests"

# TODO: Update with actual value
MOVEMENT_SPEED_dummy = 20

# TODO: Update with actual curved movement
class Bat(arcade.Sprite):
    """ Bat Class """

    def update(self):
        """ Move Bat """

        self.center_x += self.change_x

        # Check for out-of-bounds
        if self.left < 0:
            self.left = 0
        elif self.right > arcade.get_viewport()[1]:
            self.right = arcade.get_viewport()[1]

# TODO: Implement actual functionality/physics
class Stick(arcade.Sprite):
    """ Stick Class """

    # Distance from bottom to center of stick:
    def __init__(self, image, scale):
        """ Set up the stick """

        # Call the parent init
        super().__init__(image, scale)
        # Create variable to hold distance from one end of the stick to the center
        self.middle = (self.top-self.bottom)/2
    
    def vectorize(self, x_bottom, y_bottom, x_top, y_top):
        """ Return vector with length self.length """

        # Calculate vector with gradient
        v = np.array([(x_top - x_bottom), (y_top - y_bottom)])
        # Return vector witdh length self.length by
        # normalizing and multiplying with length
        v = v * self.middle/np.linalg.norm(v)
        return v

    def rotate(self, vector):
        """ Return rotation angle of the stick """
        if vector[0] == 0:
            return 0
        gradient = vector[1]/vector[0]
        return np.arctan(gradient)
        
    def update(self, x_bottom, y_bottom, x_top, y_top):
        """ Set angle and position of stick """
        vector = self.vectorize(x_bottom, y_bottom, x_top, y_top)
        self.angle += self.rotate(vector)
        # New center position is vector starting at x-coordinate of bat 
        self.center_x = x_bottom + vector[0]
        self.center_y = y_bottom + vector[1]


class MyGame(arcade.Window):
    """ Main application class """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)

        # Scene Object
        self.scene = None

        # Bat
        self.bat_list = None
        self.bat_sprite = None
        # Stick
        self.stick_list = None
        self.stick_sprite = None

        arcade.set_background_color(arcade.color.WHITE_SMOKE)

    def setup(self):
        """ Start a new game """

        # Create the Sprite lists
        self.bat_list = arcade.SpriteList()
        self.stick_list = arcade.SpriteList()

        # Add Bat to displayed scene
        self.bat_sprite = Bat("bat.png", 0.2)
        self.bat_sprite.center_x = arcade.get_viewport()[1]/2
        self.bat_sprite.center_y = 50
        self.bat_list.append(self.bat_sprite)

        # Add Stick to displayed scene
        self.stick_sprite = Stick("stick.png", 0.2)
        self.stick_sprite.center_x = self.bat_sprite.center_x
        self.stick_sprite.bottom = self.bat_sprite.top
        self.stick_list.append(self.stick_sprite)

    def on_resize(self, width, height):
        """ This method is automatically called when the window is resized. """

        # Call the parent. Failing to do this will mess up the coordinates,
        # and default to 0,0 at the center and the edges being -1 to 1.
        super().on_resize(width, height)

    def on_draw(self):
        """ Render the screen. """

        self.clear()
        self.bat_list.draw()
        self.stick_list.draw()

    # TODO: Implement actual rotated bat
    def on_key_press(self, key, modifiers):
        """ Processes pressed keys """

        if key == arcade.key.LEFT or key == arcade.key.A:
            self.bat_sprite.change_x = -MOVEMENT_SPEED_dummy
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.bat_sprite.change_x = MOVEMENT_SPEED_dummy
        elif key == arcade.key.R:
            self.stick_sprite.change_angle = -100
        elif key == arcade.key.T:
            self.stick_sprite.change_angle = 100


    # TODO: Implement actual rotated bat
    def on_key_release(self, key, modifiers):
        """ Processes released keys """

        if key == arcade.key.LEFT or key == arcade.key.A \
            or key == arcade.key.RIGHT or key == arcade.key.D:
            self.bat_sprite.change_x = 0
        elif key == arcade.key.R or key == arcade.key.T:
            self.stick_sprite.change_angle = 0

    # TODO
    def on_update(self, delta_time):
        """ Movement and game logic """

        # Move the bat with the physics engine
        self.bat_list.update()
        # TODO: Update with actual functionality
        # Randomize position and rotation of stick
        x_bottom = self.bat_sprite.center_x
        y_bottom = self.bat_sprite.top
        self.stick_list[0].update(x_bottom, y_bottom, x_bottom + self.stick_sprite.change_angle, y_bottom+50)



def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
