"""
Render Tests
"""
import arcade

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SCREEN_TITLE = "Tests"

# TODO
MOVEMENT_SPEED_dummy = 5


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


class Stick(arcade.Sprite):
    """ Stick Class """

    def update(self, bat_list):
        self.center_x = bat_list[0].center_x


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

        arcade.set_background_color(arcade.color.BLUE_GRAY)

    # TODO
    def setup(self):
        """ Start a new game """

        # Create the Sprite lists
        self.bat_list = arcade.SpriteList()
        self.stick_list = arcade.SpriteList()

        # Add Bat to displayed scene
        self.bat_sprite = Bat(":resources:images/tiles/planetHalf.png", 1)
        self.bat_sprite.center_x = arcade.get_viewport()[1]/2
        self.bat_sprite.center_y = 20
        self.bat_list.append(self.bat_sprite)

        # Add Stick to displayed scene
        self.stick_sprite = Stick(":resources:images/items/gold_4.png", 3)
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

    # TODO
    def on_key_press(self, key, modifiers):
        """ Processes pressed keys """

        if key == arcade.key.LEFT:
            self.bat_sprite.change_x = -MOVEMENT_SPEED_dummy
            self.stick_sprite.change_x = -MOVEMENT_SPEED_dummy
        elif key == arcade.key.RIGHT:
            self.bat_sprite.change_x = MOVEMENT_SPEED_dummy
            self.stick_sprite.change_x = -MOVEMENT_SPEED_dummy


    def on_key_release(self, key, modifiers):
        """ Processes released keys """

        if key == arcade.key.LEFT or key == arcade.key.A \
            or key == arcade.key.RIGHT or key == arcade.key.D:
            self.bat_sprite.change_x = 0

    # TODO
    def on_update(self, delta_time):
        """ Movement and game logic """

        # Move the bat with the physics engine
        self.bat_list.update()
        self.stick_list[0].update(self.bat_list)



def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
