"""
Render Tests
"""
import arcade

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
SCREEN_TITLE = "Tests"

# TODO
MOVEMENT_SPEED_dummy = 5 


class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=True)

        # Scene Object
        self.scene = None

        # Bat
        self.bat_sprite = None

        # Stick
        self.stick_sprite = None

        arcade.set_background_color(arcade.color.BLUE_GRAY)

    # TODO
    def setup(self): 
        """ Start a new game """

        # Initialize Scene
        self.scene = arcade.Scene()

        # Create the Sprite lists
        self.scene.add_sprite_list("Bat")
        self.scene.add_sprite_list("Stick")
 
        # Add Bat to displayed scene
        # TODO: Centering on resize
        self.bat_sprite = arcade.Sprite(":resources:images/tiles/planetHalf.png", 1)
        self.bat_sprite.center_x = SCREEN_WIDTH/2
        self.bat_sprite.center_y = 20
        self.scene.add_sprite("Bat", self.bat_sprite)

        # Add Stick to displayed scene
        self.stick_sprite = arcade.Sprite(":resources:images/items/gold_4.png", 1)
        self.stick_sprite.center_x = self.bat_sprite.center_x
        self.stick_sprite.bottom = self.bat_sprite.top
        self.scene.add_sprite("Stick", self.stick_sprite)

    # TODO
    def on_key_press(self, key, modifiers):
        """ Processes pressed keys """
    
    def on_draw(self):
        """ Render the screen. """

        self.clear()
        self.scene.draw()


def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()