import math
import random
import sys
import os
import shutil
ui_path = os.path.abspath(os.getcwd() + "/ui") 
sys.path.append(ui_path)
from render_graphics import *

class Episode:

    # Only use radians instead of converting degrees and radians back and forth
    def __init__(self, coord, speed, angle, ang_speed, episode):
        self.coord = coord
        self.speed = speed
        self.angle = angle
        self.ang_speed = ang_speed
        self.episode = episode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0

        self.max_angle = math.radians(45)
        self.max_coord = 250
        self.min_coord = -self.max_coord

 
    def update_coord(self):
        self.coord += self.speed
    
    def update_speed(self, acc):
        self.speed += acc

    def update_angle(self):
        self.angle += self.ang_speed

    def update_ang_spd(self, ang_acc):
        self.ang_speed += ang_acc
    
    def step(self, action):

        cosalpha = math.cos(self.angle)
        sinalpha = math.sin(self.angle)        

        force = self.force_mag if (action == 1) else -self.force_mag

        temp = (force + self.polemass_length * self.ang_speed**2 * sinalpha ) / self.total_mass

        alphaacc = (self.gravity * sinalpha - cosalpha * temp) \
                    / (self.length * (4.0 / 3.0 - self.masspole * cosalpha**2 / self.total_mass))
        
        xacc = temp - self.polemass_length * alphaacc * cosalpha / self.total_mass

        self.update_coord()
        self.update_speed(xacc/100)
        self.update_angle()
        self.update_ang_spd(alphaacc/100)

    def print_state(self):
        print(str(self.coord) + " " + str(self.speed) + " " + str(math.degrees(self.angle)) + " " + str(self.ang_speed))

    def outside_angle_boundry(self):
        return (self.angle < -self.max_angle or self.angle > self.max_angle)

    def outside_x_boundry(self):
        return (self.coord < self.min_coord or self.coord > self.max_coord)


# TODO: Real interface here
def get_action(state):
    return random.randint(0,1)


def main():
    # Initial state
    episode_number = 1
    state = Episode(0, 0, float(0), 0, episode_number)

    fps = 30 # frames per second
    max_frame_no = 180 # maximum frame number before game dies

    # creates folder to save the frames
    if not (os.path.isdir(str(episode_number) + "_frames/")):
        os.makedirs(str(episode_number) + "_frames/")
    else:
        shutil.rmtree(os.path.abspath(os.getcwd() + "/" + str(episode_number) + "_frames/"))
        os.makedirs(str(episode_number) + "_frames/")

    frame_no = 0
    
    while (frame_no < max_frame_no): 

        x     = state.coord
        alpha = state.angle
        
        # saving the frames 
        # TODO: Solve the rgb array/image conversion problematic
        frame = Image.fromarray(render(state.coord, state.angle, False, 1))
        filename = str(episode_number) + "_frames/" + str(frame_no) + ".png"
        frame.save(filename)

        state.step(get_action(state))

        if state.outside_angle_boundry():
            break
        if state.outside_x_boundry():
            break
        
        frame_no += 1
    
    # Once while loop is exited, the game renders game over screen
    frame = Image.fromarray(render(state.coord, state.angle, True, 1))
    filename = str(episode_number) + "_frames/" + str(frame_no) + ".png"
    frame.save(filename)

if __name__ == "__main__":
    main()