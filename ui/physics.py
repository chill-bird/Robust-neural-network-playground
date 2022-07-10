import pygame
import math

class State:

    def __init__(self, coord, speed, angle, ang_speed):
        self.coord = coord
        self.speed = speed
        self.angle = angle
        self.ang_speed = ang_speed

    def update_coord(self):
        self.coord += self.speed
    
    def update_speed(self, acc):
        self.speed += acc

    def update_angle(self):
        self.angle += self.ang_speed

    def update_ang_spd(self, ang_acc):
        self.ang_speed += ang_acc


gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0


def deg2rad(deg):
    return (deg * (math.pi / 180))

def rad2deg(rad):
    return (rad / (math.pi / 180))

def sinalpha(angle):
    math.sin(angle)


def step(state, action):

    angle_rad = deg2rad(state.angle)
    ang_spd_rad = deg2rad(state.ang_speed)

    cosalpha = math.cos(angle_rad)
    sinalpha = math.sin(angle_rad)
    

    force = force_mag if action == 1 else -force_mag

    temp = (force + polemass_length * ang_spd_rad**2 * sinalpha ) / total_mass

    alphaacc = (gravity * sinalpha - cosalpha * temp) / (
            length * (4.0 / 3.0 - masspole * cosalpha**2 / total_mass)
    )
    
    xacc = temp - polemass_length * alphaacc * cosalpha / total_mass

    alphaacc_deg = rad2deg(alphaacc) % 360

    # print(alphaacc_deg)

    state.update_coord()
    state.update_speed(xacc / 100)
    state.update_angle()
    state.update_ang_spd(alphaacc / 100)


def print_state(state):
    print(str(state.coord) + " " + str(state.speed) + " " + str(state.angle) + " " + str(state.ang_speed))

def check_angle(angle, max_angle):
    return (angle<-max_angle or angle>max_angle)

def check_border(coord, min_coord, max_coord):
    return (coord<min_coord or coord>max_coord)


# https://stackoverflow.com/questions/15098900/how-to-set-the-pivot-point-center-of-rotation-for-pygame-transform-rotate
def rotate(surface, angle, pivot, offset):
    """Rotate the surface around the pivot point.

    Args:
        surface (pygame.Surface): The surface that is to be rotated.
        angle (float): Rotate by this angle.
        pivot (tuple, list, pygame.math.Vector2): The pivot point.
        offset (pygame.math.Vector2): This vector is added to the pivot.
    """
    rotated_image = pygame.transform.rotate(surface, -angle)  # Rotate the image.
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot+rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect.


s1 = State(100, 0, 0, 0) # sample initial state

pygame.init()

wn_width=600; wn_height=400
wn=pygame.display.set_mode((wn_width,wn_height))
pygame.display.set_caption("cartpole")

clock=pygame.time.Clock()

BLACK=(0,0,0)
WHITE=(255,255,255)
BROWN=(202, 152, 101)

def draw_track():
    pygame.draw.line(wn, BLACK, (0,300), (600,300))


Y=285
cartwidth=50; cartheight=30
cart = pygame.Surface((cartwidth,cartheight))
cart.fill(BLACK)


polewidth=10; poleheight=100
pole = pygame.Surface((polewidth,poleheight))
pole.fill(BROWN)
pole.set_colorkey(WHITE)

offset = pygame.math.Vector2(0, -50)

pole_rect = pole.get_rect()


i=0 # index

action=1 # move right

fps=30 # frames per second
nof=180 # maximum frame number before game dies
crit_angle = 45 # angle where the game dies
min_x = 25; max_x = 575 # borders

"""
# creates folder to save the frames
if not (os.path.isdir("frames/")):
        os.makedirs("frames/")
"""

running=True
while running & (i<nof): 
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False


    wn.fill(WHITE)

    draw_track()

    x=s1.coord
    alpha=s1.angle
    
    wn.blit(cart,(x-25,Y))

    pivot_point = [x, Y+15]
    rot_pole, pole_rect = rotate(pole, alpha, pivot_point, offset)
    wn.blit(rot_pole,pole_rect)
    
    pygame.display.update()

    """
    # saving the frames 
    filename = "frames/" + str(i+1) + ".png"
    pygame.image.save(wn, filename)
    """

    # print("frames/" + str(i+1) + ".png")

    step(s1,action)

    if check_angle(alpha, crit_angle):
        running=False

    if check_border(x, min_x, max_x):
        running=False
    
    i+=1
    clock.tick(fps)

pygame.quit()
quit()