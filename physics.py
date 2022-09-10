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
theta_threshold_radians = 90 * 2 * math.pi / 360
x_threshold = 2.4


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

    terminated = bool(
            state.coord < -x_threshold
            or state.coord > x_threshold
            or state.angle < -theta_threshold_radians
            or state.angle > theta_threshold_radians
        )
    return terminated


def print_state(state):
    print(str(state.coord) + " " + str(state.speed) + " " + str(state.angle) + " " + str(state.ang_speed))

def check_angle(angle, max_angle):
    return (angle<-max_angle or angle>max_angle)

def check_border(coord, min_coord, max_coord):
    return (coord<min_coord or coord>max_coord)
