import pygame
import math
import render_graphics as render

class State:

    def __init__(self, coord, speed, angle, ang_speed):
        self.coord = coord
        self.speed = speed
        self.angle = angle
        self.ang_speed = ang_speed

    def update_coord(self):
        self.coord += self.speed
    
    def update_speed(self, new_speed):
        self.speed = new_speed

    def update_angle(self):
        self.angle += self.ang_speed

    def check_angle(self):
        return (self.angle >= 90 or self.angle <= -90)

def step(state):
    state.update_coord()
    state.update_angle()
    state.check_angle()



s1 = State(6, 4, 19, 2)
s2 = State(25, -5, -91, 4)
s3 = State(236, 12, 31, -1)



pygame.init()

clock=pygame.time.Clock()

WHITE=(255,255,255)
BLACK=(0,0,0)
BROWN=(202, 152, 101)
GREY=(129, 132, 203)


wn_width=600
wn_height=400
wn=pygame.display.set_mode((wn_width,wn_height))
pygame.display.set_caption("cartpole")

running=True
while running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            state=False

    wn.fill(WHITE)

    pygame.draw.line(wn, BLACK, (0,300), (600,300))

    X=s1.coord; Y=285

    frame = render.render(X, 0, False)
    frame.save("frame.png")
    
    # render.render(X, 0, False)



    """
    cardwidth=50; cardheight=30
    pygame.draw.rect(wn, BLACK, (X-25, Y, cardwidth, cardheight))

    polewidth=10; poleheight=100
    pygame.draw.rect(wn, BROWN, (X-5, Y-85, polewidth, poleheight))
    """

    pygame.display.update()
    clock.tick(30)


pygame.quit()
quit()



"""
print("state 1")
print(s1.coord)
print(s1.angle)


for i in range(10):
    step(s1)


step(s1)

#s1.update_coord()
#s1.update_angle()

print(s1.coord)
print(s1.angle)


print()
print("state 2")
print(s2.coord)
print(s2.angle)

print(s2.check_angle())

s2.update_coord()
s2.update_angle()

print(s2.coord)
print(s2.angle)

print(s2.check_angle())
"""