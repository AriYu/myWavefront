from ctypes.wintypes import FLOAT
import math
from tkinter import Y
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Tuple
from typing import List

show_animation = True

if show_animation:
    ax = plt.axes()

class Edge:
    def __init__(self, to : int, w : float) -> None:
        self.to = to # 隣接頂点番号
        self.w = w # 重み

class Node:
    def __init__(self, x : float, y : float) -> None:
        self.x = x
        self.y = y
        self.edges : List[Edge] = []

    def addEdge(self, edge) -> None:
        self.edges.append(edge)

    def addEdge(self, to : int, cost : float) -> None:
        edge = Edge(to, cost)
        self.edges.append(edge)

    def __str__(self) -> str:
        return str(self.x) + ", " + str(self.y)

class Coordinate:
    def __init__(self, x : float, y : float) -> None:
        self.x = x
        self.y = y

class Sphere:
    def __init__(self, center: Coordinate, radius : float) -> None:
        self.center = center
        self.radius = radius

def searchNearestFromList(sample_x : float, sample_y : float, ox : list[float], oy : list[float]) -> Tuple[int, float] :
    min_index = 0
    min_dist = np.hypot(ox[0] - sample_x, oy[0] - sample_y)
    for i in range(len(ox)):
        dist = np.hypot(ox[i] - sample_x, oy[i] - sample_y)
        if (dist < min_dist):
            min_dist = dist
            min_index = i

    return min_index, min_dist

class CollisionChecker(object):
    """
    Collision checker for path planning
    """
    def __init__(self,  ox : list[float], oy :list[float], rr : float) -> None:
        """
        ox : The list for x of obstacles
        oy : The list for y of obstacles
        rr : The thresold to determine collide with obstacles
        """
        self.ox = ox
        self.oy = oy
        self.rr = rr
        self.max_x : float = max(ox)
        self.min_x : float = min(ox)
        self.max_y : float = max(oy)
        self.min_y : float = min(oy)

    def isCollision(self, p : Node) -> bool:
        """
        If there are ox or oy in radius rr, return True (colliding)
        """
        _, min_dist = searchNearestFromList(p.x, p.y, self.ox, self.oy)
        if min_dist < self.rr: 
            return True # collide
        else:
            return False # no collide


    def isCollisionPath(self, p0 : Node, p1 : Node) -> bool:
        """
        p0, p1 : the vertexes which compromize a line
        Check collision between the line and ox, oy by rr.
        If there is collision, return True.
        """
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        yaw = math.atan2(dy, dx)
        d = np.hypot(dx, dy)

        D = self.rr
        n_step = round(d / D)

        x = p0.x
        y = p0.y

        for i in range(n_step):
            _, dist = searchNearestFromList(x, y, self.ox, self.oy)
            if dist <= self.rr:
                return True # collide
            x += D * math.cos(yaw)
            y += D * math.sin(yaw)
        
        return False # no collide           

    def getNearestDistance(self, pos: Coordinate) -> float:
        max_distance = 1000000.0 # should be max...
        for x, y in zip(self.ox, self.oy):
            dist = math.sqrt((x-pos.x)**2.0 + (y-pos.y)**2.0)
            if (dist < max_distance):
                max_distance = dist
        return max_distance

class WaveFront(object):

    def __init__(self, collision_checker : CollisionChecker) -> None:
        self.graph : List[Node] = []
        self.spheres : List[Sphere] = []
        self.collision_checker = collision_checker

    def search(self, start : Coordinate, goal : Coordinate, n: int) -> None:
        self.spheres.append(Sphere(start, self.collision_checker.getNearestDistance(start)))

        for i in range(10):
            qnew = Coordinate(0,0)
            qnew.x = (random.random() * (self.collision_checker.max_x - self.collision_checker.min_x)) + self.collision_checker.min_x
            qnew.y = (random.random() * (self.collision_checker.max_y - self.collision_checker.min_y)) + self.collision_checker.min_y
            self.spheres.append(Sphere(qnew, self.collision_checker.getNearestDistance(qnew)))

        return

    def plot(self) -> None:
        #print("x :" +  str(self.sphere.center.x) + ", y:" + str(self.sphere.center.y) + ", r:" + str(self.sphere.radius))
        for sphere in self.spheres:
            c = patches.Circle(xy=(sphere.center.x, sphere.center.y), radius=sphere.radius, fc='g', ec='r')
            ax.add_patch(c)


def main():
    random.seed(1)
    print("{} start !".format(__file__))

    sx = 10.0
    sy = 10.0
    gx = 50.0
    gy = 50.0
    robot_size = 5.0

    ox = []
    oy = []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40)
        oy.append(60.0 - i)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    collision_checker = CollisionChecker(ox, oy, 2.0)
    wave_front_explorer = WaveFront(collision_checker)

    start = Coordinate(sx, sy)
    goal = Coordinate(gx, gy)
    wave_front_explorer.search(start, goal, 5)
    wave_front_explorer.plot()

    if show_animation:
        #plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    main()
