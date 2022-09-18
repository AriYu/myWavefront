from __future__ import annotations

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import time

from typing import Coroutine, Tuple
from typing import List

show_animation = True

if show_animation:
    ax = plt.axes()

class Edge:
    def __init__(self, parent : Sphere, child : Sphere) -> None:
        self.parent = parent
        self.child = child

class Node:
    def __init__(self, x : float, y : float) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return str(self.x) + ", " + str(self.y)

class Coordinate:
    def __init__(self, x : float, y : float) -> None:
        self.x = x
        self.y = y

class Sphere:
    def __init__(self, center: Coordinate, radius : float, parent: Sphere) -> None:
        self.center = center
        self.radius = radius
        self.parent = parent

class PrioritySphereQueueNode:
    def __init__(self, sphere: Sphere, priority: float) -> None:
        self.sphere = sphere
        self.priority = priority

def searchNearestFromList(sample_x : float, sample_y : float, ox : list[float], oy : list[float]) -> Tuple[int, float] :
    min_index = 0
    min_dist = np.hypot(ox[0] - sample_x, oy[0] - sample_y)
    for i in range(len(ox)):
        dist = np.hypot(ox[i] - sample_x, oy[i] - sample_y)
        if (dist < min_dist):
            min_dist = dist
            min_index = i

    return min_index, min_dist

def calcEuclidDistance(p1: Coordinate, p2: Coordinate) -> float:
    return math.sqrt((p1.x-p2.x)**2.0 + (p1.y-p2.y)**2.0)

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

    def isCollision(self, p : Coordinate) -> bool:
        """
        If there are ox or oy in radius rr, return True (colliding)
        """
        _, min_dist = searchNearestFromList(p.x, p.y, self.ox, self.oy)
        if min_dist < self.rr: 
            return True # collide
        else:
            return False # no collide

    def isCollisionPath(self, p0 : Coordinate, p1 : Coordinate) -> bool:
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
            dist = calcEuclidDistance(Coordinate(x,y), pos)
            if (dist < max_distance):
                max_distance = dist
        return max_distance

def get_the_most_lower_priority_index(queue : List[PrioritySphereQueueNode]) -> int:
    index = 0
    prio = queue[0].priority
    for idx, node in enumerate(queue):
        if  node.priority < prio:
            index = idx
    return index

class WaveFront(object):

    def __init__(self, collision_checker : CollisionChecker) -> None:
        self.vertexes : List[Sphere] = []
        self.edges : List[Edge] = []
        self.sphere_que : List[PrioritySphereQueueNode] = []
        self.collision_checker = collision_checker

    def search(self, start : Coordinate, goal : Coordinate, n: int) -> None:
        radius = self.collision_checker.getNearestDistance(start)
        sphere_node = Sphere(start, radius, None)
        self.sphere_que.append(PrioritySphereQueueNode(sphere_node, calcEuclidDistance(goal, start) - radius))

        while True:
            # Get the sphere which is most close to the goal
            self.sphere_que.sort(key=lambda x: x.priority, reverse=True) # sort as descending order
            #sphere_node = self.sphere_que.pop() # get last item from sorted list and remove that. Original method
            sphere_node = self.sphere_que[-1] # get last item from but doesn't remove. My method.
            self.vertexes.append(sphere_node.sphere)
            self.edges.append(Edge(sphere_node.sphere.parent, sphere_node.sphere))

            # Check whether is the sphere reaching the goal.
            if (calcEuclidDistance(goal, sphere_node.sphere.center) < sphere_node.sphere.radius):
                self.edges.append(Edge(sphere_node.sphere, Sphere(goal, 0, sphere_node.sphere)))
                return #Tree(V, E)

            # Sample n points on shpere surface
            added_counter = 0
            for i in range(n):
                theta = random.random() * 2.0 * math.pi
                qnew = Coordinate(sphere_node.sphere.center.x + sphere_node.sphere.radius * math.cos(theta),
                                  sphere_node.sphere.center.y + sphere_node.sphere.radius * math.sin(theta))
                is_outside = True
                # Check the sphere is not inside other spheres
                for existing_sphere in self.vertexes:
                    if existing_sphere is sphere_node:
                        continue
                    if calcEuclidDistance(qnew, existing_sphere.center) < existing_sphere.radius:
                        is_outside = False
                        break
                if is_outside == False:
                    continue
                new_rudius = self.collision_checker.getNearestDistance(qnew)
                if (new_rudius < 2.0): # Reject the too small sphere
                    continue
                new_sphere = Sphere(qnew, new_rudius, sphere_node.sphere)
                self.sphere_que.append(PrioritySphereQueueNode(new_sphere, calcEuclidDistance(goal, qnew) - new_rudius))
                added_counter += 1
            
            # My method: If we cannot add any samples against the sphere, make priority low the sphere in order to select other sphere in next loop.
            if (added_counter == 0):
                self.sphere_que[-1].priority += self.sphere_que[-1].sphere.radius  #add penalty

            if (len(self.sphere_que) == 0):
                print("sphere_que length is zero!")
                break

        return

    def plot(self) -> None:
        for sphere in self.vertexes:
            c = patches.Circle(xy=(sphere.center.x, sphere.center.y), radius=sphere.radius, fc='g', ec='r', fill=False)
            ax.add_patch(c)
        for edge in self.edges:
            if edge.parent != None and edge.child != None:
                plt.plot([edge.parent.center.x, edge.child.center.x], [edge.parent.center.y, edge.child.center.y])


def main():
    #random.seed(1)
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
    start_time = time.perf_counter_ns()
    wave_front_explorer.search(start, goal, 3)
    elapsed = time.perf_counter_ns() - start_time
    print("elapsed time: {0} ms".format(elapsed*1e-6))
    wave_front_explorer.plot()

    if show_animation:
        #plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    main()
