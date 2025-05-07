import math
import random
import sys

from PIL import ImageTk, Image

class Pigeon:
    # This is the distance the pigeon can see in any direction around itself.
    viewing_distance = 1000 * (1.25/30)
    radius = 15 # The radius of the pigeon image on the screen
    memory_radius = 1000 * (1.25/30) # Radius the pigeon remembers around its home loft. This is a value shown in research (4km ~ 2.5 miles)

    # All the passive, active and geo values are needed to give the pigeon it's geomagnetic guidance, it's memory of passive
    # objects around it's loft and ensuring that it doesn't initialise itself too close to the loft.
    def __init__(self, name, xsize, ysize, passive_objects, active_objects, geomag_map):
        self.image = None # Used to prevent the picture being garbage cleaned by python
        self.saved_image = None # Used so it doesn't have to reload the image each time.

        self.alive = True
        self.name = name
        self.xv = 0 # x velocity (pixels per second)
        self.yv = 0 # y velocity (pixels per second)
        self.x = random.randint(0 + self.radius, xsize - self.radius)
        self.y = random.randint(0 + self.radius, ysize - self.radius)

        # Performance Metrics
        self.no_moves = 0 # This is used as a performance metric to see how many move updates it took the pigeon to reach its goal
        self.dist_from_loft = 1000 # This is used at the end as a performance metric to see how close it got to the goal.

        # This is used to scan through all active objects, to ensure the starting point of the pigeon is valid
        overlap_flag = True
        while overlap_flag and active_objects:
            for item in active_objects:
                    if item.getClass() == "Predator":
                        # This ensures the bird cannot be left in a predator's area
                        if intersection(item.x, item.y, self.x, self.y, item.radius, self.radius):
                            self.x = random.randint(0 + self.radius, xsize - self.radius)
                            self.y = random.randint(0 + self.radius, ysize - self.radius)
                            overlap_flag = True
                            break
                        else:
                            overlap_flag = False
                    elif item.getClass() == "Loft":
                        # This is used to find the geomagnetic location of the loft:
                        self.geomag_loft = geomag_map[item.x//100][item.y//100]

                        # This is used to ensure the bird isn't spawned too close to the loft:
                        self.dist_from_loft = math.sqrt((self.x - item.x) ** 2 + (self.y - item.y) ** 2)

                        # 666 is 2/3 of 1000 or 20/30 possible miles. The bird must be dropped at least 20 miles from the loft
                        if self.dist_from_loft < 666:
                            self.x = random.randint(0 + self.radius, xsize - self.radius)
                            self.y = random.randint(0 + self.radius, ysize - self.radius)
                            overlap_flag = True
                            break
                        else:
                            overlap_flag = False

        # Sets the starting geomagnetic values, after x and y are set
        self.current_geomag_loc = geomag_map[self.x//100][self.y//100] # This is the current geomagnetic vector for the pigeons location

        # This sets the starting vision of the pigeon, but has to be set after to ensure x and y are set correctly.
        self.pigeon_vision = self.getVision(passive_objects, active_objects)

        # This is used to generate the pigeons memory of the key points within 4km of it's home loft, as shown in research
        loft = [x for x in active_objects if x.getClass() == "Loft"][0] # This retrieves the loft instance (There should only be one)
        in_range = []
        for item in passive_objects:
            if intersection(loft.x, loft.y, item.midpoint[0], item.midpoint[1], self.memory_radius, item.radius):
                # This function below is used to get the closest point on the circumference of the item to the loft, this creates the map
                close_x, close_y, dist = get_closest_point(loft.x, loft.y, item.midpoint[0], item.midpoint[1], item.radius)
                # It is then added to the list and stored in the pigeon's memory.
                in_range.append((close_x, close_y, dist, item.name))
        for item in active_objects:
            if item.getClass() != "Loft" and intersection(loft.x, loft.y, item.x, item.y, self.memory_radius, item.radius):
                # This function below is used to get the closest point on the circumference of the item to the loft, this creates the map
                close_x, close_y, dist = get_closest_point(loft.x, loft.y, item.x, item.y, item.radius)
                # It is then added to the list and stored in the pigeon's memory.
                in_range.append((close_x, close_y, dist, item.name))
        self.memory_home = in_range

    def drawPigeon(self, canvas):
        # This circle shows the pigeon's field of view. It is not large enough to see an entire square, but adds another level of complexity
        canvas.create_oval(self.x - self.viewing_distance, self.y - self.viewing_distance, self.x + self.viewing_distance,
                           self.y + self.viewing_distance, fill='', outline="black", width=1, tag="view_distance")

        # It's necessary to load the image outside of tkinter first, to resize and rotate it appropriately
        if not self.saved_image:
            self.saved_image = Image.open("data/images/Pigeon.png").convert("RGBA").resize((30,30))
        angle = math.degrees(math.atan(self.xv/(self.yv + sys.float_info.epsilon)))
        if self.yv >= 0:
            angle += 180
        rotated_image = self.saved_image.rotate(angle) # Used to get the appropriate angle to rotate the image by.
        # The self.image has to be included to prevent the image from being garbage collected at the end of the function
        self.image = ImageTk.PhotoImage(rotated_image)

        canvas.create_image(self.x, self.y, image=self.image, anchor="center", tag=f"{self.name}")
        canvas.tag_raise(f"{self.name}")
        canvas.tag_raise("view_distance")

    # This is the movement per second, currently
    def move(self, canvas):
        self.x += self.xv
        self.y += self.yv

        # Handle boundary collisions
        if self.x >= 1000:
            self.x = 999
        elif self.x <= 0:
            self.x = 1
        if self.y >= 1000:
            self.y = 999
        elif self.y <= 0:
            self.y = 1

        canvas.delete(self.name)
        canvas.delete("view_distance")
        self.drawPigeon(canvas)

    # Used to update all aspects of the pigeon, each turn, including movement.
    # Takes inputs of the relevant objects as well as the angle the pigeon is moving in and the number of milliseconds before each update (used to calculate velocity)
    def update(self, canvas, passive_objects, active_objects, geomag_map, move_angle, update_speed):
        # Sets the velocity values used to determine movement
        hypotenuse = 1000 * (0.0138888889/30) # This ensures constant speed. 0.0138... is the real value of 50mph converted to mps and scaled with 30.
        self.xv = math.sin(math.radians(move_angle)) * hypotenuse
        self.yv = math.cos(math.radians(move_angle)) * hypotenuse

        # Runs the movement and changes the xvalue, unless the pigeon is dead already
        if self.alive:
            self.move(canvas)

        # Checks that the pigeon is alive, and not in a predator area, this function also works out probability of death and
        self.pigeonInDanger(active_objects)

        # If alive, updates vision and geomagnetic location.
        if self.alive:
            self.pigeon_vision = self.getVision(passive_objects, active_objects)
            self.current_geomag_loc = self.updateCurrentGeomag(geomag_map)

            # Updates the performance metrics
            self.no_moves += 1
            loft = [x for x in active_objects if x.getClass() == "Loft"][0]  # This retrieves the loft instance (There should only be one)
            self.dist_from_loft = math.sqrt((self.x - loft.x) ** 2 + (self.y - loft.y) ** 2)

    # Used to get the view of the current objects in the pigeons view.
    def getVision(self, passive_objects, active_objects):
        in_range = []
        for item in passive_objects:
            if intersection(self.x, self.y, item.midpoint[0], item.midpoint[1], self.viewing_distance, item.radius):
                # This function below is used to get the closest point on the circumference of the item to the loft, this creates the map
                close_x, close_y, dist = get_closest_point(self.x, self.y, item.midpoint[0], item.midpoint[1], item.radius)
                # It is then added to the list and stored in the pigeon's memory.
                in_range.append((close_x, close_y, dist, item.name))
        for item in active_objects:
            if intersection(self.x, self.y, item.x, item.y, self.viewing_distance, item.radius):
                # This function below is used to get the closest point on the circumference of the item to the loft, this creates the map
                close_x, close_y, dist = get_closest_point(self.x, self.y, item.x, item.y, item.radius)
                # It is then added to the list and stored in the pigeon's memory.
                in_range.append((close_x, close_y, dist, item.name))

        return in_range

    # Used to update the pigeons current geomagnetic position, to an accuracy of one grid square.
    def updateCurrentGeomag(self, geomag_map):
        return geomag_map[int(self.x//100)][int(self.y//100)]

    # Used to identify if the pigeon is in the sphere of the predator, and calculate chance it is killed, updating self.alive if it is.
    def pigeonInDanger(self, active_objects):
        if self.alive:
            for item in active_objects:
                if item.getClass() == "Predator":
                    # If the bird is in the predator area
                    if intersection(item.x, item.y, self.x, self.y, item.radius, 0):
                        stillAlive = random.uniform(0,1)
                        if stillAlive <= item.chance_of_death:
                            self.alive = False

    # Used to get absolute values of the differences between the vectors, used to analyse performance
    def geomagDifference(self):
        diff = []
        for x in range(0, len(self.current_geomag_loc)):
            diff.append(abs(self.current_geomag_loc[x] - self.geomag_loft[x]))

        return diff

# This is used to show if two circles intersect
def intersection(x1, y1, x2, y2, r1, r2):
    bet_center = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    if bet_center <= r1 - r2:
        return True
    elif bet_center <= r2 - r1:
        return True
    elif bet_center < r1 + r2:
        return True
    else:
        return False

# Used to find the point on the circumference of a circle, closest to a point outside of it.
def get_closest_point(x1, y1, x2, y2, r2):
    bet_center = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + sys.float_info.epsilon
    cx = x2 + r2*((x1-x2)/bet_center)
    cy = y2 + r2*((y1-y2)/bet_center)
    dist_point_to_circ = bet_center - r2

    if dist_point_to_circ <= 0: # If it is inside the circle, then this should be indicated by returning 0 0 0
        return 0, 0, 0

    return cx, cy, dist_point_to_circ



