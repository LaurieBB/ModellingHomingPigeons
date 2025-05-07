import math
import tkinter as tk
import random
from PIL import Image, ImageTk
import numpy as np

from generate_map import generate_map

# This is used to show if two circles intersect, used often to stop objects overlapping in the environment.
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

class Road:
    def __init__(self, name, pointA, pointB):
        if pointA.getClass() == "City":
            if pointB.getClass() == "City":
                self.width = 5
                self.start_point = pointA.midpoint
                self.end_point = pointB.midpoint
            elif pointB.getClass() == "Town":
                self.width = 5
                self.start_point = pointA.midpoint
                self.end_point = pointB.midpoint
            else:
                self.name = "NO CONNECTION"
                return
        elif pointA.getClass() == "Town":
            if pointB.getClass() == "City":
                self.width = 5
                self.start_point = pointA.midpoint
                self.end_point = pointB.midpoint
            elif pointB.getClass() == "Town":
                self.width = 3
                self.start_point = pointA.midpoint
                self.end_point = pointB.midpoint
            else:
                self.width = 1
                self.start_point = pointA.midpoint
                self.end_point = pointB.midpoint
        elif pointA.getClass() == "Village":
            if pointB.getClass() == "City":
                self.name = "NO CONNECTION"
                return
            elif pointB.getClass() == "Town":
                self.width = 1
                self.start_point = pointA.midpoint
                self.end_point = pointB.midpoint
            else:
                self.width = 1
                self.start_point = pointA.midpoint
                self.end_point = pointB.midpoint
        else:
            self.name = "NO CONNECTION"
            return

        self.name = name

    def drawRoad(self, canvas):
        canvas.create_line(self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1],
                           width=self.width, fill="black", tag="road")
        canvas.tag_lower("road")

class City:
    # This is to convert the size in miles, as discussed in the report, to the equivalent pixel size.
    radius = 1000 * (2.25 / 30)

    def __init__(self, name, xsize, ysize, passive_objects):
        self.name = name
        self.midpoint = [random.randint(0, xsize), random.randint(0, ysize)]

        # Attempting to ensure that the settlements spawned do not overlap
        overlap_flag = True
        while overlap_flag and passive_objects:
            for item in passive_objects:
                if intersection(item.midpoint[0], item.midpoint[1], self.midpoint[0], self.midpoint[1], item.radius, self.radius):
                    self.midpoint = [random.randint(0, xsize), random.randint(0, ysize)]
                    overlap_flag = True
                    break
                else:
                    overlap_flag = False

    def drawCity(self, canvas):
        canvas.create_oval(self.midpoint[0] - self.radius, self.midpoint[1] - self.radius,
                           self.midpoint[0] + self.radius, self.midpoint[1] + self.radius, fill="grey44", tag="city")
        canvas.tag_lower("city")

    @staticmethod
    def getClass():
        return "City"

class Town:
    # This is to convert the size in miles, as discussed in the report, to the equivalent pixel size.
    radius = 1000 * (0.5 / 30)

    def __init__(self, name, xsize, ysize, passive_objects):
        self.name = name
        self.midpoint = [random.randint(0, xsize), random.randint(0, ysize)]

        # Attempting to ensure that the settlements spawned do not overlap
        overlap_flag = True
        while overlap_flag and passive_objects:
            for item in passive_objects:
                if intersection(item.midpoint[0], item.midpoint[1], self.midpoint[0], self.midpoint[1], item.radius, self.radius):
                    self.midpoint = [random.randint(0, xsize), random.randint(0, ysize)]
                    overlap_flag = True
                    break
                else:
                    overlap_flag = False

    def drawTown(self, canvas):
        canvas.create_oval(self.midpoint[0] - self.radius, self.midpoint[1] - self.radius,
                           self.midpoint[0] + self.radius, self.midpoint[1] + self.radius, fill="grey63", tag="town")
        canvas.tag_lower("town")

    @staticmethod
    def getClass():
        return "Town"

class Village:
    # This is to convert the size in miles, as discussed in the report, to the equivalent pixel size.
    radius = 1000 * (0.2 / 30)

    def __init__(self, name, xsize, ysize, passive_objects):
        self.name = name
        self.midpoint = [random.randint(0, xsize), random.randint(0, ysize)]

        # Attempting to ensure that the settlements spawned do not overlap
        overlap_flag = True
        while overlap_flag and passive_objects:
            for item in passive_objects:
                if intersection(item.midpoint[0], item.midpoint[1], self.midpoint[0], self.midpoint[1], item.radius, self.radius):
                    self.midpoint = [random.randint(0, xsize), random.randint(0, ysize)]
                    overlap_flag = True
                    break
                else:
                    overlap_flag = False

    def drawVillage(self, canvas):
        canvas.create_oval(self.midpoint[0] - self.radius, self.midpoint[1] - self.radius,
                           self.midpoint[0] + self.radius, self.midpoint[1] + self.radius, fill="grey77", tag="village")
        canvas.tag_lower("village")

    @staticmethod
    def getClass():
        return "Village"

# The map showing the geomagnetic values for each of the squares.
class GeomagneticMap:
    def __init__(self):
        self.Map = generate_map()

    # Used to draw the 100x100 pixel grid on the  to show the Geomagnetic map
    def drawGrid(self, canvas, xsize, ysize):
        # Vertical lines
        for y in range(0, round(xsize/100)):
            canvas.create_line([(y*100, 0), (y*100, ysize)], tag='grid_line')

        # Horizontal lines
        for x in range(0, round(ysize/100)):
            canvas.create_line([(0, x*100), (xsize, x*100)], tag='grid_line')

    # Returns the map value for a specific row and column in the grid
    def getGeomagneticValues(self, row, col):
        return self.Map[row][col]

# The predator that will influence the chances of the pigeon reaching the loft, if it passes through it's area.
class Predator:
    # This radius is taken from https://enviroliteracy.org/what-is-the-range-of-the-peregrine-falcon/ and is the lowest value.
    radius = int(1000 * (6.25 / 30))
    chance_of_death = 0.01

    def __init__(self, name, xsize, ysize, passive_objects, active_objects):
        self.name = name
        # Including teh radius to find the midpoint so the region cannot spawn outside the map
        self.x = random.randint(0 + self.radius, xsize - self.radius)
        self.y = random.randint(0 + self.radius, ysize - self.radius)
        self.image = None # This is required to ensure images aren't garbage collected by python

        # Attempting to ensure that the settlements spawned do not overlap
        overlap_flag = True
        while overlap_flag and passive_objects:
            for item in passive_objects:
                if intersection(item.midpoint[0], item.midpoint[1], self.x, self.y, item.radius,
                                self.radius - (1000 * (5 / 30))): # I had to adjust the radius here, as it is too large to not overlap
                    self.x = random.randint(0 + self.radius, xsize - self.radius)
                    self.y = random.randint(0 + self.radius, ysize - self.radius)
                    overlap_flag = True
                    break
                else:
                    # This is used to ensure multiple predators or lofts don't overlap at all.
                    for act_item in active_objects:
                        if intersection(act_item.x, act_item.y, self.x, self.y, act_item.radius, self.radius):
                            self.x = random.randint(0 + self.radius, xsize - self.radius)
                            self.y = random.randint(0 + self.radius, ysize - self.radius)
                            overlap_flag = True
                            break
                    overlap_flag = False

    def drawPredator(self, canvas):
        # These circles indicate increasing danger areas for the pigeon.
        canvas.create_oval(self.x - self.radius, self.y - self.radius,
                           self.x + self.radius, self.y + self.radius, fill="#FF9238", tag="predator_area")
        canvas.create_text(self.x, self.y + self.radius, anchor="s", text=f"{self.chance_of_death*100}%")

        self.image = ImageTk.PhotoImage(Image.open("data/images/Tree.png").convert("RGBA").resize((40, 40)))

        canvas.create_image(self.x, self.y, image=self.image, anchor="center", tag="treeImg")
        canvas.tag_lower("treeImg")
        canvas.tag_lower("predator_area")


    @staticmethod
    def getClass():
        return "Predator"

# This is the destination for the pigeon
class Loft:
    radius = 20

    def __init__(self, name, xsize, ysize):
        self.name = name
        self.x = (random.choice(list(range(0 + self.radius, 300)) + list(range(700, 1000 - self.radius)))) # This ensures the loft is only placed near the edge of the map
        self.y = (random.choice(list(range(0 + self.radius, 300)) + list(range(700, 1000 - self.radius)))) # This ensures the loft is only placed near the edge of the map
        self.image = None # This is required to ensure images aren't garbage collected by python

    def drawLoft(self, canvas):
        # The window has to be included to prevent the image from being garbage collected at the end of the function
        self.image = ImageTk.PhotoImage(Image.open("data/images/Loft.png").convert("RGBA").resize((40,30)))

        canvas.create_image(self.x, self.y, image=self.image, anchor="center", tag=f"{self.name}")
        canvas.tag_raise(f"{self.name}")

    @staticmethod
    def getClass():
        return "Loft"

class Environment:
    # Starting the canvas and initialising the size, also responsible for adding the elements.
    def __init__(self):
        self.passive_objects = []
        self.active_objects = []
        self.villages = []
        self.towns = []
        self.cities = []

        self.geo_map = GeomagneticMap()

    def initialise_environment(self, window, xsize, ysize):
        # Initialise the canvas
        canvas = tk.Canvas(window, width=xsize, height=ysize, background='lightgreen')
        canvas.pack()

        # Reset the class variables
        self.passive_objects = []
        self.active_objects = []
        self.villages = []
        self.towns = []
        self.cities = []

        xsize = xsize - 1 # To stop values being placed beyond the real boundary of 999 by random.randint()
        ysize = ysize - 1 # To stop values being placed beyond the real boundary of 999 by random.randint()

        # Add the grid to show areas of geomagnetic change
        self.geo_map.drawGrid(canvas, xsize, ysize)

        # Add Loft (destination) - Required to be 1
        loft = Loft("Loft", xsize, ysize)
        loft.drawLoft(canvas)
        self.active_objects.append(loft)

        # Add City
        city = City("City1", xsize, ysize, self.passive_objects)
        city.drawCity(canvas)
        self.passive_objects.append(city)
        self.cities.append(city)

        # Add Towns
        for x in range(0, 15):
            town = Town(f"Town{x}", xsize, ysize, self.passive_objects)
            town.drawTown(canvas)
            self.passive_objects.append(town)
            self.towns.append(town)

        # Add Village
        for x in range(0, 30):
            village = Village(f"Village{x}", xsize, ysize, self.passive_objects)
            village.drawVillage(canvas)
            self.passive_objects.append(village)
            self.villages.append(village)

        # Add Roads
        # road_objects = []
        # for x in range(0, len(passive_objects)):
        #     road = Road(f"Road{x}", passive_objects[x], random.choice(passive_objects))
        #     if road.name != "NO CONNECTION":
        #         road.drawRoad(canvas)
        #         road_objects.append(road)

        # Add Predator
        for x in range(0, 1):
            predator = Predator(f"Predator{x}", xsize, ysize, self.passive_objects, self.active_objects)
            predator.drawPredator(canvas)
            self.active_objects.append(predator)

        # Used to add a scale value in the top left
        canvas.create_rectangle(20, 20, 53.3333, 22, fill="black")
        canvas.create_rectangle(20, 18, 20, 20, fill="black")
        canvas.create_rectangle(52.3333, 18, 52.3333, 20, fill="black")
        canvas.create_text(20, 36.6666, anchor="sw", text=f"1 Mile")

        # This returns the canvas, the passive_objects, the geomagnetic map and the active objects
        return canvas, self.passive_objects, self.active_objects, self.geo_map.Map

    # This is to initialise an environment, but with no canvas or representation. Just the objects. Useful for testing.
    def initialise_environment_no_draw(self, xsize, ysize):
        xsize = xsize - 1 # To stop values being placed beyond the real boundary of 999 by random.randint()
        ysize = ysize - 1 # To stop values being placed beyond the real boundary of 999 by random.randint()

        # Add Loft (destination) - Required to be 1
        loft = Loft("Loft", xsize, ysize)
        self.active_objects.append(loft)

        # Add City
        city = City("City1", xsize, ysize, self.passive_objects)
        self.passive_objects.append(city)
        self.cities.append(city)

        # Add Towns
        for x in range(0, 15):
            town = Town(f"Town{x}", xsize, ysize, self.passive_objects)
            self.passive_objects.append(town)
            self.towns.append(town)

        # Add Village
        for x in range(0, 30):
            village = Village(f"Village{x}", xsize, ysize, self.passive_objects)
            self.passive_objects.append(village)
            self.villages.append(village)

        # Add Roads
        # road_objects = []
        # for x in range(0, len(passive_objects)):
        #     road = Road(f"Road{x}", passive_objects[x], random.choice(passive_objects))
        #     if road.name != "NO CONNECTION":
        #         road.drawRoad(canvas)
        #         road_objects.append(road)

        # Add Predator
        for x in range(0, 1):
            predator = Predator(f"Predator{x}", xsize, ysize, self.passive_objects, self.active_objects)
            self.active_objects.append(predator)

        # This returns the canvas, the passive_objects, the geomagnetic map and the active objects
        return self.passive_objects, self.active_objects, self.geo_map.Map

    # Used to create a matrix storing values with the key specified below. This will be used in reinforcement learning and genetic algorithm classes.
    def create_matrix(self, xsize, ysize, draw=False):
        # Key
        # 1 = City
        # 2 = Town
        # 3 = Village
        # 1_ = Predator
        # 2_ = Loft
        search_space = np.zeros((xsize, ysize))
        for city in self.cities:
            circ = self.np_circle_func(city.midpoint[0], city.midpoint[1], city.radius, search_space)
            search_space[circ] = 1
        for town in self.towns:
            circ = self.np_circle_func(town.midpoint[0], town.midpoint[1], town.radius, search_space)
            search_space[circ] = 2
        for village in self.villages:
            circ = self.np_circle_func(village.midpoint[0], village.midpoint[1], village.radius, search_space)
            search_space[circ] = 3
        for obj in self.active_objects:
            if obj.getClass() == "Predator":
                circ = self.np_circle_func(obj.x, obj.y, obj.radius, search_space)
                for row in range(0, xsize):
                    for col in range(0, ysize):
                        if circ[row][col]:
                            search_space[row][col] = search_space[row][col] + 10
            elif obj.getClass() == "Loft":
                circ = self.np_circle_func(obj.x, obj.y, obj.radius, search_space)
                for row in range(0, xsize):
                    for col in range(0, ysize):
                        if circ[row][col]:
                            search_space[row][col] = search_space[row][col] + 20

        if draw:
            window = tk.Tk()
            canvas = tk.Canvas(window, width=1000, height=1000, bg="white")
            canvas.pack()
            for x in range(0,1000):
                for y in range(0,1000):
                    if search_space.T[x][y] == 1:
                        canvas.create_rectangle(x, y, x+1, y+1, fill="black")
                    elif search_space.T[x][y] == 2:
                        canvas.create_rectangle(x, y, x+1, y+1, fill="grey")
                    elif search_space.T[x][y] == 3:
                        canvas.create_rectangle(x, y, x+1, y+1, fill="lightgrey")
                    elif search_space.T[x][y] == 10:
                        canvas.create_rectangle(x, y, x+1, y+1, fill="yellow")
                    elif search_space.T[x][y] == 11:
                        canvas.create_rectangle(x, y, x+1, y+1, fill="bisque4")
                    elif search_space.T[x][y] == 12:
                        canvas.create_rectangle(x, y, x+1, y+1, fill="bisque3")
                    elif search_space.T[x][y] == 13:
                        canvas.create_rectangle(x, y, x+1, y+1, fill="bisque1")

            window.mainloop()

        # Have to remember that in canvas, it is 0,0 in top left and 1000,1000 in bottom right. The indexing for a np array
        # is different, therefore the transposed version is the same indexing as that of in tkinter.
        return search_space.T

    # Helper function to create circular areas in the numpy array, to show cities, towns and villages.
    def np_circle_func(self, x, y, radius, array):
        Y, X = np.ogrid[:array.shape[0], :array.shape[1]]
        euc_dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

        circ = euc_dist_from_center <= radius
        return circ


