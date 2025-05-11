# This was used to save the "Tree" image and edit it to remove the background.
image = Image.open("data/images/Loft.png").convert("RGBA")

new_image = []
for item in image.getdata():
    print(item)
    if item in [(255, 255, 255, 255), (254, 254, 254, 255), (253, 253, 253, 255), (252, 252, 252, 255),
                (251, 251, 251, 255), (250, 250, 250, 255)]:
        new_image.append((0, 0, 0, 0))
    else:
        new_image.append(item)

image.putdata(new_image)
image.save("data/images/Loft.png")


# This is the predator class with multiple spheres with different values, this could be used but for now I just want to use one

# The predator that will influence the chances of the pigeon reaching the loft, if it passes through it's area.
class Predator:
    # This radius is taken from https://enviroliteracy.org/what-is-the-range-of-the-peregrine-falcon/ and is the lowest value.
    radius = int(1000 * (6.25 / 30)) # This is the first radius, where the pigeon is inside the range, but has a low chance of being caught
    radius2 = int(1000 * (5/30))
    radius3 = int(1000 * (3.75/30))
    radius4 = int(1000 * (2.5 / 30))
    radius5 = int(1000 * (1.25 / 30))


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
        canvas.create_text(self.x, self.y + self.radius, anchor="s", text="5%")
        canvas.create_oval(self.x - self.radius2, self.y - self.radius2,
                           self.x + self.radius2, self.y + self.radius2, fill="tomato3", tag="predator_area")
        canvas.create_text(self.x, self.y + self.radius2, anchor="s", text="10%")
        canvas.create_oval(self.x - self.radius3, self.y - self.radius3,
                           self.x + self.radius3, self.y + self.radius3, fill="tomato4", tag="predator_area")
        canvas.create_text(self.x, self.y + self.radius3, anchor="s", text="20%")
        canvas.create_oval(self.x - self.radius4, self.y - self.radius4,
                           self.x + self.radius4, self.y + self.radius4, fill="tomato3", tag="predator_area")
        canvas.create_text(self.x, self.y + self.radius4, anchor="s", text="30%")
        canvas.create_oval(self.x - self.radius5, self.y - self.radius5,
                           self.x + self.radius5, self.y + self.radius5, fill="tomato4", tag="predator_area")
        canvas.create_text(self.x, self.y + self.radius5, anchor="s", text="60%")

        self.image = ImageTk.PhotoImage(Image.open("data/images/Tree.png").convert("RGBA").resize((40, 40)))

        canvas.create_image(self.x, self.y, image=self.image, anchor="center", tag="treeImg")
        canvas.tag_lower("treeImg")
        canvas.tag_lower("predator_area")

    # TODO ADD FUNCTION TO RETURN THE PERCENTAGE CHANCE A PIGEON WILL BE KILLED GIVEN IT'S CURRENT POSITION IN THE CIRCLE.

    @staticmethod
    def getClass():
        return "Predator"


# This is the old reset function from gym_evironment.py, which is used to reset to whole new environment rather than just the bird, as is currently done.
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed) # taken from the website, unsure currently if necessary for me

        # Run these to reset the environment and the pigeon locations
        self.canvas.destroy()
        self.canvas, self.passive_objects, self.active_objects, self.geomag_map = self.env_orig.initialise_environment(self.window, X_SIZE, Y_SIZE)
        self.pigeon = Pigeon("Pigeon1", X_SIZE, Y_SIZE, self.passive_objects, self.active_objects, self.geomag_map)

        # Define these as the new locations.
        self._agent_location = [self.pigeon.x, self.pigeon.y]
        loft = [f for f in self.active_objects if f.getClass() == "Loft"][0]
        self._target_location = [loft.x, loft.y]

        # Get the observations to return the newest observations after reset
        observations = self.get_observations()

        return observations