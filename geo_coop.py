import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import random

class Pop:
    """
    Creates a population of h by w agents with random beta dist. cooperation
    """

    def __init__(self, h, w, cb, dur, alphaC, betaC, pixel_size, save_dir):

        self.h = h # height of pop in pixels
        self.w = w # width of pop in pixels
        self.color = np.zeros((h, w, 3), dtype=np.uint8)
        self.cb = cb  # cost benefit ratio (rho)
        self.dur = dur  # number times interact before replicator acts
        self.coop = np.random.beta(alphaC, betaC, size=(h, w))
        self.initCoopLevel = np.mean(self.coop)
        self.payoff = np.zeros((h, w))  # most recent payoff for indiv
        self.alphaC = alphaC  # shape parameter for initial coop dist
        self.betaC = betaC  # other shape parameter for initial coop dist
        self.pixel_size = pixel_size #for HD video of simple populations
        self.save_dir = save_dir # folder path saved images

    def neighbors(self, x, y):
        if x == 0:
            if y == 0:
                return [(x + 1, y + 1), (x + 1, y), (x, y + 1)]
            elif y == self.w - 1:
                return [(x + 1, y), (x + 1, y - 1), (x, y - 1)]
            else:
                return [(x + 1, y + 1), (x + 1, y), (x + 1, y - 1), 
                    (x, y + 1), (x, y - 1)]
    
        elif y == 0:
            if x == self.h - 1:
                return [(x, y + 1), (x - 1, y + 1), (x - 1, y)]
            else:
                return [(x + 1, y + 1), (x + 1, y), (x, y + 1), 
                    (x - 1, y), (x - 1, y + 1)]

        elif x == self.h - 1:
            if y == self.w - 1:
                return [(x, y - 1), (x - 1, y), (x - 1, y - 1)]
            else:
                return [(x, y + 1), (x, y - 1), (x - 1, y + 1), 
                    (x - 1, y), (x - 1, y - 1)]
        elif y == self.w - 1:
            return [(x + 1, y), (x + 1, y - 1), (x, y - 1), 
                    (x - 1, y), (x - 1, y - 1)]
        else:
            return [(x + 1, y + 1), (x + 1, y), (x + 1, y - 1),
                (x, y + 1), (x, y - 1),
                (x - 1, y + 1), (x - 1, y), (x - 1, y - 1)]

    def donate(self, giver, receiver):
        """Update payoffs of giver and receiver."""
        self.payoff[giver[0], giver[1]] -= (self.cb * self.coop[giver[0], giver[1]])
        self.payoff[receiver[0], receiver[1]] += self.coop[giver[0], giver[1]]

    def indiv_gen(self, i):
        """
        One generation for a particular indiv. i
        """
        for j in range(self.dur):
            receiver = random.choice(self.neighbors(i[0], i[1]))
            self.donate(i, receiver)

    def whole_gen(self):
        """
        One generation at the population level
        """
        it = np.nditer(self.coop, flags=['multi_index'])
        for i in it:
            self.indiv_gen(it.multi_index)

    def save_rect(self):
        # datetime object for naming convention
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y-%m-%d %H;%M;%S;%f")
    
        plt.imsave(self.save_dir + '\\' + dt_string + '.png', self.color.repeat(self.pixel_size, axis=0).repeat(self.pixel_size, axis=1))

class Naturalist:
    """

    Naturalist sets Pop parameters and allows us to repeat
    experiments.
    """

    def __init__(self, gen, mr, msd):
        self.gen = gen  # number times replicator acts
        self.mr = mr  # mutation rate btwn 0 (never mutate) and 1 (always)
        self.msd = msd  # the standard deviation of the mutation rate, say .1
        self.rect = None
    
    def set_rect(self, Pop_instance):
        self.rect = Pop_instance

    def adopt(self, i, other, cNew):
        """
        i adopts other strategy, with possibility of mutation
        """
        if self.mr < random.random():
            cNew[i[0], i[1]] = self.rect.coop[other[0], other[1]]
        else:
            # we'll clip normal dist at edges (0 and 1) later using clip f
            # Take sample from normal dist. centered at current cooperation value.
            cNew[i[0], i[1]] = np.random.normal(loc=self.rect.coop[other[0], other[1]], scale=self.msd)



    def slope_update(self, i, other, cNew):
        """
        Probability that idiv. (i) will adopt other's strat depends on
        difference between their payoffs
        """
        # note: individuals may have negative payoffs
        if self.rect.payoff[i[0], i[1]] < self.rect.payoff[other[0], other[1]]:
            # find probability that i adopt other strategy
            ### CHANGE THIS IF UPDATES TOO SLOWLY
            prob = (
                (self.rect.payoff[other[0], other[1]] - self.rect.payoff[i[0], i[1]]) / 
                    (np.max(self.rect.payoff) - np.min(self.rect.payoff))
            )
            if random.random() < prob:
                self.adopt(i, other, cNew)
        
    def update_strat(self):
        """
        Updates the strategy of individuals in population p based off their
        payoff at the end of the most recent generation.
        """
        p = self.rect
        cNew = np.copy(p.coop)
        
        it = np.nditer(p.coop, flags=['multi_index'])
        for i in it:
            self.slope_update(it.multi_index, random.choice(p.neighbors(it.multi_index[0], it.multi_index[1])), cNew)
        p.coop = np.clip(cNew, 0, 1)
        p.payoff = np.zeros((p.h, p.w))

    def coop_color(self):
        """
        transforms cooperation value to black and white
        """
        it = np.nditer(self.rect.coop, flags=['multi_index'])
        for i in it:
            self.rect.color[it.multi_index[0], it.multi_index[1]] = np.full(3, i * 255)

    def run_pop(self):
        """
        controls the progress of tasks within all generations
        """

        # Note, if have 10 gen, will measure gen 0 -> gen 9
        # 10 total, but the first one is your initial generation
        for generation in range(self.gen):
            self.coop_color()
            self.rect.save_rect()

            self.rect.whole_gen()
            self.update_strat()


if __name__ == "__main__":

    # generate timestamp used in naming the output directory
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H;%M;%S;")

    # make output directory. Add your own filepath as a string below
    dir = "C:\\Users\\benja\\OneDrive\\Documents\\coding\\Generative\\Outputs\\Coop Output" + dt_string
    if not os.path.exists(dir):
        os.mkdir(dir)

    test = Naturalist(gen = 36000, mr = .1, msd=.05)
    test.set_rect(Pop(h = 192, w = 108, cb = .025, dur = 1, alphaC = 10, betaC = 1, pixel_size = 10, save_dir = dir))

    test.run_pop()


# The resolution of the output photos will be h * pixel_size by w * pixel_size
# example: h = 16, w = 9, pixel_size = 120. output image resolution: 1920 x 1080
# example: h = 1080, w = 1920, pixel_size = 1. Output image resolution: 1080 x 1920

# coop and not coop strategies reach equilibrium somewhere around cb values of 0.005. 

# Next steps: add to git, add automatic export of csv holding the initial parameters, and maybe even the avgs per generation
# add option for values 0 -> 1 to be converted to values on a sRGB gradient 
# add image imports as initilization
# fix the naming of files so that it lines up with Premiere Pro import conventions 
