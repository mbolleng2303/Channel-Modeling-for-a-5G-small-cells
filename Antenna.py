from matplotlib import pyplot as plt


class Antenna:

    def __init__(self, pos_x, pos_y, type, plot=True):
        self.type = type
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.P = [pos_x, pos_y]
        if plot:
            self.plot()

    def plot(self):
        if self.type == 'tx':
            plt.scatter(self.pos_x, self.pos_y, color='blue', marker='D')
        else:
            plt.scatter(self.pos_x, self.pos_y, color='violet', marker='D')






