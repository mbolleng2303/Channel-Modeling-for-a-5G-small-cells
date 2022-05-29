from matplotlib import pyplot as plt


class Wall:

    def __init__(self, x1, y1, x2, y2, street):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.P1 = [x1, y1]
        self.P2 = [x2, y2]
        self.street = street
        if self.x1 == self.x2:
            self.n1 = [1, 0]
            self.n2 = [-1, 0]
        if self.y1 == self.y2:
            self.n1 = [0, 1]
            self.n2 = [0, -1]
        self.plot()

    def plot(self):
        plt.plot([self.x1, self.x2], [self.y1, self.y2], color='black')
        '''if self.street == 's':
            plt.arrow(self.x1, self.y1, self.x2-self.x1, self.y2-self.y1, color='black', head_width =1)
        else :
            plt.arrow(self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1, color='red', head_width=1)'''







