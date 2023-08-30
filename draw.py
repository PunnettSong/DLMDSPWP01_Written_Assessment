import matplotlib.pyplot as plt

class Draw():
    def __init__(self, x, y, plt):
        self.x = x
        self.y = y
        self.plt = plt
    
    def show(self):
       self.plt.show()

class Scatter(Draw):
    def __init__(self, x, y, plt):
        super().__init__(x, y, plt)
    def draw_scatter(self):
        self.plt.scatter(self.x, self.y)
        self.plt.grid(True)
class Line(Draw):
    def __init__(self, x, y, plt):
        super().__init__(x, y, plt)
    def draw_line(self):
        self.plt.plot(self.x, self.y)
        self.plt.legend(['y34', 'y31', 'y8', 'y46'])
        self.plt.grid(True)
    
