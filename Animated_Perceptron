import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from matplotlib.animation import FuncAnimation

class Perceptron():
    #initialize hyperparameters (learning rate and number of iterations)
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):      
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] #randomly initialize weights
        self.errors_ = []   #keeps tracks of the number of errors per iteration for observation purposes

        #iterate over labelled dataset updating weights for each features accordingly
        for _ in range(self.n_iter):
            errors = 0
            for xi, label in zip(X, y):
                update = self.eta * (label-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                
            self.errors_.append(errors)
        return self

    def step_fit(self, X, y):
        #iterate over labelled dataset updating weights for each features accordingly
        for xi, label in zip(X, y):
            update = self.eta * (label-self.predict(xi))
            self.w_[1:] += update * xi
            self.w_[0] += update
        return self

    #compute the net input i.e scalar sum of X and the weights plus the bias value
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    #predict a classification for a sample of features X
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def init_plot(self):
        self.line.set_data([],[])
        return self.line,

    def animate(self, i, X, y):
        self.step_fit(X, y)
        x, y = self.plot_data(X)
        self.line.set_data(x, y)
        return self.line,

    def plot_line(self, X):
        x = []
        y = []
        slope = -(self.w_[0]/self.w_[2])/(self.w_[0]/self.w_[1])  
        intercept = -self.w_[0]/self.w_[2]
        for i in np.linspace(np.amin(X[:,0])-0.5,np.amax(X[:,0])+0.5):
            #y=mx+c, m is slope and c is intercept
            x.append(i)
            y.append((slope*i) + intercept)
            
        return x, y
    
    def animated_fit(self, X, y):
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] #randomly initialize weights
        
        #here figure must be defined as a variable so it can be passed to FuncAnimation
        fig = plt.figure()
        
        #setting x and y limits with a 0.5 offset
        ax = plt.axes(xlim=(min(X[:,0])-0.5, max(X[:,0])+0.5), ylim=(min(X[:,1])-0.5, max(X[:,1])+0.5))
        
        #plotting our training points
        ax.plot(X[0:50, 0],X[0:50, 1], "bo", label="Iris-setosa")
        ax.plot(X[50:100, 0],X[50:100, 1], "rx", label="Versicolor")
        
        #labelling
        ax.legend(loc='upper left')

        
        #initialization of seperation line and our animation object
        self.line, = ax.plot([], [], lw=2) 
        anim = FuncAnimation(fig, self.animate, init_func=self.init_plot, fargs=(X, y,), frames=self.n_iter, interval=200, blit=True)
        anim.save('learning_process.gif', writer='imagemagick')
    
#import dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#preparing our data to be understood by our model
X = df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

ppn = Perceptron(eta=0.001, n_iter=100) #initializing a new perceptron
ppn.animated_fit(X, y) 
