import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd 

def plot_flappy_values(V, agent_name):

    def get_Z(x, y):
        if (x,y) in V:
            return V[x,y]
        else:
            return 0

    def get_figure(ax):
        x_range = np.arange(0, 13)
        y_range = np.arange(-12, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
        ax.set_xlabel('Player\'s X Distance from Pipe')
        ax.set_ylabel('Player\'s Y Distance from Pipe')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(agent_name + ' State Value Plot')
    get_figure(ax)


def plot_policy_flappy(policy,agent_name):

    def get_Z(x, y):
        if (x,y) in policy:
            return policy[x,y]
        else:
            return -1 # unexplored
    
    def get_figure(ax):
        x_range = np.arange(0, 13)
        y_range = np.arange(-12, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape) 
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2'))
        plt.xticks(x_range) 
        plt.yticks(np.arange(23), range(-11, 12, 1)) 
        plt.gca().invert_xaxis() 
        ax.set_xlabel('X distance from Pipe') 
        ax.set_ylabel('Y distance from Pipe') 

        ax.grid(color='w', linestyle='-', linewidth=1) 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1) 
        cbar = plt.colorbar(surf, boundaries=[-1.5,-0.5,0.5,1.5], ticks=[-1,0,1], cax=cax) 
        cbar.ax.set_yticklabels(['Unexplored Region', 'Idle','Flap'])

    
    fig = plt.figure(figsize=(10, 10)) 
    ax = fig.add_subplot(111)
    ax.set_title(agent_name + ' Policy')
    get_figure(ax)
    plt.show() # show the plot


def plot_flappy_HP_tuning(score_history, param_names, agent_name):
    plt.figure(figsize=(20, 10))
    
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(score_history.keys())))))
    for params, scores in score_history.items():
        plt.plot(pd.Series(scores).rolling(100).mean(), label=f"{param_names}={params}")
            
    plt.title(f"Hyper Parameter Tuning {agent_name}")
    plt.xlabel('Number of Episodes')
    plt.ylabel(f"Moving Average Reward")
    plt.legend()
    plt.show()
