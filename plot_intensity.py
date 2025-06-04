import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace as st

def main():

    ### plot intensity 
    path = r'/Users/tianyizheng/Desktop/postdoc/myproject/shgondoublehole/canbeused/processed/2channelrotate/'
    csv_1 = np.genfromtxt(path+'140mw-col1-1A2A-40x-zoom2-1-0002.csv', delimiter=',', names=["x", "y1"])
    csv_2 = np.genfromtxt(path+'140mw-col1-1B1C-40x-zoom2-1-0002.csv', delimiter=',', names=["x", "y2"])
    csv_3 = np.genfromtxt(path+'140mw-col1-1B2B-40x-zoom2-2-0002.csv', delimiter=',', names=["x", "y3"])
    csv_4 = np.genfromtxt(path+'140mw-col1-1C-40x-zoom2-2-0002.csv', delimiter=',', names=["x", "y4"])
    csv_5 = np.genfromtxt(path+'140mw-col1-1C2C-40x-zoom2-2-0002.csv', delimiter=',', names=["x", "y5"])
    csv_6 = np.genfromtxt(path+'140mw-col1-1D2D-40x-zoom2-1-0002.csv', delimiter=',', names=["x", "y6"])

    plt.plot(csv_1['x'], csv_1['y1'], color='orange', linewidth=3, label='1A2A')
    plt.plot(csv_2['x'], csv_2['y2'], color='green', linewidth=3, label='1B1C')
    plt.plot(csv_3['x'], csv_3['y3'], color='blue', linewidth=3, label='1B2B')
    plt.plot(csv_4['x'], csv_4['y4'], color='black', linewidth=3, label='Un1C')
    plt.plot(csv_5['x'], csv_5['y5'], color='red', linewidth=3, label='1C2C')
    plt.plot(csv_6['x'], csv_6['y6'], color='yellow', linewidth=3, label='1D2D')
    plt.xticks(np.arange(0,7.01),weight="bold",fontsize=12) 
    plt.yticks(np.arange(20,160,10),weight="bold",fontsize=12)
    plt.xlabel('Position',fontsize=12)
    plt.ylabel('Gray_Value',fontsize=12)
    plt.title('Backward channel SHG Intensity Plot',fontsize=18)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
