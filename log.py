import matplotlib.pyplot as plt
import numpy as np



def load_scores( filename, file_dir="/Users/daddy/Desktop/projekt/log/"):
    file_object  = open(file_dir + filename , "r")
    #file_object.write("\n")
    #file_object.write(filename)
    #file_object.write("\n")

    d = file_object.read()
    data = d.split()
    return data

def plot_MA(scores, ma=10, name=""):
    #plotting

    x, y = [], []
    maxes  = []
    temp = []
    moving_avg =[]
    m_x = []


    for i in range(len(scores)):
        temp.append(scores[i])
        m_x.append(i+1)
        if i % ma == 0:
        #    y.append(np.mean(temp))
            maxes.append(max(temp))
            temp = []
            x.append(i+1)
        if i < ma:
            moving_avg.append(scores[i])
        else:
            moving_avg.append(np.mean(scores[i-ma:i]))



lr_short = [0.0001, 0.001, 0.01]
lr_long = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032,0.0064, 0.009]
discount_rates = [0.98, 0.90]
trainsteps = 1500
agents = []

low = []
high = []

for lr in lr_long:
    for dr in discount_rates:
        
        print("------------------------------ LEARNING RATE : ", lr, " ---- DISCOUNT RATE : ", dr, "----------------------------")
        
        name = str(len(agents))+ ".-" + str(lr) + "-" + str(lr) + "_" + str(dr)
        s1 = load_scores( name)
        #plot_MA(s1, name=name)
        
        agents.append(1)
        agents.append(1)

        name = str(len(agents))+ ".-" + str(lr/2) + "-" + str(lr) + "_" + str(dr) + "(diff_lr)"
        s2 = load_scores( name)
        #plot_MA(s1, name=name)

        print(np.mean(s1)," 2: ", np.mean(s2))