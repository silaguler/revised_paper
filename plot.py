import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

p = os.getcwd()#"/home/sila/Data/"

path, path2 = p + '\\data\\Lab8-MSE-All.csv', p + '\\data\\Lab8-DNN-MSE-All.csv'

lab8_df = pd.read_csv(path, header=None)
lab8_df2 = pd.read_csv(path2, header=None)

activation = np.arange(20,520,20)
activation2 = np.arange(10,110,10)
marker_list=['o','^','*','+','s']

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111) 
lab = ['$\mathcal{S}_{1}$','$\mathcal{S}_{2}$','$\mathcal{S}_{3}$','$\mathcal{S}_{4}$','$\mathcal{S}_{5}$']
for i in range(len(lab8_df)):
    
    ax.plot(activation, lab8_df.iloc[i], marker=marker_list[i],
                    label=lab[i], alpha=1)#, color=color_list[idx])
    ax.set_xlabel("Number of Hidden Units", fontsize=20);
    ax.set_ylabel("Loss", fontsize=20);  # ax.set_ylim(0.0, 0.02);  # ax.set_yticks(np.arange(0,0.03,0.005))
ax.set_ylim(0.0145,0.01750)
#ax.legend(fontsize=18)
ax.set_title("SNN - EMD$^2$", fontsize=20)
plt.tick_params(labelsize = 18)
plt.show()

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111) 
for i in range(len(lab8_df)):
    
    ax.plot(activation2, lab8_df2.iloc[i], marker=marker_list[i],
                    label=lab[i], alpha=1)#, color=color_list[idx])
    ax.set_xlabel("Number of Hidden Units", fontsize=20);
    ax.set_ylabel("Loss", fontsize=20);  # ax.set_ylim(0.0, 0.02);  # ax.set_yticks(np.arange(0,0.03,0.005))
ax.set_ylim(0.0145,0.01750)
ax.legend(fontsize=18)
ax.set_title("DNN - EMD$^2$", fontsize=20)
ax.set_ylabel('')
plt.tick_params(labelsize = 18)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
plt.show()


###################################################################################
###################################ORTAKOY PLOTS###################################
###################################################################################


fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111) 
ortakoy_lab = ['$\mathcal{S}_{1}$','$\mathcal{S}_{2}$','$\mathcal{S}_{3}$','$\mathcal{S}_{4}$','$\mathcal{S}_{5}$']
for i in range(len(ortakoy_df)):
    
    ax.plot(activation, ortakoy_df.iloc[i], marker=marker_list[i],
                    label=ortakoy_lab[i], alpha=1)#, color=color_list[idx])
    ax.set_xlabel("Number of Hidden Units", fontsize=20);
    ax.set_ylabel("Loss", fontsize=20);  # ax.set_ylim(0.0, 0.02);  # ax.set_yticks(np.arange(0,0.03,0.005))
ax.set_ylim(0.0145,0.01750)
#ax.legend(fontsize=18)
ax.set_title("SNN - EMD$^2$", fontsize=20)
plt.tick_params(labelsize = 18)
plt.show()

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111) 
for i in range(len(lab8_df)):
    
    ax.plot(activation2, ortakoy_df2.iloc[i], marker=marker_list[i],
                    label=ortakoy_lab[i], alpha=1)#, color=color_list[idx])
    ax.set_xlabel("Number of Hidden Units", fontsize=20);
    ax.set_ylabel("Loss", fontsize=20);  # ax.set_ylim(0.0, 0.02);  # ax.set_yticks(np.arange(0,0.03,0.005))
ax.set_ylim(0.0145,0.01750)
ax.legend(fontsize=18)
ax.set_title("DNN - EMD$^2$", fontsize=20)
ax.set_ylabel('')
plt.tick_params(labelsize = 18)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
plt.show()


