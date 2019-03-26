import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os


##p = os.getcwd()
p = "C:\\Users\\20184732\\Documents\\Boun\\new_conf" #os.setcwd()

path = p + '\\data\\ERR_list_Lab8_Comparison_v3.csv'
path2 = p + '\\data\\ERR_list_Ortakoy_Comparison_v3.csv'

df = pd.read_csv(path)
df2 = pd.read_csv(path2)

NN = "SNN"
SELECTION = "Active"
df_ = df
QUARTILE = .5
df_nn = df_[df_["NN Type"]==NN]
round(df_nn[df_nn["Fingerprint Selection"] == SELECTION].quantile(QUARTILE)[0],2)



fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
#ax.set_ylim(1.8,2.8)
ax.set_ylim(2,3.8)
#sns.set(rc={"font.size":12,"axes.titlesize":20,"axes.labelsize":12,"style":"white"})
a = sns.boxplot(x='Fingerprint Selection', y='Error',  hue='NN Type', data=df)
a.axes.set_title("$\mathcal{A}_2$", fontsize=20)
a.axes.set_ylabel('')
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
sns.set(font_scale = 1.5, style="white")
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(6,5))
ax2 = fig.add_subplot(111)
#sns.set(rc={"font.size":12,"axes.titlesize":20,"axes.labelsize":12})
b = sns.boxplot(x='Fingerprint Selection', y='Error',  hue='NN Type', data=df2)
b.axes.set_title("$\mathcal{A}_1$", fontsize=20)
sns.set(font_scale = 1.5, style="white")
ax2.set_ylim(2,3.8)
ax2.legend_.remove()
plt.tight_layout()
plt.show()