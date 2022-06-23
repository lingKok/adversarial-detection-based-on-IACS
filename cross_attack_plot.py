# import numpy as np
#
# import matplotlib.pyplot as plt
# y=[0.6082,0.6023,0.6102,0.6014,0.5959,0.6165,0.7288,0.7776,0.8964,0.7153,0.9253,0.9324,0.9636,0.7245]#
# x=np.arange(0,14,1)
# plt.plot(x,y,color = 'r',
#          linestyle = '-.',
#          linewidth = 1,
#          marker = 'p',
#          markersize = 5,
#          markeredgecolor = 'b',
#          markerfacecolor = 'r',
#          label='normal examples')
# _x_ticks = ["layer{}".format(i+1) for i in x ]
#
# plt.xticks(x[::1], _x_ticks[::1], rotation=45)
# plt.xlabel('Layers for ILACS estimation')
# plt.ylabel('ROC score')
# plt.ylim((0,1))
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# x = np.random.rand(100).reshape(10,10)
attacks=['FGSM','PGD','DF','JSMA',r'$CW_0$',r'$CW_2$',r'$CW_\infty$']
y=np.asarray([])




ied = np.asarray([[0.9848, 0.9601, 0.9813, 0.9812, 0.9813, 0.9816, 0.986],
                [0.9766, 0.9831, 0.9762, 0.9731, 0.973, 0.9745, 0.9851],
                [0.9797, 0.8712, 0.9808, 0.9821, 0.9816, 0.9818, 0.9825],
                [0.9504, 0.7252, 0.9676, 0.9731, 0.9719, 0.9679, 0.9792],
                [0.9445, 0.7381, 0.967, 0.9741, 0.9717, 0.966, 0.9816],
                [0.966, 0.7803, 0.9759, 0.9785, 0.9773, 0.9748, 0.9829],
                [0.9105, 0.6853, 0.956, 0.9619, 0.9594, 0.9536, 0.9762]])
kd = np.asarray([[0.5884, 0.9032, 0.5733, 0.5804, 0.5846, 0.6314, 0.5204],
                [0.554, 0.9032, 0.5378, 0.5302, 0.5426, 0.556, 0.5204],
                [0.5884, 0.9032, 0.5733, 0.5804, 0.5876, 0.6314, 0.5204],
                [0.547, 0.6023, 0.534, 0.5884, 0.5394, 0.5559, 0.5178],
                [0.5472, 0.6023, 0.5326, 0.5269, 0.5876, 0.5559, 0.5177],
                [0.5469, 0.6021, 0.5351, 0.5268, 0.5393, 0.6314, 0.5178],
                [0.5884, 0.9032, 0.5733, 0.5807, 0.5876, 0.6314, 0.5204]])
lid = np.asarray([[0.8236, 0.8029, 0.7835, 0.8029, 0.7746, 0.789, 0.7752],
                  [0.7413, 0.8143, 0.7326, 0.7591, 0.7206, 0.7452, 0.7277],
                  [0.8071, 0.8059, 0.8042, 0.8133, 0.7798, 0.7974, 0.7852],
                  [0.8051, 0.7969, 0.7808, 0.8213, 0.778, 0.7916, 0.7847],
                  [0.7948, 0.7818, 0.7993, 0.7915, 0.7915, 0.7739, 0.7244],
                  [0.7245, 0.7246, 0.7164, 0.7403, 0.6937, 0.8031, 0.6855],
                  [0.7806, 0.7766, 0.7634, 0.7775, 0.7466, 0.7723, 0.7928]])
nss = np.asarray([[0.9998, 0.9991, 0.5047, 0.5045, 0.5039, 0.5182, 0.5183],
                  [0.9999, 0.9994, 0.508, 0.5078, 0.5071, 0.5026, 0.5265],
                  [0.0018, 0.5943, 0.5195, 0.5156, 0.5154, 0.5143, 0.5143],
                  [0.0017, 0.5882, 0.5182, 0.515, 0.5149, 0.5139, 0.5117],
                  [0.0017, 0.5851, 0.5184, 0.5149, 0.5148, 0.5139, 0.5117],
                  [0.0016, 0.5543, 0.5182, 0.515, 0.5148, 0.5139, 0.5118],
                  [0.0023, 0.6536, 0.5219, 0.5177, 0.5174, 0.516, 0.5179]])
fig=plt.figure(figsize=(26,6))
c1=plt.subplot(141)
plt.imshow(ied,cmap=plt.cm.summer,vmin=0,vmax=1)
plt.xticks([i for i in range(len(attacks))],attacks,fontsize=15)
plt.yticks([i for i in range(len(attacks))],attacks,fontsize=15)
plt.title('IED',fontsize=20)
plt.subplot(142)
c2=plt.imshow(kd,cmap=plt.cm.summer,vmin=0,vmax=1)
plt.xticks([i for i in range(len(attacks))],attacks,fontsize=15)
# plt.yticks([i for i in range(len(attacks))],attacks,fontsize=18)
plt.yticks([])
plt.title('KD',fontsize=20)
plt.subplot(143)
c3=plt.imshow(lid,cmap=plt.cm.summer,vmin=0,vmax=1)
plt.xticks([i for i in range(len(attacks))],attacks,fontsize=15)
# plt.yticks([i for i in range(len(attacks))],attacks,fontsize=18)
plt.yticks([])
plt.title('LID',fontsize=20)
plt.subplot(144)
c4=plt.imshow(nss,cmap=plt.cm.summer,vmin=0,vmax=1)
plt.xticks([i for i in range(len(attacks))],attacks,fontsize=15)
# plt.yticks([i for i in range(len(attacks))],attacks,fontsize=18)
plt.yticks([])
plt.title('NSS',fontsize=20)
fig.subplots_adjust(right=0.9)
#colorbar 左 下 宽 高
l = 0.92
b = 0.12
w = 0.015
h = 1 - 2*b

#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h]
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(c3, cax=cbar_ax)

#设置colorbar标签字体等
cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
font = {'family' : 'serif',
#       'color'  : 'darkred',
    'color'  : 'black',
    'weight' : 'normal',
    'size'   : 16,
    }
 #设置colorbar的标签字体及其大小
# plt.tight_layout()
plt.show()


