import numpy as np

import matplotlib.pyplot as plt

def calculate(nature_sample,adv_sample):
    nature_sample = nature_sample.reshape(len(nature_sample),-1)
    adv_sample = adv_sample.reshape(len(adv_sample),-1)
    l2 = np.mean(np.linalg.norm((adv_sample-nature_sample),ord=2,axis=1),0)
    linf = np.mean(np.linalg.norm((adv_sample-nature_sample),ord=np.inf,axis=1),0)
    return l2,linf

#cifar10': (
    #     'airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    # )
label_list =['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
attack_list = ['FGSM', 'PGD', 'DEEPFOOL', 'JSMA','CWL0','CWL2','CWLINF']
dataset = 'MNIST'

attack_image = []
attack_label = []
attack_l2 = []
attack_linf = []
# select = 121
true_data_path = './CleanDatasets/{}/{}_inputs.npy'.format(dataset, dataset)
true_label_path = './CleanDatasets/{}/{}_labels.npy'.format(dataset, dataset)
true_data = np.load(true_data_path)
true_label = np.load(true_label_path)

true_label = np.argmax(true_label,1)
print(true_label.shape)
select = np.ones_like(true_label)
for attack in attack_list:
    adv_label_path = adv_label_path = './AdversarialExampleDatasets/{}/{}/{}_AdvLabels.npy'.format(attack, dataset, attack)
    adv_label = np.load(adv_label_path)
    # select = ((true_label!=adv_label) == select)
    tem = (true_label != adv_label)
    select = (tem & select)

print(np.argmax(select))
print(select)
select =647
# true_label = np.argmax(true_label[select])
true_label = true_label[select]
true_image = true_data[select].transpose(1,2,0)
for attack in attack_list:
    adv_data_path = './AdversarialExampleDatasets/{}/{}/{}_AdvExamples.npy'.format(attack, dataset, attack)
    adv_label_path = './AdversarialExampleDatasets/{}/{}/{}_AdvLabels.npy'.format(attack, dataset, attack)
    adv_data = np.load(adv_data_path)
    adv_label = np.load(adv_label_path)
    attack_image.append(adv_data[select].transpose(1,2,0))
    attack_label.append(adv_label[select])
    l2,linf = calculate(true_data,adv_data)
    attack_l2.append(l2)
    attack_linf.append(linf)

fig,ax = plt.subplots(1,8,figsize=(40,8))

# plt.subplot(1, 6, 1)
# plt.title('orignal')
# plt.axis('off')
# adv_data_path = './AdversarialExampleDatasets/{}/{}/{}_AdvExamples.npy'
# adv_label_path = './AdversarialExampleDatasets/{}/{}/{}_AdvLabels.npy'.format(attack, dataset, attack)
ax[0].imshow(true_image,plt.cm.gray)
# ax[0].imshow(true_image)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('orignal\nlabel:{}'.format(true_label),fontsize=15)
# ax[0].set_xlabel('orignal\nlabel: {}'.format(label_list[int(true_label)]),fontsize=15)
for i in range(len(attack_list)):
    ax[i+1].imshow(attack_image[i],plt.cm.gray)
    # ax[i + 1].imshow(attack_image[i])
    ax[i+1].set_title(attack_list[i],fontsize=15)
    # ax[i+1].axes.xaxis.set_ticks([])
    ax[i + 1].set_xticks([])
    ax[i + 1].set_yticks([])
    ax[i+1].set_xlabel('$l_2$-norm:{:.4},$l_\infty$-norm:{:.4}\n label:{}'.format(attack_l2[i],attack_linf[i],attack_label[i]),fontsize=15)
    # ax[i + 1].set_xlabel(
    #     '$l_2$-norm:{:.4},$l_\infty$-norm:{:.4}\n label: {}'.format(attack_l2[i], attack_linf[i], label_list[int(attack_label[i])]),
    #     fontsize=15)
    # plt.imshow(attack_image[i])
plt.show()
