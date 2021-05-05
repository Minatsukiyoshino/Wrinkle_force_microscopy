import os, time, argparse, network, util
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import rc
import matplotlib as mpl
import os
import skimage.io as io
import matplotlib.ticker as ticker
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def calcForce(intensity):
    return 50.0*np.tan((intensity - 255.0/2.0)/81.2)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='wrinkle_to_force_1',  help='')
parser.add_argument('--input_size', type=int, default=256, help='input image size')
parser.add_argument('--test_subfolder', required=False, default='val',  help='')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--inverse_order', type=bool, default=True, help='True: [input, target], False: [target, input]')
opt = parser.parse_args()
print(opt)

# results save path
if not os.path.isdir(opt.dataset + '_results/test_results'):
    os.mkdir(opt.dataset + '_results/test_results')
if not os.path.isdir(opt.dataset + '_results/test_results/force_x'):
    os.mkdir(opt.dataset + '_results/test_results/force_x')
if not os.path.isdir(opt.dataset + '_results/test_results/force_y'):
    os.mkdir(opt.dataset + '_results/test_results/force_y')
if not os.path.isdir(opt.dataset + '_results/test_results/cell'):
    os.mkdir(opt.dataset + '_results/test_results/cell')

# data_loader
test_loader = util.data_loader('data/' + opt.dataset + '/' + opt.test_subfolder, 1, shuffle=False)
img_size = test_loader.shape[1]

# variables
x = tf.placeholder(tf.float32, shape=(None, opt.input_size, opt.input_size, test_loader.shape[3]))

# network
G = network.generator(x, opt.ngf)

# open session and initialize all variables
saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver.restore(sess, tf.train.latest_checkpoint(opt.dataset + '_results'))

print('test start!')

per_ptime = []
total_start_time = time.time()

for iter in range(test_loader.shape[0]):
    per_start_time = time.time()
    train_img = test_loader.next_batch()
    x_ = train_img
    x_ = util.norm(x_)
    test_img = sess.run(G, {x: x_})
    num_str = test_loader.file_list[iter][:test_loader.file_list[iter].find('.')]
    path = opt.dataset + '_results/test_results/force_x/' + num_str + '.png'
    cv2.imwrite(path, cv2.cvtColor(util.denorm(test_img[0]).astype(np.float32),cv2.COLOR_BGR2GRAY) )
    per_end_time = time.time()
    per_ptime.append(per_end_time - per_start_time)

for iter_2 in range(test_loader.shape[0]):
    per_start_time = time.time()
    train_img_rotate = test_loader.next_batch_rotate()
    x_rotate = train_img_rotate
    x_rotate = util.norm(x_rotate)
    test_img_rotate = sess.run(G, {x: x_rotate})
    num_str = test_loader.file_list[iter_2][:test_loader.file_list[iter_2].find('.')]
    path = opt.dataset + '_results/test_results/force_y/' + num_str + '.png'
    cv2.imwrite(path, cv2.cvtColor(util.denorm(test_img_rotate[0]).astype(np.float32),cv2.COLOR_BGR2GRAY) )
    per_end_time = time.time()
    per_ptime.append(per_end_time - per_start_time)

total_end_time = time.time()
total_ptime = total_end_time - total_start_time

########################draw################################
def calcForce(intensity):
    return 50.0*np.tan((intensity - 255.0/2.0)/81.2)
um_pix = 0.1076
for iter in range(533):
    #img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
    #name = list[i]
    #img_x = io.imread(os.path.join(opt.dataset + '_results/test_results/force_x/', "%04d.png"%i ))
    img_x = cv2.imread('D:/student/GAN4/wrinkle_to_force_1_results/test_results/force_x/' + "%04d.png" % (iter))
    img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
    img_y = cv2.rotate(cv2.imread('D:/student/GAN4/wrinkle_to_force_1_results/test_results/force_y/' + "%04d.png" % (iter)), cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2GRAY)
    img_cell = cv2.imread('D:/student/GAN4/wrinkle_to_force_1_results/test_results/cell/' + "%04d.jpg" % (iter))
    #plt.imshow(img_cell)
    #plt.show()
    data_fx = np.array(cv2.resize(img_x, (26, 26)))
    img_cell = cv2.resize(img_cell, (900, 900))
    data_fy = np.array(cv2.resize(img_y, (26, 26)))
    #for j in range(26):
    x=[]
    y=[]
    fx=[]
    fy = []
    fa=[]
    for j in range(26):
            for i in range(26):
                x.append (32.0*i + 48.0)
                y.append (32.0*j + 48.0)
                gid = 26*j + i
                tfx= calcForce(data_fx[j][i])
                tfy= calcForce(data_fy[j][i])
                fx.append (tfx)
                fy.append (tfy)
                fa.append (np.sqrt(tfx**2.0 + tfy**2.0))
    x  = np.array(x)
    y  = np.array(y)
    fx = np.array(fx)
    fy = np.array(fy)
    #fig = plt.figure(frameon=False)
    fig = plt.figure(frameon=False)
    img = img_cell
    #####if show cell images###########
    #plt.imshow(img, cmap = "gray")
    plt.quiver(x, y, fx, fy, fa, \
            cmap=plt.cm.jet, width=0.005, scale_units='dots', norm=mpl.colors.Normalize(vmin = 0.0, vmax = 200.0))
    plt.quiver(x, y, fx, fy, width=0.005, scale_units='dots', edgecolor='w', facecolor='None', linewidth=0.2)
    faa = plt.quiver(x, y, fx, fy, fa, \
            cmap=plt.cm.jet, width=0.005, scale_units='dots', norm=mpl.colors.Normalize(vmin = 0.0, vmax = 200.0))
    #ax = fa.flatten()
    #fig.colorbar(ax)
    #plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin = 0.0, vmax = 150.0), cmap=plt.cm.jet),ax = plt.axes())
    cb = plt.colorbar(faa, fraction=0.05, pad=0.04)
    cb.ax.tick_params(labelsize=16)
    font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 20,
        }
    ax2 = cb.ax
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()
    ax2.set_title('Pa',fontdict=font)
    plt.axis("off")
    fig.savefig("force/%d.png" %iter, format="png", bbox_inches='tight', pad_inches=0, dpi=243.6)

print('total %d images generation complete!' % (iter+1))
print('Avg. one image process ptime: %.2f, total %d images process ptime: %.2f' % (np.mean(per_ptime), (iter+1), total_ptime))
