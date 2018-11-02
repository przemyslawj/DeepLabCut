import os.path
import sys


package_dir = '/home/prez/code/DeepLabCutTrained/DeepLabCut'

sys.path.append(package_dir)
# add parent directory: (where nnet & config are!)
sys.path.append(os.path.join(package_dir, "pose-tensorflow"))
sys.path.append(os.path.join(package_dir, "Generating_a_Training_Set"))

from myconfig_analysis import Task, date, \
    trainingsFraction, resnet, snapshotindex, shuffle

# Deep-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

# Dependencies for video:
import skimage
import skimage.color

import numpy as np
import os



####################################################
# Loading data, and defining model folder
####################################################

basefolder = os.path.join(package_dir, 'pose-tensorflow', 'models')
modelfolder = os.path.join(basefolder, Task + str(date) + '-trainset' +
               str(int(trainingsFraction * 100)) + 'shuffle' + str(shuffle))

cfg = load_config(os.path.join(modelfolder, 'test',"pose_cfg.yaml"))

##################################################
# Load and setup CNN part detector
##################################################

# Check which snapshots are available and sort them by # iterations
Snapshots = np.array([
    fn.split('.')[0]
    for fn in os.listdir(os.path.join(modelfolder , 'train'))
    if "index" in fn
])
increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]


##################################################
# Compute predictions over images
##################################################

# Check if data already was generated:
cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])

# Name for scorer:
trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)


cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
sess, inputs, outputs = predict.setup_pose_prediction(cfg)


def get_position(image, prev_pos):
    x = 0
    y = 0
    cropped_image = image
    if prev_pos[0] > 0 and prev_pos[1] > 0:
        BOX_SIZE = 120
        x = max(0, prev_pos[0] - BOX_SIZE/2)
        y = max(0, prev_pos[1] - BOX_SIZE/2)
        cropped_image = image[y:(y+BOX_SIZE), x:(x+BOX_SIZE)]

    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(cropped_image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    pose[0][0] += x
    pose[0][1] += y
    return pose[0][0], pose[0][1], pose[0][2]


if __name__ == '__main__':
    movie_file = '/mnt/DATA/Prez/conditioning/2018-10-29/movie/2018-10-29_F_trial_1.avi'
    import cv2
    stream = cv2.VideoCapture()
    video_opened = stream.open(movie_file)
    has_frame, frame = stream.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pos = get_position(gray_frame, None)

    frame[int(pos[0]), int(pos[1])] = (0, 0, 255)
    cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
