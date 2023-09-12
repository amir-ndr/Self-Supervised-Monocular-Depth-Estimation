# Self-supervised Monocular Depth Estimation

## Monocular Depth Estimation

Monocular depth estimation, also known as single-image depth estimation or depth prediction from a single image, is a computer vision task that involves estimating the depth or distance information of objects in a scene from a single 2D image or photograph. In other words, it aims to predict the 3D spatial information of objects in the image without the need for additional depth sensors or multiple views of the same scene.

Monocular depth estimation has various applications across multiple domains, including robotics, augmented reality, autonomous vehicles, and more.

## Model
The model is an unsupervised model consisting of two networks, one for estimating depth and the other for estimating pose. These two sub-models use a pretrained version of the ResNet18 network on the ImageNet dataset, which contains 1000 different classes. Actually in the depth estimation part we used stdcNet and darknet as backbone too. Additionally, the decoders for these two models are entirely convolutional. These two networks collaborate to make better predictions and update their weights.

Pose estimation refers to estimating the location of an object (more precisely, a pixel) in the next frame. This means that there is a sequence of frames that tracks the motion of an object, and the goal of pose estimation is to estimate the location of that object in the next frame. The depth estimation part predicts the depth map of a frame and obtains the depth for all pixels. This model is derived from the [DiPE](https://github.com/HalleyJiang/DiPE) model but achieved better results by specifically modifying loss functions, especially in critical situations such as occlusion and object motion in the opposite direction of camera movement (e.g., when a car moves in the opposite direction of the camera).
You can see the overall network architecture below:
! [Model architucture] ()

## Dataset
this model has been trained on the KITTI dataset in an unsupervised manner. This dataset is 175 gigabytes in size and consists of images with dimensions of (1242x375) in PNG format. The dataset has two well-known splits called "kitti_benchmark" and "kitti_eigen," which are used for training. These splits determine how the scenes of road recording are divided and how the images are input into the network. Our sequence frames consist of three images, and pose estimation is performed using these three images.
Please download and preprocess the KITTI dataset as [Monodepth2](https://github.com/nianticlabs/monodepth2) does. Also for training you can use commands from [DiPE Repo](https://github.com/HalleyJiang/DiPE)

## BackBones
For some tasks like autonomous driving and robotics navigation, we have to lighten the model. For this purpose we used backbones like `resnet18`, `stdcNet` and `darknet`. Also in [JETCO](https://en.jetco.co/), we had a small pist for self-driving car task and we also trained our model on pist. you can see the evaluation in the table below:
! [Evaluation] ()
Now see an example of predicting depth map of our pist image (backbone is `resnet18` and the model is trained on KITTI datadet):
! [Prediction] ()

## Acknowledgements

The project is built upon [DiPE](https://github.com/HalleyJiang/DiPE).
