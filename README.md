# 3D U-Net Convolution Neural Network with Keras

Designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on 
volumetric segmentation with a 3D U-Net. Currently, the network does not have the B-Spline deformations
that are mentioned in the paper. If you figure out a way to apply these to a 3D Keras CNN, let me know! PRs are always
welcome!

The code was written to be trained using the 
[BRATS](https://sites.google.com/site/braintumorsegmentation/home/brats2015) data set for brain tumors, but it can
be easily modify to be used in other 3D applications. To adapt the network, you might have to play with the input size
to get something that works for your data.

I used [Bohdan Pavlyshenko](https://www.kaggle.com/bpavlyshenko)'s 
[Kaggle kernel](https://www.kaggle.com/bpavlyshenko/data-science-bowl-2017/nodules-segmentation) for 2D U-Net
segmentation as a base for this 3D U-Net.
