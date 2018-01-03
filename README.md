Feature Pyramid Network on caffe

This is the unoffical version  Feature Pyramid Network for Feature Pyramid Networks for Object Detection https://arxiv.org/abs/1612.03144


the mxnet unoffical version  Feature Pyramid Network: https://github.com/unsky/Feature-Pyramid-Networks 
# results
coming soon
# usage
download  `ResNet50.caffemodel` and your dataset
### make caffe
```bash
cd FP-caffe
mkdir build
cd build
cmake ..
make -j16 all
```
### make lib

```bash
cd lib
make 
```
### train
```
./experiments/scripts/FP_Net_end2end.sh 1 FPN pascal_voc
```
### test
```
./test.sh 1 FPN pascal_voc
```

if you have issue about the fpn, open an issue.
