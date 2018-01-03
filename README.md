Feature Pyramid Network on caffe

This is the unoffical version  Feature Pyramid Network for Feature Pyramid Networks for Object Detection https://arxiv.org/abs/1612.03144

# results
`FPN(resnet50) result is implemented without OHEM and train with pascal voc 2007 + 2012 test on 2007`


|mAP@0.5|aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|
|:--:|:-------:| -----:| --:| --:|-----:|--:|--:|--:|----:|--:|


|diningtable|dog |horse|motorbike|person |pottedplant|sheep|sofa|train|tv|
|----------:|:--:|:---:| -------:| -----:| -------:|----:|---:|----:|--:|

# framework
![](framework.png)
`the red and yellow are shared params`
# about the anchor size setting
In the paper the anchor setting is `Ratios： [0.5,1,2],scales :[8,]`

With the setting and P2~P6, all anchor is `[32,64,128,512,1024]`,but this setting is suit for COCO dataset which has so many small targets.

but the voc dataset has so many `[128,256,512]`targets.

So, we desgin the anchor setting:`Ratios： [0.5,1,2],scales :[8,16]`, this is very import for voc dataset.

# usage
download  voc07,12 dataset `ResNet50.caffemodel` 
- OneDrive download: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
### compile  caffe & lib
```bash
cd caffe-fpn
mkdir build
cd build
cmake ..
make -j16 all
cd lib
make 
```
### train & test
```
./experiments/scripts/FP_Net_end2end.sh 1 FPN pascal_voc
./test.sh 1 FPN pascal_voc
```

### TODO List
 - [x] all tests passed
 - [x] evaluate  object detection  performance on voc
 
### feature pyramid networks for object detection

Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2016). Feature pyramid networks for object detection. arXiv preprint arXiv:1612.03144.
