#include <vector>
#include<iostream>
#include "caffe/layers/conadd_layer.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe {
template <typename Dtype>
__global__ void Conadd() { }


template <typename Dtype>
void ConaddLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{ 
	vector<int> top_shape = bottom[0]->shape();
	const int num_axes = bottom[0]->num_axes(); 
	if (bottom.size() == 1) { return; }
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_scal(top[0]->count(),(Dtype)(0),top_data);
	caffe_abs(top[0]->count(),top_data,top_data);
    for(int i=0;i < bottom.size(); i++)
	{	
	//	top[0]->ShareData(*bottom[0]);      	
		CHECK_EQ(num_axes, bottom[i]->num_axes() )<< "the"<< i << "-th input num_axes is different to 1-th input";
		//实现conadd
	    const Dtype *bottom_data = bottom[i]->cpu_data();
	    caffe_add(top[0]->count(), bottom_data, top_data, top_data);
	}

}
template <typename Dtype>
void ConaddLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  {
	if(propagate_down[0])
	{
		if (bottom.size() == 1) { return; }
		for(int i=0;i<bottom.size();i++)
		{
			bottom[i]->ShareDiff(*top[0]);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ConaddLayer);

}  // namespace caffe
