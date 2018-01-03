#include <vector>
#include "caffe/layers/conadd_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>
using  namespace std;
namespace caffe {

template <typename Dtype>
void ConaddLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{//conadd参数检查

}
template <typename Dtype>
void ConaddLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>&bottom,const vector<Blob<Dtype>*>&top)
{ 
//	cout<< bottom[0]->asum_data() << " "<< top[0]->asum_data() << endl;
	const int num_axes=bottom[0]->num_axes();//bottom的维度
	//检查合并的四维shape如果没指定，用bottom的维度进行初始化
	vector<int>top_shape = bottom[0]->shape();
	for(int i=0;i<bottom.size();i++)
	{   
		for(int j=0;j<num_axes;j++)
		{
			if(bottom[i]->shape(j)!=top_shape[j])
			{   
			    Blob<Dtype> B( bottom[i]->shape());
			    B.CopyFrom(*bottom[i]);
				const Dtype * data_b = bottom[i]->cpu_data();
				bottom[i]->Reshape(top_shape);
				Dtype * temp = bottom[i]->mutable_cpu_data();
				for( int ii=0; ii < B.count(); ii++)
				{
					temp[ii] = data_b[ii];
				}
			}
		}
	}
//初始化top 
	top[0]->Reshape(top_shape);
	if(bottom.size()==1)
	{
		top[0]->ShareData(*bottom[0]);
		top[0]->ShareDiff(*bottom[0]);
	}
}

template<typename Dtype>
void ConaddLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
	const int num_axes=bottom[0]->num_axes();//bottom的维度
	vector<int>top_shape=bottom[0]->shape();
	if(bottom.size()==1){return;}
	Dtype * top_data=top[0]->mutable_cpu_data();
	for(int i=0;i<bottom.size();i++)
	{   
		CHECK_EQ(num_axes,bottom[i]->num_axes())<<"the"<<i<<"-th input num_axes is different to 1-th input";
		for(int j=0;j<num_axes;j++)
		{ CHECK_EQ(num_axes,bottom[j]->num_axes())<<"the inputs are different in the number of the dimensionality ";
			//如果维度不同 填充0
			if(bottom[i]->shape(j)!=top_shape[j])
			{
				Blob<Dtype> temp;
				temp.CopyFrom(*bottom[i],0,0);
				Dtype *temp_data=temp.mutable_cpu_data();
				bottom[i]->Reshape(top_shape);
				Dtype * data_b=bottom[i]->mutable_cpu_data();
				for(int k=0;k<bottom[i]->count();k++)
				{
					data_b[k]=temp_data[k];
				}
				cout<<"debug:"<<bottom[i]->data_at(0,0,1,1)<<endl;//debug
			}
			CHECK_EQ(top_shape[j],bottom[i]->shape(j))<<"debug:padding Failure";
		}
		//实现conadd
		Dtype *bottom_data=bottom[i]->mutable_cpu_data();
		for(int m=0;m<top[0]->count();m++)
			top_data[m]=top_data[m]+bottom_data[m];
	}
}


template<typename Dtype>
void ConaddLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if(bottom.size()==1){return;}
	for(int i=0;i<bottom.size();i++)
	{
		bottom[i]->ShareDiff(*top[0]);
	}
}

#ifdef CPU_ONLY
STUB_GPU(ConaddLayer);
#endif

INSTANTIATE_CLASS(ConaddLayer);
REGISTER_LAYER_CLASS(Conadd);

}  // namespace caffe
