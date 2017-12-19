#include <vector>
#include "caffe/filler.hpp"
#include <iostream>
#include "caffe/layers/deformable_conv_layer.hpp"
using namespace std;
namespace caffe {
template <typename Dtype>
 int DeformableConvolutionLayer<Dtype>::input_shape(int i) {
   return (*(this->bottom_shape_))[this->channel_axis_ + i];
  }


template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = this->channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + this->num_spatial_axes_)
      << "bottom num_axes may not change.";
  this->num_ = bottom[0]->count(0, this->channel_axis_);
  CHECK_EQ(bottom[0]->shape(this->channel_axis_), this->channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  // Shape the tops.
  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);

  }
  
  if (reverse_dimensions()) {
    this->conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    this->conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  this->col_offset_ = this->kernel_dim_ * this->conv_out_spatial_dim_;
  this->output_offset_ = this->conv_out_channels_ * this->conv_out_spatial_dim_ / this->group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  this->conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(this->channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(this->channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  this->col_buffer_shape_.clear();
  this->col_buffer_shape_.push_back(this->kernel_dim_ * this->group_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      this->col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      this->col_buffer_shape_.push_back(this->output_shape_[i]);
    }
  }
  this->col_buffer_.Reshape(this->col_buffer_shape_);
  
  this->input_offset_dim_ = bottom[1]->count(this->channel_axis_);


  this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
  this->top_dim_ = top[0]->count(this->channel_axis_);
  this->num_kernels_im2col_ = this->conv_in_channels_ * this->conv_out_spatial_dim_;
  this->num_kernels_col2im_ = reverse_dimensions() ? this->top_dim_ : this->bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, this->out_spatial_dim_);
    this->bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(this->bias_multiplier_.count(), Dtype(1),
        this->bias_multiplier_.mutable_cpu_data());
  }
}
template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}
template <typename Dtype>
void DeformableConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  // Configure the kernel size, padding, stride, and inputs.
  DeformableConvolutionParameter conv_param = this->layer_param_.deformable_convolution_param();
  this->force_nd_im2col_ = conv_param.force_nd_im2col();
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = this->channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  this->num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(this->num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = this->stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << this->num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);
  this->num_output_ = this->layer_param_.deformable_convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.deformable_convolution_param().group();
  this->deformable_group_ = this->layer_param_.deformable_convolution_param().deformable_group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";


  CHECK_EQ(bottom[1]->shape(1), kernel_shape_data[0]*kernel_shape_data[1]*this->deformable_group_*2)
  << "Number channels of offset should be kernel*h*kernel_w*deformance*2";

  CHECK_EQ(bottom[1]->shape(2), bottom[0]->shape(2))
  << "Height and width of deformable conv layer and offset should be equal";
// cout<<bottom[1]->shape(2)<<" "<< bottom[1]->shape(3)<<endl;
// cout<<bottom[0]->shape(2)<<" "<< bottom[0]->shape(3)<<endl;
// cout<<bottom[1]->shape_string()<<endl;
// cout<<bottom[0]->shape_string()<<endl;
  CHECK_EQ(bottom[1]->shape(3), bottom[0]->shape(3))
  << "Height and width of deformable conv layer and offset should be equal";
  if (reverse_dimensions()) {
    this->conv_out_channels_ = this->channels_;
    this->conv_in_channels_ = this->num_output_;
  } else {
    this->conv_out_channels_ = this->num_output_;
    this->conv_in_channels_ = this->channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = this->conv_out_channels_;
  weight_shape[1] = this->conv_in_channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = this->layer_param_.deformable_convolution_param().bias_term();
  vector<int> bias_shape(this->bias_term_, this->num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.deformable_convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.deformable_convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->kernel_dim_ = this->blobs_[0]->count(1);
  this->weight_offset_ = this->conv_out_channels_ * this->kernel_dim_ / this->group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}


#ifdef CPU_ONLY
STUB_GPU(DeformableConvolutionLayer);
#endif
INSTANTIATE_CLASS(DeformableConvolutionLayer);




}  // namespace caffe
