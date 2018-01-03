#ifndef _CAFFE_UTIL_DEFORMABLE_IM2COL_HPP_
#define _CAFFE_UTIL_DEFORMABLE_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void deformable_im2col_gpu(const Dtype* data_im, const Dtype* data_offset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group,
    Dtype* data_col);
void ooop();

template <typename Dtype>
void deformable_col2im_gpu(const Dtype* data_col, const Dtype* data_offset, const int channels,
    const int height, const int width, const int num_kernels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, Dtype* grad_im);

template <typename Dtype>
void deformable_col2im_coord_gpu(const Dtype* data_col,const Dtype* data_im,  const Dtype* data_offset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, Dtype* grad_offset);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
