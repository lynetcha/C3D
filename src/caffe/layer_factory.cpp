// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_LAYER_FACTORY_HPP_
#define CAFFE_LAYER_FACTORY_HPP_

#include <string>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/convolution3d_layer.hpp"
#include "caffe/pool3d_layer.hpp"
#include "caffe/volume_data_layer.hpp"
#include "caffe/video_data_layer.hpp"
#include "caffe/deconvolution3d_layer.hpp"
#include "caffe/video_segmentation_data_layer.hpp"
#include "caffe/voxel_wise_softmax_layer.hpp"
#include "caffe/voxel_wise_softmax_loss_layer.hpp"
#include "caffe/video_with_voxel_truth_data_layer.hpp"
#include "caffe/voxel_custom_loss_layer.hpp"
#include "caffe/down_sampling_layer.hpp"
#include "caffe/input_layer.hpp"
#include "caffe/multistage_mean_field_layer3D.hpp"
//#include "caffe/eltwise_layer.hpp"
using std::string;

namespace caffe {


// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  const string& name = param.name();
  const LayerParameter_LayerType& type = param.type();
  switch (type) {
  case LayerParameter_LayerType_ACCURACY:
    return new AccuracyLayer<Dtype>(param);
  case LayerParameter_LayerType_BNLL:
    return new BNLLLayer<Dtype>(param);
  case LayerParameter_LayerType_CONCAT:
    return new ConcatLayer<Dtype>(param);
  case LayerParameter_LayerType_CONVOLUTION:
    return new ConvolutionLayer<Dtype>(param);
  case LayerParameter_LayerType_DATA:
    return new DataLayer<Dtype>(param);
  case LayerParameter_LayerType_DROPOUT:
    return new DropoutLayer<Dtype>(param);
  case LayerParameter_LayerType_EUCLIDEAN_LOSS:
    return new EuclideanLossLayer<Dtype>(param);
  case LayerParameter_LayerType_ELTWISE_PRODUCT:
    return new EltwiseProductLayer<Dtype>(param);
  case LayerParameter_LayerType_FLATTEN:
    return new FlattenLayer<Dtype>(param);
  case LayerParameter_LayerType_HDF5_DATA:
    return new HDF5DataLayer<Dtype>(param);
  case LayerParameter_LayerType_HDF5_OUTPUT:
    return new HDF5OutputLayer<Dtype>(param);
  case LayerParameter_LayerType_HINGE_LOSS:
    return new HingeLossLayer<Dtype>(param);
  case LayerParameter_LayerType_IMAGE_DATA:
    return new ImageDataLayer<Dtype>(param);
  case LayerParameter_LayerType_IM2COL:
    return new Im2colLayer<Dtype>(param);
  case LayerParameter_LayerType_INFOGAIN_LOSS:
    return new InfogainLossLayer<Dtype>(param);
  case LayerParameter_LayerType_INNER_PRODUCT:
    return new InnerProductLayer<Dtype>(param);
  case LayerParameter_LayerType_LRN:
    return new LRNLayer<Dtype>(param);
  case LayerParameter_LayerType_MEMORY_DATA:
    return new MemoryDataLayer<Dtype>(param);
  case LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
    return new MultinomialLogisticLossLayer<Dtype>(param);
  case LayerParameter_LayerType_POOLING:
    return new PoolingLayer<Dtype>(param);
  case LayerParameter_LayerType_POWER:
    return new PowerLayer<Dtype>(param);
  case LayerParameter_LayerType_RELU:
    return new ReLULayer<Dtype>(param);
  case LayerParameter_LayerType_SIGMOID:
    return new SigmoidLayer<Dtype>(param);
  case LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
    return new SigmoidCrossEntropyLossLayer<Dtype>(param);
  case LayerParameter_LayerType_SOFTMAX:
    return new SoftmaxLayer<Dtype>(param);
  case LayerParameter_LayerType_SOFTMAX_LOSS:
    return new SoftmaxWithLossLayer<Dtype>(param);
  case LayerParameter_LayerType_SPLIT:
    return new SplitLayer<Dtype>(param);
  case LayerParameter_LayerType_TANH:
    return new TanHLayer<Dtype>(param);
  case LayerParameter_LayerType_WINDOW_DATA:
    return new WindowDataLayer<Dtype>(param);
  case LayerParameter_LayerType_CONVOLUTION3D:
	return new Convolution3DLayer<Dtype>(param);
  case LayerParameter_LayerType_POOLING3D:
	return new Pooling3DLayer<Dtype>(param);
  case LayerParameter_LayerType_VOLUME_DATA:
	return new VolumeDataLayer<Dtype>(param);
  case LayerParameter_LayerType_VIDEO_DATA:
	return new VideoDataLayer<Dtype>(param);
  case LayerParameter_LayerType_DECONVOLUTION3D:
  	return new Deconvolution3DLayer<Dtype>(param);
  case LayerParameter_LayerType_VIDEO_SEGMENTATION_DATA:
	return new VideoSegmentationDataLayer<Dtype>(param);
  case LayerParameter_LayerType_VOXEL_SOFTMAX:
	return new VoxelWiseSoftmaxLayer<Dtype>(param);
  case LayerParameter_LayerType_VOXEL_SOFTMAX_LOSS:
	return new VoxelWiseSoftmaxWithLossLayer<Dtype>(param);
  case LayerParameter_LayerType_VIDEO_WITH_VOXEL_TRUTH_DATA:
	return new VideoWithVoxelTruthDataLayer<Dtype>(param);
  case LayerParameter_LayerType_VOXEL_CUSTOM_LOSS:
	return new VoxelCustomLossLayer<Dtype>(param);
  case LayerParameter_LayerType_DOWN_SAMPLING:
	return new DownSamplingLayer<Dtype>(param);
  case LayerParameter_LayerType_INPUT:
	return new InputLayer<Dtype>(param);
  case LayerParameter_LayerType_MULTISTAGE_MEANFIELD_3D:
	return new MultiStageMeanfield3DLayer<Dtype>(param);
  //case LayerParameter_LayerType_ELTWISE:
	//return new EltwiseLayer<Dtype>(param);
  case LayerParameter_LayerType_NONE:
    LOG(FATAL) << "Layer " << name << " has unspecified type.";
    break;
  default:
    LOG(FATAL) << "Layer " << name << " has unknown type " << type;
  }
  // just to suppress old compiler warnings.
  return (Layer<Dtype>*)(NULL);
}

template Layer<float>* GetLayer(const LayerParameter& param);
template Layer<double>* GetLayer(const LayerParameter& param);

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_HPP_
