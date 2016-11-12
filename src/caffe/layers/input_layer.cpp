#include <vector>

#include "caffe/input_layer.hpp"

namespace caffe {

template <typename Dtype>
void InputLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {
  const int num_top = (*top).size();
  CHECK(num_top == 1) << "Input Layers supports 1 top only";
  const InputParameter& param = this->layer_param_.input_param();
  for (int i = 0; i < num_top; ++i) {
    (*top)[i]->Reshape(param.num(),param.channels(),param.length(), param.height(), param.width());
  }
}

INSTANTIATE_CLASS(InputLayer);
//REGISTER_LAYER_CLASS(Input);

}  // namespace caffe
