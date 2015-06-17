#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neonet.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
NeoNet<Dtype>::NeoNet() {
  Init();
}

template <typename Dtype>
NeoNet<Dtype>::NeoNet(Phase phase) {
  NetParameter param;
  //ReadNetParamsFromTextFileOrDie(param_file, &param);
  param.mutable_state()->set_phase(phase);
  Init();
}

template <typename Dtype>
void NeoNet<Dtype>::Init() {
  LOG(INFO) << "Initting";
  //// Set phase from the state.
  //phase_ = in_param.state().phase();
  //// Filter layers based on their include/exclude rules and
  //// the current NetState.
  //NetParameter filtered_param;
  //FilterNet(in_param, &filtered_param);
  //LOG(INFO) << "Initializing net from parameters: " << std::endl
            //<< filtered_param.DebugString();
  //// Create a copy of filtered_param with splits added where necessary.
  //NetParameter param;
  //InsertSplits(filtered_param, &param);
  //// Basically, build all the layers and set up their connections.
  //name_ = param.name();
  //map<string, int> blob_name_to_idx;
  //set<string> available_blobs;
  //CHECK(param.input_dim_size() == 0 || param.input_shape_size() == 0)
      //<< "Must specify either input_shape OR deprecated input_dim, not both.";
  //if (param.input_dim_size() > 0) {
    //// Deprecated 4D dimensions.
    //CHECK_EQ(param.input_size() * 4, param.input_dim_size())
        //<< "Incorrect input blob dimension specifications.";
  //} else {
    //CHECK_EQ(param.input_size(), param.input_shape_size())
        //<< "Exactly one input_shape must be specified per input.";
  //}
  //memory_used_ = 0;
  //// set the input blobs
  //for (int input_id = 0; input_id < param.input_size(); ++input_id) {
    //const int layer_id = -1;  // inputs have fake layer ID -1
    //AppendTop(param, layer_id, input_id, &available_blobs, &blob_name_to_idx);
  //}
  //DLOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(Dtype);
  //// For each layer, set up its input and output
  //bottom_vecs_.resize(param.layer_size());
  //top_vecs_.resize(param.layer_size());
  //bottom_id_vecs_.resize(param.layer_size());
  //param_id_vecs_.resize(param.layer_size());
  //top_id_vecs_.resize(param.layer_size());
  //bottom_need_backward_.resize(param.layer_size());
  //for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    //// Inherit phase from net if unset.
    //if (!param.layer(layer_id).has_phase()) {
      //param.mutable_layer(layer_id)->set_phase(phase_);
    //}
    //// Setup layer.
    //const LayerParameter& layer_param = param.layer(layer_id);
    //if (layer_param.propagate_down_size() > 0) {
      //CHECK_EQ(layer_param.propagate_down_size(),
          //layer_param.bottom_size())
          //<< "propagate_down param must be specified "
          //<< "either 0 or bottom_size times ";
    //}
    //layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    //layer_names_.push_back(layer_param.name());
    //LOG(INFO) << "Creating Layer " << layer_param.name();
    //bool need_backward = false;

    //// Figure out this layer's input and output
    //for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         //++bottom_id) {
      //const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       //&available_blobs, &blob_name_to_idx);
      //// If a blob needs backward, this layer should provide it.
      //need_backward |= blob_need_backward_[blob_id];
    //}
    //int num_top = layer_param.top_size();
    //for (int top_id = 0; top_id < num_top; ++top_id) {
      //AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
    //}
    //// If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    //// specified fewer than the required number (as specified by
    //// ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    //Layer<Dtype>* layer = layers_[layer_id].get();
    //if (layer->AutoTopBlobs()) {
      //const int needed_num_top =
          //std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      //for (; num_top < needed_num_top; ++num_top) {
        //// Add "anonymous" top blobs -- do not modify available_blobs or
        //// blob_name_to_idx as we don't want these blobs to be usable as input
        //// to other layers.
        //AppendTop(param, layer_id, num_top, NULL, NULL);
      //}
    //}
    //// After this layer is connected, set it up.
    //LOG(INFO) << "Setting up " << layer_names_[layer_id];
    //layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    //for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      //if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        //blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      //}
      //blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      //LOG(INFO) << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      //if (layer->loss(top_id)) {
        //LOG(INFO) << "    with loss weight " << layer->loss(top_id);
      //}
      //memory_used_ += top_vecs_[layer_id][top_id]->count();
    //}
    //DLOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    //const int param_size = layer_param.param_size();
    //const int num_param_blobs = layers_[layer_id]->blobs().size();
    //CHECK_LE(param_size, num_param_blobs)
        //<< "Too many params specified for layer " << layer_param.name();
    //ParamSpec default_param_spec;
    //for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      //const ParamSpec* param_spec = (param_id < param_size) ?
          //&layer_param.param(param_id) : &default_param_spec;
      //const bool param_need_backward = param_spec->lr_mult() > 0;
      //need_backward |= param_need_backward;
      //layers_[layer_id]->set_param_propagate_down(param_id,
                                                  //param_need_backward);
    //}
    //for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      //AppendParam(param, layer_id, param_id);
    //}
    //// Finally, set the backward flag
    //layer_need_backward_.push_back(need_backward);
    //if (need_backward) {
      //for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        //blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      //}
    //}
  //}
  //// Go through the net backwards to determine which blobs contribute to the
  //// loss.  We can skip backward computation for blobs that don't contribute
  //// to the loss.
  //// Also checks if all bottom blobs don't need backward computation (possible
  //// because the skip_propagate_down param) and so we can skip bacward
  //// computation for the entire layer
  //set<string> blobs_under_loss;
  //set<string> blobs_skip_backp;
  //for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    //bool layer_contributes_loss = false;
    //bool layer_skip_propagate_down = true;
    //for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      //const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      //if (layers_[layer_id]->loss(top_id) ||
          //(blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        //layer_contributes_loss = true;
      //}
      //if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        //layer_skip_propagate_down = false;
      //}
      //if (layer_contributes_loss && !layer_skip_propagate_down)
        //break;
    //}
    //// If this layer can skip backward computation, also all his bottom blobs
    //// don't need backpropagation
    //if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      //layer_need_backward_[layer_id] = false;
      //for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               //++bottom_id) {
        //bottom_need_backward_[layer_id][bottom_id] = false;
      //}
    //}
    //if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    //if (layer_need_backward_[layer_id]) {
      //LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
    //} else {
      //LOG(INFO) << layer_names_[layer_id]
                //<< " does not need backward computation.";
    //}
    //for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         //++bottom_id) {
      //if (layer_contributes_loss) {
        //const string& blob_name =
            //blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        //blobs_under_loss.insert(blob_name);
      //} else {
        //bottom_need_backward_[layer_id][bottom_id] = false;
      //}
      //if (!bottom_need_backward_[layer_id][bottom_id]) {
        //const string& blob_name =
                   //blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        //blobs_skip_backp.insert(blob_name);
      //}
    //}
  //}
  //// Handle force_backward if needed.
  //if (param.force_backward()) {
    //for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      //layer_need_backward_[layer_id] = true;
      //for (int bottom_id = 0;
           //bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        //bottom_need_backward_[layer_id][bottom_id] =
            //bottom_need_backward_[layer_id][bottom_id] ||
            //layers_[layer_id]->AllowForceBackward(bottom_id);
        //blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            //blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            //bottom_need_backward_[layer_id][bottom_id];
      //}
      //for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           //++param_id) {
        //layers_[layer_id]->set_param_propagate_down(param_id, true);
      //}
    //}
  //}
  //// In the end, all remaining blobs are considered output blobs.
  //for (set<string>::iterator it = available_blobs.begin();
      //it != available_blobs.end(); ++it) {
    //LOG(INFO) << "This network produces output " << *it;
    //net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    //net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  //}
  //for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    //blob_names_index_[blob_names_[blob_id]] = blob_id;
  //}
  //for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    //layer_names_index_[layer_names_[layer_id]] = layer_id;
  //}
  //GetLearningRateAndWeightDecay();
  //debug_info_ = param.debug_info();
  //LOG(INFO) << "Network initialization done.";
  //LOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(Dtype);
}

// Helper for NeoNet::Init: add a new input or top blob to the net.  (Inputs have
// layer_id == -1, tops have layer_id >= 0.)
template <typename Dtype>
void NeoNet<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  //shared_ptr<LayerParameter> layer_param((layer_id >= 0) ?
    //(new LayerParameter(param.layer(layer_id))) : NULL);
  //const string& blob_name = layer_param ?
      //(layer_param->top_size() > top_id ?
          //layer_param->top(top_id) : "(automatic)") : param.input(top_id);
  //// Check if we are doing in-place computation
  //if (blob_name_to_idx && layer_param && layer_param->bottom_size() > top_id &&
      //blob_name == layer_param->bottom(top_id)) {
    //// In-place computation
    //LOG(INFO) << layer_param->name() << " -> " << blob_name << " (in-place)";
    //top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    //top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  //} else if (blob_name_to_idx &&
             //blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    //// If we are not doing in-place computation but have duplicated blobs,
    //// raise an error.
    //LOG(FATAL) << "Duplicate blobs produced by multiple sources.";
  //} else {
    //// Normal output.
    //if (layer_param) {
      //LOG(INFO) << layer_param->name() << " -> " << blob_name;
    //} else {
      //LOG(INFO) << "Input " << top_id << " -> " << blob_name;
    //}
    //shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    //const int blob_id = blobs_.size();
    //blobs_.push_back(blob_pointer);
    //blob_names_.push_back(blob_name);
    //blob_need_backward_.push_back(false);
    //if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    //if (layer_id == -1) {
      //// Set the (explicitly specified) dimensions of the input blob.
      //if (param.input_dim_size() > 0) {
        //blob_pointer->Reshape(param.input_dim(top_id * 4),
                              //param.input_dim(top_id * 4 + 1),
                              //param.input_dim(top_id * 4 + 2),
                              //param.input_dim(top_id * 4 + 3));
      //} else {
        //blob_pointer->Reshape(param.input_shape(top_id));
      //}
      //net_input_blob_indices_.push_back(blob_id);
      //net_input_blobs_.push_back(blob_pointer.get());
    //} else {
      //top_id_vecs_[layer_id].push_back(blob_id);
      //top_vecs_[layer_id].push_back(blob_pointer.get());
    //}
  //}
  //if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for NeoNet::Init: add a new bottom blob to the net.
template <typename Dtype>
int NeoNet<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  //const LayerParameter& layer_param = param.layer(layer_id);
  //const string& blob_name = layer_param.bottom(bottom_id);
  //if (available_blobs->find(blob_name) == available_blobs->end()) {
    //LOG(FATAL) << "Unknown blob input " << blob_name
               //<< " (at index " << bottom_id << ") to layer " << layer_id;
  //}
  //const int blob_id = (*blob_name_to_idx)[blob_name];
  //LOG(INFO) << layer_names_[layer_id] << " <- " << blob_name;
  //bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  //bottom_id_vecs_[layer_id].push_back(blob_id);
  //available_blobs->erase(blob_name);
  //bool propagate_down = true;
  //// Check if the backpropagation on bottom_id should be skipped
  //if (layer_param.propagate_down_size() > 0)
    //propagate_down = layer_param.propagate_down(bottom_id);
  //const bool need_backward = blob_need_backward_[blob_id] &&
                          //propagate_down;
  //bottom_need_backward_[layer_id].push_back(need_backward);
  //return blob_id;
}

template <typename Dtype>
void NeoNet<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  //const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  //const int param_size = layer_param.param_size();
  //string param_name =
      //(param_size > param_id) ? layer_param.param(param_id).name() : "";
  //if (param_name.size()) {
    //param_display_names_.push_back(param_name);
  //} else {
    //ostringstream param_display_name;
    //param_display_name << param_id;
    //param_display_names_.push_back(param_display_name.str());
  //}
  //const int net_param_id = params_.size();
  //params_.push_back(layers_[layer_id]->blobs()[param_id]);
  //param_id_vecs_[layer_id].push_back(net_param_id);
  //param_layer_indices_.push_back(make_pair(layer_id, param_id));
  //if (!param_size || !param_name.size() || (param_name.size() &&
      //param_names_index_.find(param_name) == param_names_index_.end())) {
    //// This layer "owns" this parameter blob -- it is either anonymous
    //// (i.e., not given a param_name) or explicitly given a name that we
    //// haven't already seen.
    //param_owners_.push_back(-1);
    //if (param_name.size()) {
      //param_names_index_[param_name] = net_param_id;
    //}
  //} else {
    //// Named param blob with name we've seen before: share params
    //const int owner_net_param_id = param_names_index_[param_name];
    //param_owners_.push_back(owner_net_param_id);
    //const pair<int, int>& owner_index =
        //param_layer_indices_[owner_net_param_id];
    //const int owner_layer_id = owner_index.first;
    //const int owner_param_id = owner_index.second;
    //LOG(INFO) << "Sharing parameters '" << param_name << "' owned by "
              //<< "layer '" << layer_names_[owner_layer_id] << "', param "
              //<< "index " << owner_param_id;
    //Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    //Blob<Dtype>* owner_blob =
        //layers_[owner_layer_id]->blobs()[owner_param_id].get();
    //const int param_size = layer_param.param_size();
    //if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  //ParamSpec_DimCheckMode_PERMISSIVE)) {
      //// Permissive dimension checking -- only check counts are the same.
      //CHECK_EQ(this_blob->count(), owner_blob->count())
          //<< "Shared parameter blobs must have the same count.";
    //} else {
      //// Strict dimension checking -- all dims must be the same.
      //CHECK(this_blob->shape() == owner_blob->shape());
    //}
    //layers_[layer_id]->blobs()[param_id]->ShareData(
        //*layers_[owner_layer_id]->blobs()[owner_param_id]);
  //}
}

//template <typename Dtype>
//Dtype NeoNet<Dtype>::ForwardFromTo(int start, int end) {
  //CHECK_GE(start, 0);
  //CHECK_LT(end, layers_.size());
  //Dtype loss = 0;
  //if (debug_info_) {
    //for (int i = 0; i < net_input_blobs_.size(); ++i) {
      //InputDebugInfo(i);
    //}
  //}
  //for (int i = start; i <= end; ++i) {
    //// LOG(ERROR) << "Forwarding " << layer_names_[i];
    //Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    //loss += layer_loss;
    //if (debug_info_) { ForwardDebugInfo(i); }
  //}
  //return loss;
//}

template <typename Dtype>
const vector<Blob<Dtype>*>& NeoNet<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  //// Copy bottom to internal bottom
  //for (int i = 0; i < bottom.size(); ++i) {
    //net_input_blobs_[i]->CopyFrom(*bottom[i]);
  //}
  //return ForwardPrefilled(loss);
}

//template <typename Dtype>
//void NeoNet<Dtype>::BackwardFromTo(int start, int end) {
  //CHECK_GE(end, 0);
  //CHECK_LT(start, layers_.size());
  //for (int i = start; i >= end; --i) {
    //if (layer_need_backward_[i]) {
      //layers_[i]->Backward(
          //top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      //if (debug_info_) { BackwardDebugInfo(i); }
    //}
  //}
//}

template <typename Dtype>
void NeoNet<Dtype>::InputDebugInfo(const int input_id) {
  //const Blob<Dtype>& blob = *net_input_blobs_[input_id];
  //const string& blob_name = blob_names_[net_input_blob_indices_[input_id]];
  //const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
  //LOG(INFO) << "    [Forward] "
     //<< "Input " << blob_name << " data: " << data_abs_val_mean;
}

template <typename Dtype>
void NeoNet<Dtype>::ForwardDebugInfo(const int layer_id) {
  //for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    //const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    //const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    //const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    //LOG(INFO) << "    [Forward] "
       //<< "Layer " << layer_names_[layer_id] << ", top blob " << blob_name
       //<< " data: " << data_abs_val_mean;
  //}
  //for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       //++param_id) {
    //const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    //const int net_param_id = param_id_vecs_[layer_id][param_id];
    //const string& blob_name = param_display_names_[net_param_id];
    //const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    //LOG(INFO) << "    [Forward] "
       //<< "Layer " << layer_names_[layer_id] << ", param blob " << blob_name
       //<< " data: " << data_abs_val_mean;
  //}
}

template <typename Dtype>
void NeoNet<Dtype>::BackwardDebugInfo(const int layer_id) {
  //const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  //for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    //if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    //const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    //const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    //const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    //LOG(INFO) << "    [Backward] "
        //<< "Layer " << layer_names_[layer_id] << ", bottom blob " << blob_name
        //<< " diff: " << diff_abs_val_mean;
  //}
  //for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       //++param_id) {
    //if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    //const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    //const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    //LOG(INFO) << "    [Backward] "
        //<< "Layer " << layer_names_[layer_id] << ", param blob " << param_id
        //<< " diff: " << diff_abs_val_mean;
  //}
}

template <typename Dtype>
void NeoNet<Dtype>::UpdateDebugInfo(const int param_id) {
  //const Blob<Dtype>& blob = *params_[param_id];
  //const int param_owner = param_owners_[param_id];
  //const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  //const string& param_display_name = param_display_names_[param_id];
  //const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  //if (param_owner < 0) {
    //const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    //LOG(INFO) << "    [Update] Layer " << layer_name
        //<< ", param " << param_display_name
        //<< " data: " << data_abs_val_mean << "; diff: " << diff_abs_val_mean;
  //} else {
    //const string& owner_layer_name =
        //layer_names_[param_layer_indices_[param_owner].first];
    //LOG(INFO) << "    [Update] Layer " << layer_name
        //<< ", param blob " << param_display_name
        //<< " (owned by layer " << owner_layer_name << ", "
        //<< "param " << param_display_names_[param_owners_[param_id]] << ")"
        //<< " diff: " << diff_abs_val_mean;
  //}
}

//template <typename Dtype>
//void NeoNet<Dtype>::BackwardFrom(int start) {
  //BackwardFromTo(start, 0);
//}

template <typename Dtype>
void NeoNet<Dtype>::Backward() {
  //BackwardFromTo(layers_.size() - 1, 0);
  //if (debug_info_) {
    //Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    //for (int i = 0; i < params_.size(); ++i) {
      //if (param_owners_[i] >= 0) { continue; }
      //asum_data += params_[i]->asum_data();
      //asum_diff += params_[i]->asum_diff();
      //sumsq_data += params_[i]->sumsq_data();
      //sumsq_diff += params_[i]->sumsq_diff();
    //}
    //const Dtype l2norm_data = std::sqrt(sumsq_data);
    //const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    //LOG(ERROR) << "    [Backward] All net params (data, diff): "
        //<< "L1 norm = (" << asum_data << ", " << asum_diff << "); "
        //<< "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  //}
}

//template <typename Dtype>
//void NeoNet<Dtype>::Reshape() {
  //for (int i = 0; i < layers_.size(); ++i) {
    //layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  //}
//}

template <typename Dtype>
void NeoNet<Dtype>::Update() {
  //// First, accumulate the diffs of any shared parameters into their owner's
  //// diff. (Assumes that the learning rate, weight decay, etc. have already been
  //// accounted for in the current diff.)
  //for (int i = 0; i < params_.size(); ++i) {
    //if (param_owners_[i] < 0) { continue; }
    //if (debug_info_) { UpdateDebugInfo(i); }
    //const int count = params_[i]->count();
    //const Dtype* this_diff;
    //Dtype* owner_diff;
    //switch (Caffe::mode()) {
    //case Caffe::CPU:
      //this_diff = params_[i]->cpu_diff();
      //owner_diff = params_[param_owners_[i]]->mutable_cpu_diff();
      //caffe_add(count, this_diff, owner_diff, owner_diff);
      //break;
    //case Caffe::GPU:
//#ifndef CPU_ONLY
      //this_diff = params_[i]->gpu_diff();
      //owner_diff = params_[param_owners_[i]]->mutable_gpu_diff();
      //caffe_gpu_add(count, this_diff, owner_diff, owner_diff);
//#else
      //NO_GPU;
//#endif
      //break;
    //default:
      //LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    //}
  //}
  //// Now, update the owned parameters.
  //for (int i = 0; i < params_.size(); ++i) {
    //if (param_owners_[i] >= 0) { continue; }
    //if (debug_info_) { UpdateDebugInfo(i); }
    //params_[i]->Update();
  //}
}

//template <typename Dtype>
//bool NeoNet<Dtype>::has_blob(const string& blob_name) const {
  //return blob_names_index_.find(blob_name) != blob_names_index_.end();
//}

//template <typename Dtype>
//const shared_ptr<Blob<Dtype> > NeoNet<Dtype>::blob_by_name(
    //const string& blob_name) const {
  //shared_ptr<Blob<Dtype> > blob_ptr;
  //if (has_blob(blob_name)) {
    //blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  //} else {
    //blob_ptr.reset((Blob<Dtype>*)(NULL));
    //LOG(WARNING) << "Unknown blob name " << blob_name;
  //}
  //return blob_ptr;
//}

//template <typename Dtype>
//bool NeoNet<Dtype>::has_layer(const string& layer_name) const {
  //return layer_names_index_.find(layer_name) != layer_names_index_.end();
//}

//template <typename Dtype>
//const shared_ptr<Layer<Dtype> > NeoNet<Dtype>::layer_by_name(
    //const string& layer_name) const {
  //shared_ptr<Layer<Dtype> > layer_ptr;
  //if (has_layer(layer_name)) {
    //layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  //} else {
    //layer_ptr.reset((Layer<Dtype>*)(NULL));
    //LOG(WARNING) << "Unknown layer name " << layer_name;
  //}
  //return layer_ptr;
//}

INSTANTIATE_CLASS(NeoNet);

}  // namespace caffe
