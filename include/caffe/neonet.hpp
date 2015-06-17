#ifndef CAFFE_NEO_NET_HPP_
#define CAFFE_NEO_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class NeoNet {
 public:
  explicit NeoNet();
  explicit NeoNet(Phase phase);
  virtual ~NeoNet() {}

  void Init();
  //const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL);

  //Dtype ForwardFromTo(int start, int end);
  //Dtype ForwardFrom(int start);
  //Dtype ForwardTo(int end);
  void ForwardLayer(const string& data) {
    LayerParameter layer_param;
    CHECK(layer_param.ParseFromString(data));
    LOG(INFO) << layer_param.DebugString();
    CHECK(layer_param.has_name());
    const string& layer_name = layer_param.name();
    shared_ptr<Layer<Dtype> > layer;
    bool new_layer = layers_map_.find(layer_name) == layers_map_.end();
    if (new_layer) {
      layer = LayerRegistry<Dtype>::CreateLayer(layer_param);;
      LOG(INFO) << "Creating Layer " << layer_name;
      layers_map_[layer_name] = layer;
    } else {
      layer = layers_map_[layer_name];
      CHECK(layer->is_fresh()) << "Layer with name " << layer_name << " is already used";
    }
    layers_.push_back(layer);
    vector<Blob<Dtype>*> bottom_vec;
    vector<Blob<Dtype>*> top_vec;
    vector<bool> propagate_down;

    const vector<string>& bottom_names = bottom_blobs_names_[layer_name];
    bool reset_bottoms = false;
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
      const string& blob_name = layer_param.bottom(bottom_id);
      CHECK(top_blobs_.find(blob_name) != top_blobs_.end()) 
        << "Could not find bottom: '" << blob_name << "' for layer: " << layer_name;
      if (bottom_id >= bottom_names.size() || bottom_names[bottom_id] != blob_name) {
        reset_bottoms = true;
        break;
      }
    }
    if (reset_bottoms) {
      bottom_blobs_[layer_name].clear();
      for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
        const string& blob_name = layer_param.bottom(bottom_id);
        shared_ptr<Blob<Dtype> > top_blob = top_blobs_[blob_name];
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>(top_blob->shape()));
        blob_pointer->ShareData(*top_blob);
        if (!layer->overwrites_delta()) {
          // if layer accumulates delta rather than overwriting
          blob_pointer->ShareDiff(*top_blob);
        }
        bottom_blobs_[layer_name].push_back(blob_pointer);
      }
    }
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
      bottom_vec.push_back(bottom_blobs_[layer_name][bottom_id].get());
      propagate_down.push_back(true);
    }

    bool reset_tops = false;
    for (int top_id = 0; top_id < layer_param.top_size(); ++top_id) {
      const string& blob_name = layer_param.top(top_id);
      bool new_top = (top_blobs_.find(blob_name) == top_blobs_.end());
      if (new_top) {
        LOG(INFO) << layer_name << " -> " << blob_name;
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        top_blobs_[blob_name] = blob_pointer;
        reset_tops = true;
      }
      top_vec.push_back(top_blobs_[blob_name].get());
    }

    if (new_layer) {
      layer->SetUp(bottom_vec, top_vec);
    } else if (reset_bottoms || reset_tops) {
      layer->Reshape(bottom_vec, top_vec);
    }

      

      //AppendBottom(param, layer_id, bottom_id,
                   //&available_blobs, &blob_name_to_idx);
  }
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);
  //string Forward(const string& input_blob_protos, Dtype* loss = NULL);

  void Backward();
  //void BackwardFromTo(int start, int end);
  //void BackwardFrom(int start);
  //void BackwardTo(int end);

  //void Reshape();

  /// @brief Updates the network weights based on the diff values computed.
  void Update();

  //void ShareTrainedLayersWith(const NeoNet* other);

  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }

  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int>& param_owners() const { return param_owners_; }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  //bool has_blob(const string& blob_name) const;
  //const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  //bool has_layer(const string& layer_name) const;
  //const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

 protected:
  // Helpers for Init.
  /// @brief Append a new input or top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward about input Blobs.
  void InputDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  map<string, shared_ptr<Layer<Dtype> > > layers_map_;
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;
  /// @brief the blobs storing top results after each layer.
  map<string, shared_ptr<Blob<Dtype> > > top_blobs_;
  map<string, vector<shared_ptr<Blob<Dtype> > > > bottom_blobs_;
  map<string, vector<string> > bottom_blobs_names_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int> > param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;
  /// the learning rate multipliers
  vector<float> params_lr_;
  /// the weight decay multipliers
  vector<float> params_weight_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;

  DISABLE_COPY_AND_ASSIGN(NeoNet);
};


}  // namespace caffe

#endif  // CAFFE_NEO_NET_HPP_
