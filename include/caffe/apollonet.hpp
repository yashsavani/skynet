#ifndef CAFFE_APOLLO_NET_HPP_
#define CAFFE_APOLLO_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"

#include <stdexcept>

namespace caffe {

template <typename Dtype>
class ApolloNet {
 public:
  explicit ApolloNet();
  virtual ~ApolloNet() {}

  void Init() {
    phase_ = TRAIN;
  }

  Dtype ForwardLayer(const string& layer_param_string, const string& runtime_param_string) {
    LayerParameter current_layer_param;
    RuntimeParameter runtime_param;
    ECHECK(runtime_param.ParseFromString(runtime_param_string), "");
    ECHECK(current_layer_param.ParseFromString(layer_param_string), "");
    ECHECK(current_layer_param.has_name(), "");
    const string& layer_name = current_layer_param.name();
    shared_ptr<Layer<Dtype> > layer;
    const bool new_layer = layers_map_.find(layer_name) == layers_map_.end();
    if (new_layer) {
      layer = LayerRegistry<Dtype>::CreateLayer(current_layer_param);;
      LOG(INFO) << "Creating Layer " << layer_name;
      LOG(INFO) << current_layer_param.DebugString();
      layers_map_[layer_name] = layer;
      current_layers_set_.insert(layer_name);
    } else {
      layer = layers_map_[layer_name];
      std::pair<set<string>::iterator,bool> ret = current_layers_set_.insert(layer_name);
      ECHECK(ret.second, "Layer with name '" << layer_name << "' is already used");
      ECHECK(layer->layer_param().type() == current_layer_param.type(), 
          "WARNING: layer with name '" << current_layer_param.name() << "' and different type already exists.");
    }
    layer->set_runtime_param(runtime_param);

    current_layers_vec_.push_back(layer_name);
    vector<Blob<Dtype>*> bottom_vec;
    vector<Blob<Dtype>*> top_vec;

    const vector<string>& bottom_names = bottom_blob_names_[layer_name];
    bool reset_bottoms = current_layer_param.bottom_size() != bottom_names.size();
    for (int bottom_id = 0; bottom_id < current_layer_param.bottom_size(); ++bottom_id) {
      const string& blob_name = current_layer_param.bottom(bottom_id);
      ECHECK(top_blobs_.find(blob_name) != top_blobs_.end(), 
          "Could not find bottom: '" << blob_name << "' for layer: " << layer_name);
      if (bottom_names.size() > bottom_id && bottom_names[bottom_id] != blob_name) { reset_bottoms = true; }
    }

    if (new_layer || reset_bottoms) {
      // layer is new, or it's list of bottoms has changed. Reset the bottom blobs
      bottom_blobs_[layer_name].clear();
      bottom_blob_names_[layer_name].clear();
      for (int bottom_id = 0; bottom_id < current_layer_param.bottom_size(); ++bottom_id) {
        const string& blob_name = current_layer_param.bottom(bottom_id);
        bottom_blob_names_[layer_name].push_back(blob_name);
        shared_ptr<Blob<Dtype> > top_blob = top_blobs_[blob_name];
        shared_ptr<Blob<Dtype> > bottom_blob(new Blob<Dtype>(top_blob->shape()));
        bottom_blob->ShareData(*top_blob);
        if (!layer->overwrites_delta()) {
          // if layer accumulates delta rather than overwriting, we can save memory
          bottom_blob->ShareDiff(*top_blob);
        }
        bottom_blobs_[layer_name].push_back(bottom_blob);
      }
      layer->reset_bottoms(bottom_blob_names_[layer_name]);
    } else {
      // Reshape bottom_blobs to match their respective top blobs 
      for (int bottom_id = 0; bottom_id < current_layer_param.bottom_size(); ++bottom_id) {
        const string& blob_name = current_layer_param.bottom(bottom_id);
        shared_ptr<Blob<Dtype> > top_blob = top_blobs_[blob_name];
        shared_ptr<Blob<Dtype> > bottom_blob = bottom_blobs_[layer_name][bottom_id];
        bottom_blob->ReshapeLike(*top_blob);
      }
    }

    for (int bottom_id = 0; bottom_id < current_layer_param.bottom_size(); ++bottom_id) {
      bottom_vec.push_back(bottom_blobs_[layer_name][bottom_id].get());
    }

    for (int top_id = 0; top_id < current_layer_param.top_size(); ++top_id) {
      const string& blob_name = current_layer_param.top(top_id);
      if (top_blobs_.find(blob_name) == top_blobs_.end()) {
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        top_blobs_[blob_name] = blob_pointer;
      }
      Blob<Dtype>* top_blob = top_blobs_[blob_name].get();
      top_vec.push_back(top_blob);
      if (top_blob->DiffInitialized() && !layer->is_loss()) {
        // Zero out top_diffs, except for loss blobs, which never change
        top_blob->SetDiffValues(0.);
      }
    }

    if (new_layer) {
      layer->SetUp(bottom_vec, top_vec);
      AddParamMasters(layer);
    }
    Dtype loss = 0;
    layer->set_phase(phase_);
    loss = layer->Forward(bottom_vec, top_vec);
    return loss;
  }

  void AddParamMasters(shared_ptr<Layer<Dtype> > layer) {
    //hook up param names and lr_mults with Net
    vector<string> param_names;
    vector<Dtype> param_decay_mults;
    vector<Dtype> param_lr_mults;
    const LayerParameter& layer_param = layer->layer_param();
    const int param_size = layer_param.param_size();
    const string& layer_name = layer_param.name();
    if (param_size > 0) {
      // new layer has explitily named it's params
      ECHECK(param_size == layer->blobs().size(), "Layer: '" << layer_name << "' declared an incorrect number of params");
      for (int i = 0; i < layer->blobs().size(); ++i) {
        string param_name;
        if (layer_param.param(i).has_name()) {
          param_name = layer_param.param(i).name();
          ECHECK(param_name.find(".p") == string::npos, "named param '" << param_name << "' cannot contain .p");
        } else {
          stringstream ss;
          ss << layer_param.name() << ".p" << i;
          param_name = ss.str();
        }
        param_names.push_back(param_name);
        param_decay_mults.push_back(layer_param.param(i).decay_mult());
        param_lr_mults.push_back(layer_param.param(i).lr_mult());
      }
    } else {
      for (int i = 0; i < layer->blobs().size(); ++i) {
        stringstream ss;
        ss << layer_param.name() << ".p" << i;
        param_names.push_back(ss.str());
        param_decay_mults.push_back(Dtype(1.));
        param_lr_mults.push_back(Dtype(1.));
      }
    }
    layer->set_param_names(param_names);
    for (int i = 0; i < layer->blobs().size(); ++i) {
      const string& param_name = layer->param_names()[i];
      if (params_.find(param_name) == params_.end()) {
        params_[param_name] = layer->blobs()[i];
        shared_ptr<Blob<Dtype> > master_ptr(new Blob<Dtype>(layer->blobs()[i]->shape()));
        param_masters_[param_name] = master_ptr; 
      } else {
        layer->blobs()[i]->ShareData(*params_[param_name]);
        layer->blobs()[i]->ShareDiff(*params_[param_name]);
      }
      param_decay_mults_[param_name] = param_decay_mults[i];
      param_lr_mults_[param_name] = param_lr_mults[i];
    }
  }

  void Backward() {
    for (int layer_id = current_layers_vec_.size() - 1; layer_id >= 0; --layer_id) {
      const string& layer_name = current_layers_vec_[layer_id];
      for (int i = 0; i < bottom_blobs_[layer_name].size(); ++i) {
        bottom_blobs_[layer_name][i]->SetDiffValues(Dtype(0.));
      }
    }

    for (int layer_id = current_layers_vec_.size() - 1; layer_id >= 0; --layer_id) {
      const string& layer_name = current_layers_vec_[layer_id];
      BackwardLayer(layer_name);
    }
  }

  void BackwardLayer(const string& layer_name) {
    shared_ptr<Layer<Dtype> > layer = layers_map_[layer_name];
    const LayerParameter& layer_param = layer->layer_param();
    vector<Blob<Dtype>*> bottom_vec;
    vector<Blob<Dtype>*> top_vec;
    for (int top_id = 0; top_id < layer_param.top_size(); ++top_id) {
      const string& blob_name = layer_param.top(top_id);
      top_vec.push_back(top_blobs_[blob_name].get());
    }
    vector<shared_ptr<Blob<Dtype> > > bottom_blobs = bottom_blobs_[layer_name];
    vector<bool> propagate_down;
    for (int bottom_id = 0; bottom_id < bottom_blobs.size(); ++bottom_id) {
      bottom_vec.push_back(bottom_blobs[bottom_id].get());
      propagate_down.push_back(true);
    }
    layer->Backward(top_vec, propagate_down, bottom_vec);

    if (layer->overwrites_delta()) {
      // if layer overwrites delta
      for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
        const string& bottom_name = layer_param.bottom(bottom_id);
        // add layer's bottom diff buffer to previous layer's top diffs
        top_blobs_[bottom_name]->AddDiffFrom(*bottom_vec[bottom_id]);
      }
    }
    for (int i = 0; i < layer->param_names().size(); ++i) {
      const string& param_name = layer->param_names()[i];
      // add param diff to master diff
      param_masters_[param_name]->AddDiffFrom(*layer->blobs()[i]);
      
    }
  }

  void UpdateLayer(const string& layer_name, set<string>& updated_params,
                   Dtype lr, Dtype momentum, Dtype decay_rate) {
    const shared_ptr<Layer<Dtype> > layer = layers_map_[layer_name];
    for (int i = 0; i < layer->param_names().size(); ++i) {
      const string& param_name = layer->param_names()[i];
      if (updated_params.find(param_name) == updated_params.end()) {
        // parameter is first instance of named param
        updated_params.insert(param_name);

        // Regularize, copy masters back to param diffs, update the params, and zero out diffs after.
        shared_ptr<Blob<Dtype> > param_layer = layer->blobs()[i];
        shared_ptr<Blob<Dtype> > param_master = param_masters_[param_name];
        param_master->L2Regularize(decay_rate * param_decay_mults_[param_name], *param_layer);
        param_layer->CopyDiffFrom(*param_master);
        param_layer->Update(lr * param_lr_mults_[param_name]);
        param_layer->scale_diff(Dtype(0.));
        param_master->scale_diff(momentum);
      }
    }

    current_layers_set_.erase(layer_name);
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update(Dtype lr, Dtype momentum, Dtype clip_gradients, Dtype decay_rate) {
    set<string> updated_params;
    if (clip_gradients > 0) {
      Dtype diff_l2_norm = DiffL2Norm();
      if (diff_l2_norm > clip_gradients) {
        Dtype scale_factor = clip_gradients / diff_l2_norm;
        lr *= scale_factor;
        //LOG(INFO) << "Scaling down gradients by factor: " << scale_factor;
      }
    }
    for (int layer_id = current_layers_vec_.size() - 1; layer_id >= 0; --layer_id) {
      const string& layer_name = current_layers_vec_[layer_id];
      UpdateLayer(layer_name, updated_params, lr, momentum, decay_rate);
      current_layers_vec_.pop_back();
    }
  }

  void ResetForward() {
    current_layers_vec_.clear();
    current_layers_set_.clear();
  }

  Dtype DiffL2Norm() {
    Dtype sumsq_diff = 0.;
    set<string> squared_params;
    for (int layer_id = current_layers_vec_.size() - 1; layer_id >= 0; --layer_id) {
      const string& layer_name = current_layers_vec_[layer_id];
      const shared_ptr<Layer<Dtype> > layer = layers_map_[layer_name];
      for (int i = 0; i < layer->param_names().size(); ++i) {
        const string& param_name = layer->param_names()[i];
        if (squared_params.find(param_name) == squared_params.end()) {
          // parameter is first instance of named param
          sumsq_diff += param_masters_[param_name]->sumsq_diff();
        }
      }
    }
    return std::sqrt(sumsq_diff);
  }

  void CopyTrainedLayersFrom(const NetParameter& param) {
    int num_source_layers = param.layer_size();
    for (int i = 0; i < num_source_layers; ++i) {
      const LayerParameter& source_layer = param.layer(i);
      const string& source_layer_name = source_layer.name();

      if (layers_map_.find(source_layer_name) == layers_map_.end()) {
        LOG(INFO) << "Ignoring source layer " << source_layer_name;
        continue;
      }

      LOG(INFO) << "Copying source layer " << source_layer_name;
      vector<shared_ptr<Blob<Dtype> > >& target_blobs =
          layers_map_[source_layer_name]->blobs();
        
      ECHECK(target_blobs.size() == source_layer.blobs_size(),
          "Incompatible number of blobs for layer " << source_layer_name);
      for (int j = 0; j < target_blobs.size(); ++j) {
        const bool kReshape = false;
        target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
      }
    }
  }

  void CopyTrainedLayersFrom(const string trained_filename) {
    NetParameter param;
    ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
    CopyTrainedLayersFrom(param);
  }

  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }

  inline void set_phase_test() {
    phase_ = TEST;
  }
  inline void set_phase_train() {
    phase_ = TRAIN;
  }
  inline map<string, shared_ptr<Blob<Dtype> > >& params() {
    return params_;
  }
  inline map<string, shared_ptr<Blob<Dtype> > >& blobs() {
    return top_blobs_;
  }
  inline map<string, shared_ptr<Layer<Dtype> > >& layers() {
    return layers_map_;
  }
  //inline map<string, vector<string> > layer_params() {
    //map<string, vector<string> > layer_params_map;
    //typename map<string, shared_ptr<Layer<Dtype> > >::iterator it;
    //for (it = layers_map_.begin(); it != layers_map_.end(); ++it) {
      //const string& layer_name = it->first;
      //layer_params_map[layer_name] = it->second->param_names();
    //}
    //return layer_params_map;
  //}
 protected:
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  map<string, shared_ptr<Layer<Dtype> > > layers_map_;
  /// @brief the blobs storing top results after each layer.
  map<string, shared_ptr<Blob<Dtype> > > top_blobs_;
  map<string, shared_ptr<Blob<Dtype> > > param_masters_;
  map<string, shared_ptr<Blob<Dtype> > > params_;
  map<string, Dtype> param_decay_mults_;
  map<string, Dtype> param_lr_mults_;
  map<string, vector<shared_ptr<Blob<Dtype> > > > bottom_blobs_;
  map<string, vector<string> > bottom_blob_names_;
  vector<string> current_layers_vec_;
  set<string> current_layers_set_;

  DISABLE_COPY_AND_ASSIGN(ApolloNet);
};


}  // namespace caffe

#endif  // CAFFE_APOLLO_NET_HPP_
