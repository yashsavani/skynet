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
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class NeoNet {
 public:
  explicit NeoNet();
  explicit NeoNet(Phase phase);
  virtual ~NeoNet() {}

  void Init() {
    phase_ = TRAIN;
  }

  void ForwardLayer(const string& data, bool reshape_only) {
    LayerParameter layer_param;
    CHECK(layer_param.ParseFromString(data));
    CHECK(layer_param.has_name());
    const string& layer_name = layer_param.name();
    shared_ptr<Layer<Dtype> > layer;
    bool new_layer = layers_map_.find(layer_name) == layers_map_.end();
    if (new_layer) {
      layer = LayerRegistry<Dtype>::CreateLayer(layer_param);;
      LOG(INFO) << "Creating Layer " << layer_name;
      LOG(INFO) << layer_param.DebugString();
      layers_map_[layer_name] = layer;
    } else {
      layer = layers_map_[layer_name];
      std::pair<set<string>::iterator,bool> ret = current_layers_set_.insert(layer_name);
      CHECK(ret.second) << "Layer with name " << layer_name << " is already used";
      CHECK(layer->layer_param().type() == layer_param.type()) << "WARNING: layer with name '" << layer_param.name() << "' and different type already exists.";
    }
    current_layers_vec_.push_back(layer_name);
    vector<Blob<Dtype>*> bottom_vec;
    vector<Blob<Dtype>*> top_vec;

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
      Blob<Dtype>* top_blob = top_blobs_[blob_name].get();
      top_vec.push_back(top_blob);
      if (top_blob->DiffInitialized() && !layer->is_loss()) {
        // Zero out top_diffs, except for loss blobs, which never change
        caffe_set(top_blob->count(), Dtype(0.), top_blob->mutable_cpu_diff());
      }
    }

    if (new_layer) {
      layer->SetUp(bottom_vec, top_vec);
      vector<string> param_names;
      vector<Dtype> param_lr_mults;
      const int param_size = layer_param.param_size();
      if (param_size > 0) {
        // new layer has named params
        CHECK_EQ(param_size, layer->blobs().size()) << "Layer: '" << layer_name << "' declared an incorrect number of params";
        for (int i = 0; i < layer->blobs().size(); ++i) {
          string param_name;
          if (layer_param.param(i).has_name()) {
            param_name = layer_param.param(i).name();
            CHECK(param_name.find(".p") == string::npos) << "named param '" << param_name << "' cannot contain .p";
          } else {
            stringstream ss;
            ss << layer_param.name() << ".p" << i;
            param_name = ss.str();
          }
          param_names.push_back(param_name);
          param_lr_mults.push_back(layer_param.param(i).has_lr_mult() ? layer_param.param(i).lr_mult() : Dtype(1.));
        }
      } else {
        for (int i = 0; i < layer->blobs().size(); ++i) {
          stringstream ss;
          ss << layer_param.name() << ".p" << i;
          param_names.push_back(ss.str());
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
          param_lr_mults_[param_name] = param_lr_mults[i];
        } else {
          layer->blobs()[i]->ShareData(*params_[param_name]);
          layer->blobs()[i]->ShareDiff(*params_[param_name]);
        }
      }
    } else if (reset_bottoms || reset_tops) {
      layer->Reshape(bottom_vec, top_vec);
    }
    if (!reshape_only) {
      layer->set_phase(phase_);
      layer->Forward(bottom_vec, top_vec);
    }
  }

  void Backward() {
    for (int layer_id = current_layers_vec_.size() - 1; layer_id >= 0; --layer_id) {
      const string& layer_name = current_layers_vec_[layer_id];
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
          Dtype* master_ptr = top_blobs_[bottom_name]->mutable_cpu_diff();
          const Dtype* slave_ptr = bottom_vec[bottom_id]->cpu_diff();
          // TODO: add ifdef CAFFE_GPU
          caffe_add(bottom_vec[bottom_id]->count(), master_ptr, slave_ptr, master_ptr);
        } }
      for (int i = 0; i < layer->param_names().size(); ++i) {
        const string& param_name = layer->param_names()[i];
        // TODO: check for GPU
        caffe_add(layer->blobs()[i]->count(), layer->blobs()[i]->cpu_diff(),
            param_masters_[param_name]->cpu_diff(),
            param_masters_[param_name]->mutable_cpu_diff());
      }
    }
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update(Dtype lr, Dtype momentum, Dtype clip_gradients) {
    set<string> updated_params;
    for (int layer_id = current_layers_vec_.size() - 1; layer_id >= 0; --layer_id) {
      const string& layer_name = current_layers_vec_[layer_id];
      if (clip_gradients > 0) {
        Dtype sumsq_diff = GetSumSqDiff();
        if (sumsq_diff > clip_gradients) {
          lr *= clip_gradients / sumsq_diff;
        }
      }
      UpdateLayer(layer_name, updated_params, lr, momentum);
      current_layers_vec_.pop_back();
    }
  }

  Dtype GetSumSqDiff() {
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
    return sumsq_diff;
  }

  void UpdateLayer(const string& layer_name, set<string>& updated_params, Dtype lr, Dtype momentum) {
    const shared_ptr<Layer<Dtype> > layer = layers_map_[layer_name];
    for (int i = 0; i < layer->param_names().size(); ++i) {
      const string& param_name = layer->param_names()[i];
      if (updated_params.find(param_name) == updated_params.end()) {
        // parameter is first instance of named param
        updated_params.insert(param_name);
        // TODO: add GPU support
        caffe_copy(layer->blobs()[i]->count(), param_masters_[param_name]->cpu_diff(),
            layer->blobs()[i]->mutable_cpu_diff());
        layer->blobs()[i]->Update(lr * param_lr_mults_[param_name]);

        caffe_cpu_scale(layer->blobs()[i]->count(), momentum,
            param_masters_[param_name]->cpu_diff(),
            param_masters_[param_name]->mutable_cpu_diff());
        caffe_set(layer->blobs()[i]->count(), Dtype(0.),
            layer->blobs()[i]->mutable_cpu_diff());
      }
    }

    current_layers_set_.erase(layer_name);
  }
  //void ShareTrainedLayersWith(const NeoNet* other);

  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return current_layers_vec_; }
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
  inline map<string, vector<string> > layer_params() {
    map<string, vector<string> > layer_params_map;
    typename map<string, shared_ptr<Layer<Dtype> > >::iterator it;
    for (it = layers_map_.begin(); it != layers_map_.end(); ++it) {
      const string& layer_name = it->first;
      layer_params_map[layer_name] = it->second->param_names();
    }
    return layer_params_map;
  }
 protected:
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  map<string, shared_ptr<Layer<Dtype> > > layers_map_;
  /// @brief the blobs storing top results after each layer.
  map<string, shared_ptr<Blob<Dtype> > > top_blobs_;
  map<string, shared_ptr<Blob<Dtype> > > param_masters_;
  map<string, shared_ptr<Blob<Dtype> > > params_;
  map<string, Dtype> param_lr_mults_;
  map<string, vector<shared_ptr<Blob<Dtype> > > > bottom_blobs_;
  map<string, vector<string> > bottom_blobs_names_;
  vector<string> current_layers_vec_;
  set<string> current_layers_set_;

  DISABLE_COPY_AND_ASSIGN(NeoNet);
};


}  // namespace caffe

#endif  // CAFFE_NEO_NET_HPP_
