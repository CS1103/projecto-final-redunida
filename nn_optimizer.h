//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#include <unordered_map>
#include <cmath>
#include "nn_interfaces.h"

namespace utec::neural_network {

    template<typename T>
    class SGD final : public IOptimizer<T>{
        T lr_;
    public:
        explicit SGD(T lr=0.01): lr_(lr){}
        void update(Tensor<T,2>& P,const Tensor<T,2>& g) override { P += g * (-lr_);}
        void step() override {}
    };

    template<typename T>
    class Adam final : public IOptimizer<T>{
        T lr_, b1_, b2_, eps_; std::unordered_map<const void*, std::pair<Tensor<T,2>,Tensor<T,2>>> st_; size_t t_=0;
    public:
        Adam(T lr=0.001,T b1=0.9,T b2=0.999,T eps=1e-8): lr_(lr), b1_(b1), b2_(b2), eps_(eps){}
        void update(Tensor<T,2>& P,const Tensor<T,2>& g) override {
            ++t_; auto &pair=st_[&P];
            if(pair.first.size()==0){pair.first=Tensor<T,2>(P.shape()); pair.first.fill(0); pair.second=Tensor<T,2>(P.shape()); pair.second.fill(0);} auto &m=pair.first; auto &v=pair.second;
            for(size_t k=0;k<P.size();++k){m.begin()[k]=b1_*m.begin()[k]+(1-b1_)*g.begin()[k]; v.begin()[k]=b2_*v.begin()[k]+(1-b2_)*g.begin()[k]*g.begin()[k]; T mh=m.begin()[k]/(1-std::pow(b1_,t_)); T vh=v.begin()[k]/(1-std::pow(b2_,t_)); P.begin()[k]-=lr_*mh/(std::sqrt(vh)+eps_);} }
        void step() override {}
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
