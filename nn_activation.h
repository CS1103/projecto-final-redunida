//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#include <cmath>
#include "nn_interfaces.h"

namespace utec::neural_network {

// ReLU
    template<typename T>
    class ReLU final : public ILayer<T>{
        Tensor<T,2> Z_cache_;
    public:
        Tensor<T,2> forward(const Tensor<T,2>& Z) override{
            Z_cache_=Z; Tensor<T,2> A(Z.shape());
            for(size_t k=0;k<Z.size();++k) A.begin()[k]=std::max<T>(0,Z.begin()[k]);
            return A;
        }
        Tensor<T,2> backward(const Tensor<T,2>& dA) override{
            Tensor<T,2> dZ(dA.shape());
            for(size_t k=0;k<dZ.size();++k) dZ.begin()[k]= (Z_cache_.begin()[k]>0)? dA.begin()[k] : 0;
            return dZ;
        }
        void update_params(IOptimizer<T>&) override {}
    };

// Sigmoid
    template<typename T>
    class Sigmoid final : public ILayer<T>{
        Tensor<T,2> A_cache_;
    public:
        Tensor<T,2> forward(const Tensor<T,2>& Z) override{
            A_cache_=Tensor<T,2>(Z.shape());
            for(size_t k=0;k<Z.size();++k) A_cache_.begin()[k]=1.0/(1.0+std::exp(-Z.begin()[k]));
            return A_cache_;
        }
        Tensor<T,2> backward(const Tensor<T,2>& dA) override{
            Tensor<T,2> dZ(dA.shape());
            for(size_t k=0;k<dZ.size();++k){ T s=A_cache_.begin()[k]; dZ.begin()[k]=dA.begin()[k]*s*(1-s);} return dZ;
        }
        void update_params(IOptimizer<T>&) override {}
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
