//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#include <cmath>
#include <algorithm>
#include "nn_interfaces.h"

namespace utec::neural_network {

// MSE
    template<typename T>
    class MSELoss final : public ILoss<T,2>{
        Tensor<T,2> y_pred_, y_true_;
    public:
        MSELoss(const Tensor<T,2>& y_pred,const Tensor<T,2>& y_true): y_pred_(y_pred), y_true_(y_true){}
        T loss() const override {
            T acc{}; for(size_t k=0;k<y_pred_.size();++k){T d=y_pred_.begin()[k]-y_true_.begin()[k]; acc+=d*d;} return acc/static_cast<T>(y_pred_.size());
        }
        Tensor<T,2> loss_gradient() const override {
            Tensor<T,2> dL(y_pred_.shape()); T factor=static_cast<T>(2.0/y_pred_.size()); for(size_t k=0;k<dL.size();++k) dL.begin()[k]=factor*(y_pred_.begin()[k]-y_true_.begin()[k]); return dL;
        }
    };

    template<typename T>
    class BCELoss final : public ILoss<T,2>{
        Tensor<T,2> y_pred_, y_true_;
    public:
        BCELoss(const Tensor<T,2>& y_pred,const Tensor<T,2>& y_true): y_pred_(y_pred), y_true_(y_true){}
        T loss() const override {
            const T eps=1e-12; T acc{}; for(size_t k=0;k<y_pred_.size();++k){T p=std::clamp(y_pred_.begin()[k],eps,1-eps); T y=y_true_.begin()[k]; acc+=-(y*std::log(p)+(1-y)*std::log(1-p));} return acc/static_cast<T>(y_pred_.size());
        }
        Tensor<T,2> loss_gradient() const override {
            const T eps=1e-12; Tensor<T,2> dL(y_pred_.shape()); for(size_t k=0;k<dL.size();++k){T p=std::clamp(y_pred_.begin()[k],eps,1-eps); T y=y_true_.begin()[k]; dL.begin()[k]=(p-y)/(p*(1-p)*static_cast<T>(y_pred_.size()));} return dL;
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
