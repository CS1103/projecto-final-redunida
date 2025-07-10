//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#include <vector>
#include <memory>
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers_;
    public:
        void add_layer(std::unique_ptr<ILayer<T>> l){layers_.emplace_back(std::move(l));}

        Tensor<T,2> predict(const Tensor<T,2>& X){Tensor<T,2> A=X; for(auto &l:layers_) A=l->forward(A); return A;}

        template<template<typename...> class Loss, typename Optim=SGD<T>>
        void train(const Tensor<T,2>& X,const Tensor<T,2>& Y,size_t epochs,size_t batch,T lr){
            Optim opt(lr); size_t n=X.shape()[0];
            for(size_t e=0;e<epochs;++e){
                for(size_t i=0;i<n;i+=batch){size_t end=std::min(n,i+batch);
                    Tensor<T,2> Xb({end-i,X.shape()[1]}),Yb({end-i,Y.shape()[1]});
                    for(size_t r=0;r<end-i;++r){ for(size_t c=0;c<X.shape()[1];++c) Xb(r,c)=X(i+r,c); for(size_t c=0;c<Y.shape()[1];++c) Yb(r,c)=Y(i+r,c);}
                    Tensor<T,2> A=Xb; for(auto &l:layers_) A=l->forward(A);
                    Loss<T> loss(A,Yb); Tensor<T,2> dA=loss.loss_gradient();
                    for(auto it=layers_.rbegin(); it!=layers_.rend(); ++it) dA=(*it)->backward(dA);
                    for(auto &l:layers_) l->update_params(opt);
                    opt.step();
                }
            }
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
