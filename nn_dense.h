//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#include "nn_interfaces.h"
#include "nn_optimizer.h"

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T>{
        Tensor<T,2> W_, b_, Xc_, dW_, db_;
    public:
        template<typename InitW, typename InitB>
        Dense(size_t in_f,size_t out_f,InitW init_w, InitB init_b): W_({in_f,out_f}), b_({1,out_f}), dW_({in_f,out_f}), db_({1,out_f}){ init_w(W_); init_b(b_);}
        Tensor<T,2> forward(const Tensor<T,2>& X) override { Xc_=X; return Tensor<T,2>::matmul(X,W_)+b_; }
        Tensor<T,2> backward(const Tensor<T,2>& dZ) override { dW_=Tensor<T,2>::matmul(Xc_.transpose(),dZ); db_=dZ.sum_rows(); return Tensor<T,2>::matmul(dZ,W_.transpose()); }
        void update_params(IOptimizer<T>& opt) override { opt.update(W_,dW_); opt.update(b_,db_);}
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
