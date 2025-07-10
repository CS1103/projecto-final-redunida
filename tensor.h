//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#ifndef UTEC_ALGEBRA_TENSOR_AVAILABLE
#define UTEC_ALGEBRA_TENSOR_AVAILABLE 0
#endif

#if !UTEC_ALGEBRA_TENSOR_AVAILABLE
#include <vector>
#include <array>
#include <cassert>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

namespace utec::algebra {
    template<typename T, size_t Rank=2>
    class Tensor {
        static_assert(Rank==2, "Solo Rank==2 en stub");
        size_t rows_{}, cols_{};
        std::vector<T> data_;
    public:
        Tensor() = default;
        Tensor(size_t r,size_t c): rows_(r), cols_(c), data_(rows_*cols_){ }

        Tensor(std::initializer_list<size_t> dims){
            assert(dims.size()==2);
            auto it=dims.begin(); rows_=*it++; cols_=*it; data_.resize(rows_*cols_);
        }
        Tensor(std::initializer_list<size_t> dims, std::initializer_list<T> init) = delete;

        explicit Tensor(const std::array<size_t,2>& sh): Tensor(sh[0],sh[1]){ }

        static Tensor from_shape(const std::array<size_t,2>& sh){ return Tensor(sh); }

        Tensor(const std::array<size_t,2>& sh, std::initializer_list<T> init): Tensor(sh[0],sh[1]){
            assert(init.size()==data_.size()); std::copy(init.begin(),init.end(),data_.begin());
        }

        Tensor& operator=(std::initializer_list<T> init){ assert(init.size()==data_.size()); std::copy(init.begin(),init.end(),data_.begin()); return *this; }

        T& operator()(size_t r,size_t c){ return data_[r*cols_+c]; }
        const T& operator()(size_t r,size_t c) const { return data_[r*cols_+c]; }
        size_t size() const { return data_.size(); }
        std::array<size_t,2> shape() const { return {rows_,cols_}; }
        void fill(T v){ std::fill(data_.begin(), data_.end(), v); }

        auto begin(){ return data_.begin(); }
        auto end(){ return data_.end(); }
        auto begin() const { return data_.begin(); }
        auto end()   const { return data_.end(); }

        Tensor operator+(const Tensor& B) const {
            assert(cols_==B.cols_ && (rows_==B.rows_ || B.rows_==1 || rows_==1));
            Tensor C({rows_,cols_});
            for(size_t i=0;i<rows_;++i)
                for(size_t j=0;j<cols_;++j){ size_t ka=i*cols_+j; size_t kb=(B.rows_==1? j : ka); C.data_[ka]=data_[ka]+B.data_[kb]; }
            return C;
        }
        Tensor operator-(const Tensor& B) const {
            assert(cols_==B.cols_ && (rows_==B.rows_ || B.rows_==1 || rows_==1));
            Tensor C({rows_,cols_});
            for(size_t i=0;i<rows_;++i)
                for(size_t j=0;j<cols_;++j){ size_t ka=i*cols_+j; size_t kb=(B.rows_==1? j : ka); C.data_[ka]=data_[ka]-B.data_[kb]; }
            return C;
        }
        Tensor operator*(T s) const { Tensor C({rows_,cols_}); for(size_t k=0;k<size();++k) C.data_[k]=data_[k]*s; return C; }
        friend Tensor operator*(T s,const Tensor& A){ return A*s; }
        Tensor operator/(T s) const { Tensor C({rows_,cols_}); for(size_t k=0;k<size();++k) C.data_[k]=data_[k]/s; return C; }

        Tensor& operator+=(const Tensor& B){ *this = *this + B; return *this; }

        Tensor transpose() const {
            Tensor R({cols_,rows_});
            for(size_t i=0;i<rows_;++i) for(size_t j=0;j<cols_;++j) R(j,i)=(*this)(i,j);
            return R;
        }
        Tensor sum_rows() const {
            Tensor R({1,cols_}); R.fill(0);
            for(size_t i=0;i<rows_;++i) for(size_t j=0;j<cols_;++j) R(0,j)+=(*this)(i,j);
            return R;
        }

        static Tensor matmul(const Tensor& A,const Tensor& B){
            assert(A.cols_==B.rows_);
            Tensor C({A.rows_,B.cols_}); C.fill(0);
            for(size_t i=0;i<A.rows_;++i)
                for(size_t k=0;k<A.cols_;++k){ T a=A(i,k); for(size_t j=0;j<B.cols_;++j) C(i,j)+=a*B(k,j); }
            return C;
        }

        friend std::ostream& operator<<(std::ostream& os,const Tensor& M){
            os << "{\n";
            for(size_t i=0;i<M.rows_;++i){
                for(size_t j=0;j<M.cols_;++j){ os << M(i,j); if(j+1<M.cols_) os << ' '; }
                os << (i+1<M.rows_? '\n' : '\n');
            }
            os << "}";
            return os;
        }
    };

    template <typename F, typename T, size_t Rank>
    auto apply(const Tensor<T,Rank>& A, F&& f)
    {
        using R = std::invoke_result_t<F,T>;
        Tensor<R,Rank> B(A.shape());
        for(size_t k = 0; k < A.size(); ++k)
            B.begin()[k] = std::forward<F>(f)(A.begin()[k]);
        return B;
    }

}
#endif

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
