#include <iostream>
#include <memory>
#include "tensor.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include "neural_network.h"

using namespace utec::neural_network;

int main() {
    using T = float;

    // Dataset XOR
    Tensor<T, 2> X({4, 2}, {0,0, 0,1, 1,0, 1,1});
    Tensor<T, 2> Y({4, 1}, {0, 1, 1, 0});

    // Inicializadores básicos
    auto init_w = [](Tensor<T,2>& W) {
        for (auto& val : W) val = static_cast<T>((rand() % 200 - 100) / 100.0); // valores entre -1 y 1
    };
    auto init_b = [](Tensor<T,2>& B) {
        B.fill(0);
    };

    // Crear red neuronal
    NeuralNetwork<T> model;
    model.add_layer(std::make_unique<Dense<T>>(2, 4, init_w, init_b));      // Capa oculta 1
    model.add_layer(std::make_unique<ReLU<T>>());                           // Activación ReLU
    model.add_layer(std::make_unique<Dense<T>>(4, 1, init_w, init_b));      // Capa salida
    model.add_layer(std::make_unique<Sigmoid<T>>());                        // Activación Sigmoid

    // Entrenamiento
    std::cout << "Entrenando la red neuronal sobre XOR...\n";
    model.train<BCELoss>(X, Y, 1000, 4, 0.1);  // 1000 épocas, batch completo, lr = 0.1

    // Predicción
    std::cout << "\nPredicciones después del entrenamiento:\n";
    auto Y_pred = model.predict(X);
    std::cout << "Entrada\t\tPredicción\n";
    for (size_t i = 0; i < X.shape()[0]; ++i) {
        std::cout << X(i,0) << " " << X(i,1) << "\t\t" << Y_pred(i,0) << "\n";
    }

    return 0;
}
