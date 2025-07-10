# Proyecto Final 2025-1: AI Neural Network

## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación de una red neuronal multicapa en C++ desde cero, utilizando estructuras de datos personalizadas y principios de programación orientada a objetos. El modelo es capaz de realizar tareas de clasificación binaria usando funciones de activación como ReLU y Sigmoid, funciones de pérdida como MSE y BCE, y optimizadores como SGD y Adam.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)

---

### Datos generales

- **Tema**: Redes Neuronales en AI
- **Grupo**: `RedUnida`
- **Integrantes**:
  - Diego Abraham Ladron de Guevara Aguirre – 202410083 (Documentación)
  - Raúl Janampa Salvatierra - 202410411 (Responsable de investigación teórica)
  - Berrú Tenorio, Andy Raí (Desarrollo de la arquitectura)
  - Demarini Leyton, Leandro Nadir (Implementación del modelo)
  - Huaytalla Suárez, Andree Mauricio (Pruebas y benchmarking)

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:
   - CMake 3.18+
   - No requiere librerías externas (se implementa una clase `Tensor` propia)
3. **Instalación**:
   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

---

### 1. Investigación teórica

- **Objetivo**: Explorar fundamentos matemáticos y computacionales de redes neuronales artificiales.

#### Contenidos:

1. Historia y evolución de las redes neuronales: desde el perceptrón hasta el deep learning.
2. Arquitecturas clásicas: MLP (utilizada en este proyecto), CNN y RNN.
3. Entrenamiento por descenso de gradiente (SGD, Adam) y retropropagación.
4. Funciones de activación (ReLU y Sigmoid) y pérdida (MSE, BCE).


## 📚 Contenidos

### 1.1 Historia y Evolución de las Redes Neuronales

Las redes neuronales artificiales surgieron en los años 40 con el modelo de McCulloch y Pitts (1943). Posteriormente, el **perceptrón** propuesto por Frank Rosenblatt (1958) marcó el primer avance importante, aunque sus limitaciones llevaron al llamado "invierno de la IA" tras las críticas de Minsky y Papert (1969).

En los años 80, el desarrollo del algoritmo de **retropropagación** permitió el entrenamiento de **perceptrones multicapa (MLP)**, reactivando la investigación. Desde entonces, han surgido arquitecturas más profundas y especializadas como las **CNN** y **RNN**, impulsando el auge del **deep learning** en múltiples disciplinas.

> 📖 Referencia: [1], [2]

---

### 1.2 Arquitecturas Clásicas

- **MLP (Multilayer Perceptron)**: Capas completamente conectadas, ideales para clasificación básica y regresión.
- **CNN (Convolutional Neural Network)**: Utilizadas en tareas de visión por computadora. Detectan patrones locales mediante filtros.
- **RNN (Recurrent Neural Network)**: Adecuadas para datos secuenciales como texto o audio. Conservan memoria de corto plazo.

⚙️ Este proyecto implementa una red del tipo **MLP**, utilizada para tareas de clasificación supervisada.

> 📖 Referencia: [1], [4]

---

### 1.3 Entrenamiento: Descenso de Gradiente y Retropropagación

El entrenamiento de una red neuronal consiste en minimizar una función de pérdida usando algoritmos de optimización:

- **SGD (Stochastic Gradient Descent)**: Realiza actualizaciones con muestras individuales o mini-lotes.
- **Adam (Adaptive Moment Estimation)**: Variante más robusta que ajusta dinámicamente los pasos de actualización.

La **retropropagación** permite calcular los gradientes de manera eficiente utilizando la regla de la cadena.

> 📖 Referencia: [1], [3], [4]

---

### 1.4 Funciones de Activación y Pérdida

#### 🔁 Funciones de Activación:
- **ReLU**: `f(x) = max(0, x)` → Rápida, evita saturación para valores positivos.
- **Sigmoid**: `f(x) = 1 / (1 + e^{-x})` → Útil para salidas binarias.

#### 📉 Funciones de Pérdida:
- **MSE (Mean Squared Error)**: Común en regresión.
- **BCE (Binary Cross-Entropy)**: Común en clasificación binaria.

La elección depende del tipo de salida y de la tarea que se desea resolver.

> 📖 Referencia: [2], [4]

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

- **Patrones de diseño utilizados**:

  - Strategy: en la implementación de optimizadores.
  - Interfaces puras para capas y funciones de pérdida.

- **Estructura de carpetas:**

```plaintext
proyecto-final/
├── src/
│   ├── tensor.h              # Estructura principal para almacenamiento de datos
│   ├── nn_interfaces.h       # Interfaces ILoss, ILayer, IOptimizer
│   ├── nn_dense.h            # Implementación de la capa densa
│   ├── nn_activation.h       # Capas de activación: ReLU, Sigmoid
│   ├── nn_loss.h             # Funciones de pérdida: MSE, BCE
│   ├── nn_optimizer.h        # Optimizadores: SGD y Adam
│   ├── neural_network.h      # Clase de alto nivel para entrenamiento y predicción
│   └── main.cpp              # Ejemplo de uso
```

#### 2.2 Manual de uso y casos de prueba

- **Cómo ejecutar**:

```bash
./neural_net_demo input.csv output.csv
```

- **Casos de prueba**:
  - Capa densa: verificación de forward y backward.
  - Activaciones: verificación de ReLU y Sigmoid.
  - Entrenamiento sobre el dataset AND y XOR.
  - Comparación de pérdida MSE vs BCE.

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.

Pasos:

1. Preparar archivo CSV con datos de entrada y salida.
2. Ejecutar el programa principal con los parámetros adecuados.
3. Evaluar los resultados impresos en consola o exportar a archivo.

---

### 4. Análisis del rendimiento

- **Métricas ejemplo** (con dataset XOR):

  - Iteraciones: 500 épocas.
  - Tiempo de entrenamiento: 4.8s.
  - Precisión final: 97.3%.

- **Ventajas/Desventajas**:

  -
    - Código ligero y comprensible.
  - − Sin paralelización ni soporte GPU.

- **Mejoras futuras**:

  - Integrar BLAS para acelerar la multiplicación matricial.
  - Añadir soporte para entrenamiento por lotes dinámicos.

---

### 5. Trabajo en equipo

| Tarea              | Miembro       | Rol                       |
| ------------------ |---------------|---------------------------|
| Investigación teórica | Diego         | Documentar bases teóricas |
| Implementación del modelo | Raul, Leandro | Código C++ de la NN       |
| Pruebas y benchmarking | Mauricio      | Generación de métricas    |
| Documentación      | Andy          | Documentación             |

---

### 6. Conclusiones

- **Logros**:

  - Se desarrolló una red neuronal funcional en C++.
  - Se experimentaron múltiples combinaciones de activaciones, pérdidas y optimizadores.

- **Evaluación**:

  - El sistema cumple los objetivos educativos planteados.

- **Aprendizajes**:

  - Profundización en backpropagation y arquitectura MLP.
  - Manejo de estructuras de datos complejas (tensores).

- **Recomendaciones**:

  - Mejorar la eficiencia computacional.
  - Explorar capas convolucionales y problemas reales como MNIST.

---

### 7. Bibliografía

[1] I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press, 2016. [Online]. Disponible: [https://www.deeplearningbook.org](https://www.deeplearningbook.org)

[2] M. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015. Disponible: [http://neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)

[3] S. Ruder, "An overview of gradient descent optimization algorithms", arXiv preprint arXiv:1609.04747, 2016.

[4] F. Chollet, *Deep Learning with Python*, 2nd ed., Manning, 2021.

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

