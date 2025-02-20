#pragma once

#include "types.hpp"

START_BLUST_NAMESPACE

enum activations {
    relu,
    sigmoid,
    softmax,
};

enum error_funcs {
    mean_squared_error
};

class BaseErrorFunction {
public:
    virtual number_t error(matrix_t& outputs, matrix_t& expected) = 0;
    virtual matrix_t d_cost(matrix_t& outputs, matrix_t& expected) = 0;
};

typedef std::function<matrix_t(matrix_t&)> base_function_t;
typedef std::unique_ptr<BaseErrorFunction> base_error_func_t;

typedef struct function_info {
    base_function_t activ;
    base_function_t deriv;
} function_info;

class ReLU {
public:
    // f(x) = { 0, if x <= 0; x if x > 0}
    static matrix_t activation(matrix_t& weighted_input) 
    {
        matrix_t mapped(weighted_input);
        for (size_t i = 0; i < mapped.size(); ++i)
            mapped(i) = mapped(i) > 0 ? mapped(i) : 0;
        return mapped;
    }

    // d(f(x)) / d(x) = { 0 if f(x) <= 0, 1 if f(x) > 0}
    static matrix_t derivative(matrix_t& activations)
    {
        matrix_t mapped(activations);
        for (size_t i = 0; i < mapped.size(); ++i)
            mapped(i) = int(mapped(i) > 0);
        return mapped;
    }
};

class Sigmoid {
public:
    // f(x) = 1 / (1 + exp(-x))
    static matrix_t activation(matrix_t& weighted_input) 
    {
        matrix_t mapped(weighted_input);
        for (size_t i = 0; i < mapped.size(); ++i)
            mapped(i) = mapped(i) > 0 ? mapped(i) : 0;
        return mapped;
    }

    // d(f(x)) / d(x) = f(x) * (1 - f(x))
    static matrix_t derivative(matrix_t& activations)
    {
        matrix_t mapped(activations);
        for (size_t i = 0; i < mapped.size(); ++i)
            mapped(i) = mapped(i) * (1.0 - mapped(i));
        return mapped;
    }
};

class Softmax {
public:
    // f([x1, x2, ..., xn]) = [ exp(x1) / sum, exp(x2) / sum, ..., exp(xn) / sum]
    // where sum = exp(x1) + exp(x2) + ... + exp(xn)
    static matrix_t activation(matrix_t& weighted_input) 
    {
        matrix_t softmax(weighted_input);
        number_t sum = 0;
        
        for (size_t i = 0; i < softmax.size(); ++i)
        {
            softmax(i) = expf(softmax(i));
            sum += softmax(i);
        }

        for (size_t i = 0; i < softmax.size(); ++i)
        {
            softmax(i) /= sum;
        }

        return softmax;
    }

    // d(f(x)) / d(x) = softmax * ( 1 - softmax) 
    // f(xi) = exp(xi) / sum
    // d(sum) / d(xi) = exp(xi)
    //
    // d(f(xi)) / d(xi) = (exp(xi) * sum - exp(xi) * exp(xi)) / sum^2  
    //
    //                  = (exp(xi) / sum) * ((sum - exp(xi))/ sum) 
    //
    //                  = softmax_i * ( 1 - softmax_i)
    static matrix_t derivative(matrix_t& activations)
    {
        matrix_t mapped(activations);
        for (size_t i = 0; i < mapped.size(); ++i)
            mapped(i) = mapped(i) * (1.0 - mapped(i));
        return mapped;
    }
};


class MeanSquaredError : public BaseErrorFunction{
public:

    // Sum of squared diffrences of output_i and expected_i
    // S : (1 / N) * Sum from i = 0, to N (outputs(i) - expected(i))^2
    number_t error(matrix_t& outputs, matrix_t& expected){
        number_t err = 0;
        number_t diff = 0;

        for (size_t i = 0; i < outputs.size(); i++) {
            diff = (outputs(i) - expected(i));
            err += diff * diff;
        }
        return err / outputs.size();
    }

    // Get dC / dA matrix
    matrix_t d_cost(matrix_t& outputs, matrix_t& expected)
    {
        matrix_t dC(outputs.dim());

        for (size_t i = 0; i < outputs.size(); ++i)
            dC(i) = 2 * (outputs(i) - expected(i));

        return dC;
    }
};

BaseErrorFunction* get_error_function(error_funcs type)
{
    switch(type){
        default:
            return new MeanSquaredError();
    }
}

/**
 * @brief Get the activation and derivative functions from a type
 */
function_info get_functions(activations type)
{
    switch (type)
    {
        case sigmoid:
            return {Sigmoid::activation, Sigmoid::derivative};
        case softmax:
            return {Softmax::activation, Softmax::derivative};
        default:
            return {ReLU::activation, ReLU::derivative};
    }
}

END_BLUST_NAMESPACE