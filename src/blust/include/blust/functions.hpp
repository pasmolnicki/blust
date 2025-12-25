#pragma once

#include <functional>

#include <blust/types.hpp>
#include <blust/backend/ops_tensor.hpp>

START_BLUST_NAMESPACE

class base_error_functor {
public:

    virtual number_t error(tensor_t& outputs, tensor_t& expected) = 0;
    virtual void d_cost(tensor_t& outputs, tensor_t& expected, tensor_t& result) = 0;

    tensor_t d_cost(tensor_t& outputs, tensor_t& expected) {
        tensor_t result(outputs.dim());
        d_cost(outputs, expected, result);
        return result;
    }
};

class base_activation_functor {
public:
    virtual void activation(tensor_t& weighted_input, tensor_t& result) = 0;
    virtual void derivative(tensor_t& activations, tensor_t& result) = 0;
    
    // Convenience methods
    tensor_t activation(tensor_t& weighted_input) {
        tensor_t result(weighted_input.dim());
        activation(weighted_input, result);
        return result;
    }

    tensor_t derivative(tensor_t& activations) {
        tensor_t result(activations.dim());
        derivative(activations, result);
        return result;
    }
};

typedef std::shared_ptr<base_activation_functor> base_activation_t;
typedef std::shared_ptr<base_error_functor> error_functor_t;


// F(x) = x, dF/dx = 1
class Identity : public base_activation_functor {
public:
    void activation(tensor_t& weighted_input, tensor_t& result) override {
        for (size_t i = 0; i < weighted_input.size(); ++i)
            result(i) = weighted_input(i);
    }

    void derivative(tensor_t& activations, tensor_t& result) override {
        result.fill(1.0f);
    }
};

// f(x) = { 0, if x <= 0; x if x > 0}
class ReLU : public base_activation_functor {
public:
    // f(x) = { 0, if x <= 0; x if x > 0}
    void activation(tensor_t& weighted_input, tensor_t& result) override
    {
        for (size_t i = 0; i < result.size(); ++i)
            result(i) = weighted_input(i) > 0 ? weighted_input(i) : 0;
    }

    // d(f(x)) / d(x) = { 0 if f(x) <= 0, 1 if f(x) > 0}
    void derivative(tensor_t& activations, tensor_t& result) override
    {
        for (size_t i = 0; i < result.size(); ++i)
            result(i) = tensor_t::ntype(activations(i) > 0);
    }
};

// f(x) = 1 / (1 + exp(-x))
class Sigmoid : public base_activation_functor {
public:
    // f(x) = 1 / (1 + exp(-x))
    void activation(tensor_t& weighted_input, tensor_t& result) override
    {
        for (size_t i = 0; i < result.size(); ++i) {
			result(i) = 1.0f / (1.0f + expf(-weighted_input(i)));
        }
    }

    // d(f(x)) / d(x) = f(x) * (1 - f(x))
    void derivative(tensor_t& activations, tensor_t& result) override
    {
        for (size_t i = 0; i < result.size(); ++i)
            result(i) = activations(i) * (1.0f - activations(i));
    }
};

// f([x1, x2, ..., xn]) = [ exp(x1) / sum, exp(x2) / sum, ..., exp(xn) / sum]
// where sum = exp(x1) + exp(x2) + ... + exp(xn)
class Softmax : public base_activation_functor {
public:
    // f([x1, x2, ..., xn]) = [ exp(x1) / sum, exp(x2) / sum, ..., exp(xn) / sum]
    // where sum = exp(x1) + exp(x2) + ... + exp(xn)
    void activation(tensor_t& weighted_input, tensor_t& result) override
    {
        number_t sum = 0;
        for (size_t i = 0; i < result.size(); ++i)
        {
            result(i) = expf(weighted_input(i));
            sum += result(i);

            if (std::isnan(result(i)))
				result(i) = 0;
        }

		// add a constant to avoid division by zero
		sum += 1e-4;

        for (size_t i = 0; i < result.size(); ++i)
        {
            result(i) /= sum;
			if (std::isnan(result(i)))
				result(i) = 0;
        }
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
    void derivative(tensor_t& activations, tensor_t& results) override
    {
        for (size_t i = 0; i < results.size(); ++i)
            results(i) = activations(i) * (1.0f - activations(i));
    }
};

// Sum of squared diffrences of output_i and expected_i
// S : (1 / N) * Sum from i = 0, to N (outputs(i) - expected(i))^2
class MeanSquaredError : public base_error_functor{
public:

    using base_error_functor::d_cost;
    using base_error_functor::error;

    // Sum of squared diffrences of output_i and expected_i
    // S : (1 / N) * Sum from i = 0, to N (outputs(i) - expected(i))^2
    number_t error(tensor_t& outputs, tensor_t& expected) override {
        number_t err = 0;
        number_t diff = 0;

        for (size_t i = 0; i < outputs.size(); i++) {
            diff = (outputs(i) - expected(i));
            err += diff * diff;
        }

        return err;
    }

    // Get dC / dA matrix
    void d_cost(tensor_t& outputs, tensor_t& expected, tensor_t& dC) override
    {
        for (size_t i = 0; i < outputs.size(); ++i) {
            dC(i) = 2 * (outputs(i) - expected(i));
        }
    }
};

inline base_error_functor* get_error_functor(error_funcs type)
{
    switch(type){
    case mean_squared_error:
        default:
            return new MeanSquaredError();
    }
}

/**
 * @brief Get the activation and derivative functions from a type
 */
inline base_activation_t get_base_activation(activation_type type)
{
    switch (type)
    {
        case relu:
            return std::make_shared<ReLU>();
        case sigmoid:
            return std::make_shared<Sigmoid>();
        case softmax:
            return std::make_shared<Softmax>();
        case none:
        default:
            return std::make_shared<Identity>();
    }
}

END_BLUST_NAMESPACE