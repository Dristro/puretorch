/*

`Variable`, used to perform autograd functionality.

*/

#pragma once

#include <iostream>
#include <vector>

#include "context.hpp"
#include "ops/function.hpp"

class Variable {
public:
	Variable();

	// Varaible (main) attrs
	bool requires_grad;
	Function grad_fn;
	bool is_leaf;

	// Operations (they call to ops)
	Variable add(Variable other);
	Variable sub(Variable other);
	Variable mul(Variable other);
	Variable div(Variable other);
	Variable neg();
	Variable pow(float y);
	Variable exp();
	Variable log(float base);
	Variable matmul(Variable other);
	Variable transpose();
	Variable reshape(std::vector<int> shape);
	Variable relu();
	Variable mean(int dim);

	// Generic functions and properties
	void zero_grad(bool set_to_none = true);
	void requires_grad_(bool value = true);
	std::vector<float> get_data();
	std::vector<float> grad();
	std::vector<int> shape();
	int ndim();


	// Backward logic
	void backward(std::optional<float> gradient = 1.0);

private:
	std::vector<float> data;
	std::vector<float> grad;

	
	std::vector<int> shape;
	int ndim;
	int version;

	void bump_version();
};
