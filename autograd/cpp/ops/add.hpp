#pragma once

#include <iostream>;
#include "function.hpp"

class Add : public Function {
public:
	Varaible forward(Context ctx, Varaible& a, Variable& b) override {
		ctx.save_for_backward({NULL, NULL});
		Variable out = a + b;  // need to implement actual logic here
		return out;

	};
	std::vector<float> backward(Context ctx, std::vector<float> grad_output) override {
		return {grad_output, grad_output};
	};

};
