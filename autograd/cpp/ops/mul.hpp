#pragma once

#include <iostream>;
#include <vector>;
#include "function.hpp";

class Variable;

class Mul : public Function {
public:
	Varaible forward(Context ctx, Varaible& a, Variable& b) override {
		ctx.save_for_backward({ &a, &b });
		
		Variable out = a * b;  // need to implement actual logic here
		return out;
	};

	std::vector<float> backward(Context ctx, std::vector<float> grad_output) override {
		auto tensors = ctx.saved_tensors();
		Variable* a = tensors[0];
		Variable* b = tensors[1];
		
		std::vector<float> grad_a = grad_output * (*b).data;
		std::vector<float> grad_b = grad_output * (*a).data;

		return { grad_a, grad_b };
	};
};



class Mul : public Function {
public:
    Variable forward(Context& ctx, Variable& a, Variable& b) override {
        ctx.save_for_backward({ &a, &b });

        // Assuming Variable supports operator*
        Variable out = a * b;
        return out;
    }

    std::vector<Variable> backward(Context& ctx, const Variable& grad_output) override {
        auto tensors = ctx.saved_tensors();
        Variable* a = tensors[0];
        Variable* b = tensors[1];

        // Gradients: dL/da = grad_output * b, dL/db = grad_output * a
        Variable grad_a = grad_output * (*b);
        Variable grad_b = grad_output * (*a);

        return { grad_a, grad_b };
    }
};
