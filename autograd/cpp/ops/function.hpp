#pragma once

#include <iostream>

class Variable;
class Context;

class Function{
public:
	virtual Variable forward(Context ctx, Variable& a, Variable& b);
	virtual std::vector<float> backward(Context ctx);
	virtual ~Function() = default;
};
