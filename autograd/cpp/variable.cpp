/*

`Variable` implementation.
Adding all the functionality to the class.

*/

#include "variable.hpp"
#include <context.hpp>

Variable::Variable(
	std::vector<float> data,
	bool requires_grad,
	Context grad_fn,
	bool is_leaf,
)
	: data(data),
	  requires_grad(requires_grad),
	  is_leaf(is_leaf),
	  ndim(0),
	  version(0),
{
	this->shape = { static_cast<int>(data.size()) };
}

/**
 * Performs Variable addition, backward safe.
 * Uses implementation from ops.
 * **/
Variable Variable::add(Variable other) {
	Variable out = this->data + other.data;
	return out;
}

/**
 * Sets the Variable's grad to all-zeros or NULL.
 * By default, sets grad to NULL for mem eff.
 * Args:
 *     set_to_none (bool): set grad to NULL if true
 * **/
void Variable::zero_grad(bool set_to_none = true) {
    if (set_to_none) {
    	this->grad = NULL;
    } else {
    	this->grad.clear();
    }
}

// Remove after test-build
int main() {
	std::cout << "Hello there\n";
	Variable var = Variable(
		{1, 2, 3},
		false,
		Context,
	);

	return 0;
}
