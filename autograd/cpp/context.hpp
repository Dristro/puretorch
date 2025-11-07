/*

`Context`, used to store the reference to tensors during ops.

Each context object will store the reference to the tensors used.

Manipulate `Context` via:
	save_for_backward(): stores the tensor references to object
	get_saved_tensors(): returns the tensor references

Context also tracks the version-snapshot of itself, this is used
to maintain a linear compute-graph (without inplace operations etc).

*/

#pragma once

#include <iostream>

class Variable;

class Context {
private:
	std::vector<Variable*> saved_tensors;
	int version_snapshot;

public:
	Context();

	void save_for_backward(std::vector<Variable*>& tensors) {
		saved_tensors = tensors;
	};
	std::vector<Variable*> get_saved_tensors(){
		return saved_tensors;
	};
};
