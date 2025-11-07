#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <cassert>

namespace linarg{

class Array{
public:
    
    // Array attrs, data members
    std::vector<float> data;
    std::vector<int> shape;

    // Constructors
    Array() = default;

    // Create from shape, filled with zero (defailt)
    explicit Array(const std::vector<int> shape_, float fill = 0.0f)
    : shape(shape_) {
        data.resize(numel());
        if (fill != 0.0f)
            std::fill(data.begin(), data.end(), fiil);
    }

    // Create from flat vector + shape (shape must match length)
    Array(std::vecotr<float> data_, const std::vector<int>& shape_)
    : data(std::move(data_)), shape(shape_) {
        if (numel() != static_cast<int>(data.size()))
            throw std::runtime_error("Array: data size does not match shape");
    }

    // 


};
}



namespace linalg {

class Array {
public:
    // -- member data
    std::vector<float> data;
    std::vector<int> shape; // e.g., {2,3,4}

    // -- constructors
    Array() = default;

    // Create from shape, filled with zero
    explicit Array(const std::vector<int>& shape_, float fill = 0.0f)
        : shape(shape_) {
        data.resize(numel());
        if (fill != 0.0f) std::fill(data.begin(), data.end(), fill);
    }

    // Create from flat vector + shape (shape must match length)
    Array(std::vector<float> data_, const std::vector<int>& shape_)
        : data(std::move(data_)), shape(shape_) {
        if (numel() != static_cast<int>(data.size()))
            throw std::runtime_error("Array: data size does not match shape");
    }

    // Create from initializer list for 1-D arrays
    Array(std::initializer_list<float> elems)
        : data(elems), shape({static_cast<int>(elems.size())}) {}

    // Copy / move default
    Array(const Array&) = default;
    Array(Array&&) noexcept = default;
    Array& operator=(const Array&) = default;
    Array& operator=(Array&&) noexcept = default;

    // -- utilities
    int ndim() const { return static_cast<int>(shape.size()); }
    int numel() const {
        if (shape.empty()) return 0;
        long long n = 1;
        for (int d : shape) n *= d;
        return static_cast<int>(n);
    }

    bool empty() const { return data.empty(); }

    // Returns pointer to raw data (useful forinterop)
    float* data_ptr() { return data.empty() ? nullptr : data.data(); }
    const float* data_ptr() const { return data.empty() ? nullptr : data.data(); }

    // reshape in place (throws if elements mismatch)
    void reshape(const std::vector<int>& new_shape) {
        long long new_num = 1;
        for (int d : new_shape) new_num *= d;
        if (new_num != numel()) throw std::runtime_error("reshape: incompatible size");
        shape = new_shape;
    }

    // return a reshaped copy
    Array reshaped(const std::vector<int>& new_shape) const {
        Array out = *this;
        out.reshape(new_shape);
        return out;
    }

    // simple 2D transpose (only for 2-D arrays)
    Array transpose() const {
        if (ndim() != 2) throw std::runtime_error("transpose: only for 2-D arrays");
        int r = shape[0], c = shape[1];
        Array out({c, r});
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                out.data[j * r + i] = data[i * c + j];
        return out;
    }

    // matmul for 2D matrices: this (m x k) @ other (k x n) => (m x n)
    Array matmul(const Array& other) const {
        if (ndim() != 2 || other.ndim() != 2)
            throw std::runtime_error("matmul: both operands must be 2-D");
        int m = shape[0], k = shape[1];
        int k2 = other.shape[0], n = other.shape[1];
        if (k != k2) throw std::runtime_error("matmul: inner dimensions mismatch");
        Array out({m, n}, 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int t = 0; t < k; ++t) {
                    acc += data[i * k + t] * other.data[t * n + j];
                }
                out.data[i * n + j] = acc;
            }
        }
        return out;
    }

    // elementwise map
    Array map_unary(const std::function<float(float)>& fn) const {
        Array out = *this;
        for (size_t i = 0; i < data.size(); ++i) out.data[i] = fn(data[i]);
        return out;
    }

    // elementwise binary with same shape (or scalar RHS)
    Array map_binary(const Array& other, const std::function<float(float, float)>& fn) const {
        if (shape == other.shape) {
            Array out(shape);
            out.data.resize(data.size());
            for (size_t i = 0; i < data.size(); ++i) out.data[i] = fn(data[i], other.data[i]);
            return out;
        }
        // allow other to be scalar (shape empty or numel == 1)
        if (other.numel() == 1) {
            float v = other.data[0];
            Array out(shape);
            for (size_t i = 0; i < data.size(); ++i) out.data[i] = fn(data[i], v);
            return out;
        }
        // allow this to be scalar (numel == 1)
        if (this->numel() == 1) {
            float v = this->data[0];
            Array out(other.shape);
            for (size_t i = 0; i < out.data.size(); ++i) out.data[i] = fn(v, other.data[i]);
            return out;
        }
        throw std::runtime_error("map_binary: unsupported broadcast (only same-shape or scalar supported)");
    }

    // elementwise ops (non-mutating)
    Array add(const Array& other) const { return map_binary(other, [](float a, float b) { return a + b; }); }
    Array sub(const Array& other) const { return map_binary(other, [](float a, float b) { return a - b; }); }
    Array mul(const Array& other) const { return map_binary(other, [](float a, float b) { return a * b; }); }
    Array div(const Array& other) const { return map_binary(other, [](float a, float b) { return a / b; }); }

    // scalar convenience
    Array add(float scalar) const { return map_unary([scalar](float a){ return a + scalar; }); }
    Array mul(float scalar) const { return map_unary([scalar](float a){ return a * scalar; }); }
    Array div(float scalar) const { return map_unary([scalar](float a){ return a / scalar; }); }

    // element access by flattened index (no bounds check)
    float& operator[](size_t idx) { return data[idx]; }
    const float& operator[](size_t idx) const { return data[idx]; }

    // optionally access by multi-dimensional index
    float& at(const std::vector<int>& idx) {
        size_t off = offset(idx);
        return data[off];
    }
    const float& at(const std::vector<int>& idx) const {
        size_t off = offset(idx);
        return data[off];
    }

    // in-place ops
    Array& operator+=(const Array& other) {
        if (shape == other.shape) {
            for (size_t i = 0; i < data.size(); ++i) data[i] += other.data[i];
            return *this;
        }
        if (other.numel() == 1) {
            float v = other.data[0];
            for (auto &x : data) x += v;
            return *this;
        }
        throw std::runtime_error("operator+=: unsupported broadcast");
    }
    Array& operator*=(float scalar) { for (auto &x : data) x *= scalar; return *this; }

    // arithmetic operators
    friend Array operator+(const Array& a, const Array& b) { return a.add(b); }
    friend Array operator-(const Array& a, const Array& b) { return a.sub(b); }
    friend Array operator*(const Array& a, const Array& b) { return a.mul(b); }
    friend Array operator/(const Array& a, const Array& b) { return a.div(b); }

    // print (debug)
    void print_debug() const {
        std::cout << "Array(shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i + 1 < shape.size()) std::cout << ",";
        }
        std::cout << "], data=[";
        for (size_t i = 0; i < std::min<size_t>(data.size(), 16); ++i) {
            std::cout << data[i];
            if (i + 1 < data.size()) std::cout << ",";
        }
        if (data.size() > 16) std::cout << "...";
        std::cout << "])\n";
    }

private:
    // compute flat offset from multi-dim index (row-major)
    size_t offset(const std::vector<int>& idx) const {
        if (idx.size() != shape.size()) throw std::runtime_error("offset: index dims mismatch");
        size_t off = 0;
        size_t stride = 1;
        // row-major: last dimension varies fastest
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            off += static_cast<size_t>(idx[i]) * stride;
            stride *= static_cast<size_t>(shape[i]);
        }
        return off;
    }
};

} // namespace linalg
