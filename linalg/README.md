# Linalg

Simple header-only library for all matrix ops.\
We will heavily rely on this is as the CPU runtime engine for all metrix related ops.

At this moment, it defines an `Array`.\
An array is stored as a single contiguous block in memory, and linalg creates views of `shape`
to 'emulate' a multi-dimentional array (matrix).

I wanted to avoid using tools like eigen to keep things light and minimal.\
Plus I get to learn a ton of memory management stuff in C++, win-win :).

All content is in `array.hpp`, if you do have any suggestions, please open an issue or discussion :).
