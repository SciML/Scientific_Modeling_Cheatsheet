#!/usr/bin/env julia

"""
Test suite for Scientific Modeling Cheatsheet
Runs the generator which tests all code examples
"""

# Run the generator script
include("../src/generate_cheatsheet.jl")

# The generator returns 0 on success, 1 on failure
exit_code = main()

if exit_code == 0
    println("\n✅ All tests passed!")
else
    println("\n❌ Some tests failed. See output above for details.")
    exit(exit_code)
end