#!/usr/bin/env julia

"""
Test suite for Scientific Modeling Cheatsheet
Runs the generator which extracts and validates all code examples
"""

# Run the generator script
include("../src/generate_cheatsheet.jl")

# Call the main function
examples = generate_cheatsheet_from_original()

println("\n✅ Generated cheatsheet with $(length(examples)) code examples!")
println("✅ HTML file: scientific_modeling_cheatsheet.html")
println("✅ Extracted examples: extracted_examples.txt")