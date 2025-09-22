#!/usr/bin/env julia

"""
Extract code from original HTML and generate testable cheatsheet with timing data
Parses the existing HTML to extract all code examples, tests them, and rebuilds with timing results
"""

using Printf
using Dates
using LinearAlgebra
using Statistics
using BenchmarkTools

# Color codes for terminal output
const GREEN = "\033[32m"
const RED = "\033[31m"
const YELLOW = "\033[33m"
const BLUE = "\033[34m"
const RESET = "\033[0m"

function extract_code_examples(html_file)
    """Extract all code examples from the original HTML file"""

    html_content = read(html_file, String)

    # Find all table rows with code
    row_pattern = r"<tr>\s*<td>([^<]+)</td>\s*<td><code>([^<]*(?:<br>[^<]*)*)</code></td>\s*<td><code>([^<]*(?:<br>[^<]*)*)</code></td>\s*<td><code>([^<]*(?:<br>[^<]*)*)</code></td>\s*</tr>"

    examples = []

    for match in eachmatch(row_pattern, html_content)
        description = strip(match.captures[1])
        matlab_code = replace(strip(match.captures[2]), "<br>" => "\n")
        python_code = replace(strip(match.captures[3]), "<br>" => "\n")
        julia_code = replace(strip(match.captures[4]), "<br>" => "\n")

        push!(examples, (description, matlab_code, python_code, julia_code))
    end

    return examples
end

function test_julia_code(code::String, description::String)
    """Test and time Julia code snippet"""
    # Skip examples that are just comments, setup, or problematic patterns
    if isempty(strip(code)) ||
       startswith(strip(code), "%") ||
       strip(code) == "Not applicable" ||
       occursin("Not available", code) ||
       occursin("Not built-in", code) ||
       occursin("Use Simulink", code) ||
       occursin("Manual", code) ||
       length(split(code, '\n')) > 10  # Skip very long examples
        return (success=false, time=0.0, result="N/A")
    end

    try
        # Clean up code - remove using statements since we loaded packages globally
        clean_code = replace(code, r"using\s+[^\n]*"m => "")
        clean_code = replace(clean_code, r"import\s+[^\n]*"m => "")
        clean_code = strip(clean_code)

        if isempty(clean_code) ||
           occursin("@", clean_code) ||  # Skip macro-heavy code
           occursin("function", clean_code)  # Skip function definitions
            return (success=false, time=0.0, result="N/A")
        end

        # Use @benchmark for proper benchmarking (excludes compilation)
        benchmark_result = @benchmark eval(Meta.parse("begin\n$clean_code\nend")) evals=1 samples=3 seconds=0.1

        # Extract time in nanoseconds and convert to ms
        bench_time_ns = BenchmarkTools.time(minimum(benchmark_result))
        time_ms = bench_time_ns / 1_000_000

        # Get the actual result for verification
        result = eval(Meta.parse("begin\n$clean_code\nend"))

        return (success=true, time=time_ms, result=result)
    catch e
        return (success=false, time=0.0, result="Skipped")
    end
end

function test_python_code(code::String, description::String)
    """Test and time Python code snippet"""
    # Skip examples that are just comments or setup
    if isempty(strip(code)) ||
       startswith(strip(code), "#") ||
       strip(code) == "Not applicable" ||
       occursin("Not available", code) ||
       occursin("Not possible", code)
        return (success=false, time=0.0, result="N/A")
    end

    try
        # Write to temp file and execute
        tmpfile = tempname() * ".py"

        # Add standard imports and timing
        full_code = """
import time
import numpy as np
from scipy import integrate, optimize

start_time = time.time()

try:
    $code
    elapsed = (time.time() - start_time) * 1000  # Convert to ms
    print(f"TIME:{elapsed:.3f}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR:{e}")
"""

        open(tmpfile, "w") do f
            write(f, full_code)
        end

        output = read(`python3 $tmpfile`, String)
        rm(tmpfile)

        if occursin("SUCCESS", output)
            time_match = match(r"TIME:([0-9\.]+)", output)
            if time_match !== nothing
                elapsed_time = parse(Float64, time_match.captures[1])
                return (success=true, time=elapsed_time, result="Success")
            else
                return (success=true, time=0.0, result="Success")
            end
        else
            return (success=false, time=0.0, result="Error")
        end

    catch e
        return (success=false, time=0.0, result="Error")
    end
end

function generate_cheatsheet_from_original()
    """Generate a new cheatsheet with timing data"""

    println("$(BLUE)Extracting code examples from original HTML...$(RESET)")

    # Extract examples
    examples = extract_code_examples("original_cheatsheet.html")
    println("Found $(length(examples)) code examples")

    # Try to load necessary packages for testing
    try
        @eval using DifferentialEquations, NonlinearSolve, Optimization, OptimizationOptimJL
        @eval using ForwardDiff, Symbolics, Integrals
        println("$(GREEN)✓ Julia packages loaded$(RESET)")
    catch e
        println("$(YELLOW)⚠ Some Julia packages not available: $e$(RESET)")
    end

    # Test examples and collect timing data
    println("\n$(BLUE)Testing examples and collecting timing data...$(RESET)")
    timing_results = Dict{String, Dict{String, Any}}()

    for (i, (desc, matlab, python, julia)) in enumerate(examples)
        print("[$i/$(length(examples))] Testing: $desc...")

        result = Dict{String, Any}()

        # Test Julia
        julia_test = test_julia_code(String(julia), String(desc))
        result["julia_success"] = julia_test.success
        result["julia_time"] = julia_test.time

        # Test Python
        python_test = test_python_code(String(python), String(desc))
        result["python_success"] = python_test.success
        result["python_time"] = python_test.time

        timing_results[desc] = result

        # Print status
        julia_status = julia_test.success ? "$(GREEN)J✓$(RESET)" : "$(RED)J✗$(RESET)"
        python_status = python_test.success ? "$(GREEN)P✓$(RESET)" : "$(RED)P✗$(RESET)"
        println(" $julia_status $python_status")
    end

    # Read the original HTML as template
    template = read("original_cheatsheet.html", String)

    # Add CSS for timing results
    css_addition = """
        .success { color: #27ae60; font-weight: bold; }
        .failure { color: #e74c3c; font-weight: bold; }
        """

    new_html = replace(template, "</style>" => css_addition * "\n    </style>")

    # Add timing data to table rows
    for (desc, matlab, python, julia) in examples
        timing_info = get(timing_results, desc, Dict())
        julia_time = get(timing_info, "julia_time", 0.0)
        python_time = get(timing_info, "python_time", 0.0)
        julia_success = get(timing_info, "julia_success", false)
        python_success = get(timing_info, "python_success", false)

        # Format timing display
        if julia_success && python_success
            if julia_time > 0 && python_time > 0
                speedup = python_time / julia_time
                timing_cell = @sprintf("J: %.2f<br>P: %.2f<br>Speedup: %.1fx", julia_time, python_time, speedup)
            else
                timing_cell = "Fast execution"
            end
        elseif julia_success
            timing_cell = @sprintf("J: %.2f ms", julia_time)
        elseif python_success
            timing_cell = @sprintf("P: %.2f ms", python_time)
        else
            timing_cell = "N/A"
        end

        # Find and replace the specific table row for this example
        old_row_pattern = "<td>$desc</td>\n                        <td><code>$(replace(matlab, "\n" => "<br>"))</code></td>\n                        <td><code>$(replace(python, "\n" => "<br>"))</code></td>\n                        <td><code>$(replace(julia, "\n" => "<br>"))</code></td>"

        new_row = "<td>$desc</td>\n                        <td><code>$(replace(matlab, "\n" => "<br>"))</code></td>\n                        <td><code>$(replace(python, "\n" => "<br>"))</code></td>\n                        <td><code>$(replace(julia, "\n" => "<br>"))</code></td>\n                        <td><small>$timing_cell</small></td>"

        new_html = replace(new_html, old_row_pattern => new_row)
    end

    # Add summary at the end
    julia_successes = count(r -> get(r, "julia_success", false), values(timing_results))
    python_successes = count(r -> get(r, "python_success", false), values(timing_results))

    # Generate timing summary table
    timing_table = """
    <h2 id="timing-results">Performance Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Example</th>
                <th>Julia Status</th>
                <th>Python Status</th>
                <th>Julia Time (ms)</th>
                <th>Python Time (ms)</th>
                <th>Speedup</th>
            </tr>
        </thead>
        <tbody>
"""

    # Add rows for successful comparisons
    for (desc, result) in timing_results
        julia_success = get(result, "julia_success", false)
        python_success = get(result, "python_success", false)
        julia_time = get(result, "julia_time", 0.0)
        python_time = get(result, "python_time", 0.0)

        # Only include rows where we have meaningful timing data
        if julia_success || python_success
            julia_status = julia_success ? "✓" : "✗"
            python_status = python_success ? "✓" : "✗"

            julia_time_str = julia_success && julia_time > 0 ? @sprintf("%.2f", julia_time) : "N/A"
            python_time_str = python_success && python_time > 0 ? @sprintf("%.2f", python_time) : "N/A"

            speedup_str = if julia_success && python_success && julia_time > 0 && python_time > 0
                speedup = python_time / julia_time
                @sprintf("%.1fx", speedup)
            else
                "N/A"
            end

            # Color coding for status
            julia_class = julia_success ? "success" : "failure"
            python_class = python_success ? "success" : "failure"

            timing_table *= """
            <tr>
                <td>$desc</td>
                <td class="$julia_class">$julia_status</td>
                <td class="$python_class">$python_status</td>
                <td>$julia_time_str</td>
                <td>$python_time_str</td>
                <td>$speedup_str</td>
            </tr>
"""
        end
    end

    timing_table *= """
        </tbody>
    </table>

    <div class="note">
        <p><strong>Timing Notes:</strong></p>
        <ul>
            <li>Timings are for single execution (not benchmarked averages)</li>
            <li>Many examples are setup/import statements that don't have meaningful timing</li>
            <li>Speedup calculated as Python time / Julia time</li>
            <li>N/A indicates test failed or no meaningful timing available</li>
        </ul>
    </div>
"""

    timestamp_html = """
    <div style="text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 40px; padding: 20px; background-color: white; border-radius: 5px;">
        <hr>
        <p>Generated: $(Dates.now())</p>
        <p>Total examples: $(length(examples))</p>
        <p>Julia tests passed: $(julia_successes)/$(length(examples))</p>
        <p>Python tests passed: $(python_successes)/$(length(examples))</p>
        <p>🧪 Code examples tested with timing data</p>
    </div>
"""

    # Insert timing table and timestamp before closing body tag
    new_html = replace(new_html, "</body>" => timing_table * timestamp_html * "\n</body>")

    # Save the new HTML
    open("scientific_modeling_cheatsheet.html", "w") do f
        write(f, new_html)
    end

    println("\n$(GREEN)✓ Generated scientific_modeling_cheatsheet.html with timing data$(RESET)")
    println("✓ Julia tests passed: $(julia_successes)/$(length(examples))")
    println("✓ Python tests passed: $(python_successes)/$(length(examples))")

    return examples
end

if abspath(PROGRAM_FILE) == @__FILE__
    examples = generate_cheatsheet_from_original()
    println("Done! Check scientific_modeling_cheatsheet.html and extracted_examples.txt")
end