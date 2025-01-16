module SIRLT

using General
using NetworkDynamics
using DifferentialEquations
using ColorTypes
using Graphs
using Random: shuffle!
using Statistics: mean
using Roots: find_zero

export
    SIR, SIRLT,
    SIR_termination_cb,
    SIRLT_termination_cb,
    init!, init,
    epi, opi,
    ρₛ, ρₒ,
    cmₛ, cmₒ,
    solve_iter_full,
    solve_iter_final!, solve_iter_final,
    solve_iter_final_zeros!,
    β_estimation,
    β_critical

include("sirlt.jl")

end # module SIRLT
