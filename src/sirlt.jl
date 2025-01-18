###############
# Basic rates #
###############

function P₁₁(es, β)
    return (1 - β)^count_states(es, 1, 2)
end

function P₂₂(μ)
    return 1 - μ
end

function P₃₃(ν)
    return 1 - ν
end

function linearC(es, α)
    return (α * count_states(es, 1, 2) + (1 - α) * count_states(es, 2, 2)) / length(es)
end

function linearC_positive(es, α)
    return (α * count_states(es, 1, 2) / length(es)) + ((1 - α) * (count_states(es, 2, 2) + 1) / (length(es) + 1))
end

function linearC_negative(es, α)
    return (α * count_states(es, 1, 2) / length(es)) + ((1 - α) * count_states(es, 2, 2) / (length(es) + 1))
end

function linearC(es, α, d₁, d₂)
    return α * count_states(es, 1, 2) / d₁ + (1 - α) * count_states(es, 2, 2) / d₂
end

function linearC_positive(es, α, d₁, d₂)
    return α * count_states(es, 1, 2) / d₁ + (1 - α) * (count_states(es, 2, 2) + 1) / (d₂ + 1)
end

function linearC_negative(es, α, d₁, d₂)
    α * count_states(es, 1, 2) / d₁ + (1 - α) * count_states(es, 2, 2) / (d₂ + 1)
end

function exponentialC(es, α₁, α₂; a=2.4)
    return 1 - exp(-1.0 * a * (α₁ * count_states(es, 1, 2) + α₂ * count_states(es, 2, 2)) / length(es))
end

function exponentialC(es, α₁, α₂, d₁, d₂; a=2.4)
    return 1 - exp(-1.0 * a * (α₁ * count_states(es, 1, 2) / d₁ + α₂ * count_states(es, 2, 2) / d₂))
end

function deterministicA(x, Θ)
    return x >= Θ
end

function stochasticA(x)
    return x >= rand()
end

############
# Dynamics #
############

function epi_vertex!(vₙ, v, es, p, ns)
    s = v[1]

    if s == 1
        vₙ[1] = rand() < P₁₁(es, p.β) ? 1 : 2
    elseif s == 2
        vₙ[1] = rand() < P₂₂(p.μ) ? 2 : ns
    elseif s == 3
        vₙ[1] = rand() < P₃₃(p.ν) ? 3 : 1
    end
end

function sir_vertex!(vₙ, v, es, p, t)
    epi_vertex!(vₙ, v, es, p, 3)
end

function sis_vertex!(vₙ, v, es, p, t)
    epi_vertex!(vₙ, v, es, p, 1)
end

function sirlt_vertex(; degrees=[], nl_stoc=false, inf_A=false, bias=nothing)
    if nl_stoc
        A = length(degrees) == 2 ?
            (es, p) -> stochasticA(exponentialC(es, p.α, p.α₂, degrees[1], degrees[2])) :
            (es, p) -> stochasticA(exponentialC(es, p.α, p.α₂))
    else
        C = isnothing(bias) ? linearC :
            bias ? linearC_positive :
                   linearC_negative
        A = length(degrees) == 2 ?
            (es, p) -> deterministicA(C(es, p.α, degrees[1], degrees[2]), p.Θ) :
            (es, p) -> deterministicA(C(es, p.α), p.Θ)
    end

    f = (vₙ, v, es, p, t) -> begin
        s = v[1]
        vₙ[2] = A(es, p) ? 2 : 1

        if s == 1 && vₙ[2] == 2
            vₙ[1] = 1
        else
            sir_vertex!(vₙ, v, es, p, t)
            if (inf_A && vₙ[1] == 2)
                vₙ[2] = 2
            end
        end
    end

    return ODEVertex(f=f, dim=2, sym=(:s, :o))
end

function epi_vertex()
    return ODEVertex(f=(vₙ, v, es, p, t) -> epi_vertex!(vₙ, v, es, p, 3), dim=2, sym=(:s, :o))
end

function SIR_dynamics(graph::AbstractGraph)
    return network_dynamics(epi_vertex(), IStaticEdge(2), graph)
end

function SIRLT_dynamics(graph::AbstractGraph; nl_stoc=false, inf_A=false, parallel=false)
    return network_dynamics(sirlt_vertex(nl_stoc=nl_stoc, inf_A=inf_A), IStaticEdge(2), graph; parallel=parallel)
end

function SIRLT_dynamics(graph::AbstractGraph, ω::Real; nl_stoc=false, inf_A=false, parallel=false)
    v_f = ds -> sirlt_vertex(degrees=ds, nl_stoc=nl_stoc, inf_A=inf_A, bias=(rand() <= ω))
    vertices = [v_f(degree(g, i)) for i in 1:nv(g)]
    return network_dynamics(vertices, IStaticEdge(2), graph; parallel=parallel)
end

function SIRLT_dynamics(g_epi::AbstractGraph, g_opi::AbstractGraph; nl_stoc=false, inf_A=false, parallel=false)
    v_f = ds -> sirlt_vertex(degrees=ds, nl_stoc=nl_stoc, inf_A=inf_A)
    vs, es, gc = combine_graphs(v_f, g_epi, g_opi)
    return network_dynamics(vs, es, gc; parallel=parallel)
end

function SIRLT_dynamics(g_epi::AbstractGraph, g_opi::AbstractGraph, ω::Real; nl_stoc=false, inf_A=false, parallel=false)
    v_f = ds -> sirlt_vertex(degrees=ds, nl_stoc=nl_stoc, inf_A=inf_A, bias=(rand() <= ω))
    vs, es, gc = combine_graphs(v_f, g_epi, g_opi)
    return network_dynamics(vs, es, gc; parallel=parallel)
end

function SIR_termination_condition(u, t)
    if t % 2 == 0 # Try winning time by not checking every timestep
        return count(u[1:2:end] .== 2) == 0
    else
        false
    end
end

function SIR_termination_condition(u, t, int)
    return SIR_termination_condition(u, t)
end

SIR_termination_cb = DiscreteCallback(SIR_termination_condition, terminate!; save_positions=(true, false))

function SIRLT_termination_condition(u, t, int)
    if t % 2 == 0 # Try winning time by not checking every timestep
        for i in 1:2:length(u)
            if u[i] == 2 || u[i+1] != int.uprev[i+1] # If infected or opinion changed
                return false
            end
        end
        return true
    else
        false
    end
end

SIRLT_termination_cb = DiscreteCallback(SIRLT_termination_condition, terminate!; save_positions=(true, false))

function init!(x₀; Θ=0.5, nb_I=1, prop_A=0.5, prerun=false, graph=nothing)
    N = length(x₀)/2 |> Int64

    s₀ = @view x₀[1:2:end]
    o₀ = @view x₀[2:2:end]

    s₀ .= 1
    o₀ .= 1 * (prop_A .> rand(N)) .+ 1

    if prerun
        @assert g isa AbstractGraph
        mnd = network_dynamics(majority_vertex(1, 2), IStaticEdge(1), graph)

        pm = (Θ=Θ,)

        mprob = DiscreteProblem(mnd, o₀, (1, round(sqrt(N))), pm)
        sol = solve(mprob, FunctionMap(); callback=majority_termination_cb)

        o₀ .= sol[end]
    end

    possibilities = findall(isequal(1), o₀)
    if length(possibilities) < nb_I
        possibilities = 1:N
    elseif length(possibilities) == nb_I
        s₀[possibilities] .= 2
        return
    end
    for _ in 1:nb_I
        infected = false
        while !infected
            i = rand(possibilities)
            if s₀[i] == 1
                s₀[i] = 2
                infected = true
            end
        end
    end
end

function init(nb_nodes; Θ=0.5, nb_I=1, prop_A=0.5, prerun=false, graph=nothing)
    x₀ = zeros(Int8, 2*nb_nodes)
    init!(x₀; Θ=Θ, nb_I=nb_I, prop_A=prop_A, prerun=prerun, graph=graph)

    return x₀
end

##################
# MC Simulations #
##################
function solve_ensemble(x₀, nd, x₀_gen!, p, Tₑ, Nb, output_func, save_everystep; early_termination=true)
    function update_x₀(prob, i, repeat)
        x₀_gen!(prob.u0)
        prob
    end

    x₀_gen!(x₀)
    prob = DiscreteProblem(nd, x₀, (1, Tₑ), p)
    eprob = EnsembleProblem(prob; output_func=output_func, prob_func=update_x₀, safetycopy=false)
    if early_termination
        return solve(eprob, FunctionMap(), EnsembleSerial(); callback=SIR_termination_cb, save_everystep=save_everystep, trajectories=Nb)
    else
        return solve(eprob, FunctionMap(), EnsembleSerial(); callback=SIRLT_termination_cb, save_everystep=save_everystep, trajectories=Nb)
    end
end

"""
    solve_iter_full(x₀, nd, x₀_gen!, p, Tₑ, Nb)

# Arguments
- `x₀`: Scratch space for initial condition
- `nd`: Network dynamics
- `x₀_gen!`: Function to overwrite initial condition `x₀`
- `p`: Parameters
- `Tₑ`: Final time
- `Nb`: Number of trajectories

# Returns
- `avgₛ`: Average density of the epidemiological states (dim 2) over time (dim 1)
- `avgₒ`: Average density of the awareness states (dim 2) over time (dim 1)
"""
function solve_iter_full(x₀, nd, x₀_gen!, p, Tₑ, Nb; early_termination=true)
    avgₛ = zeros(Tₑ, 4) 
    avgₒ = zeros(Tₑ, 2)
    max_t = 0

    sols = solve_ensemble(x₀, nd, x₀_gen!, p, Tₑ, Nb, (sol, i) -> (sol, false), true; early_termination=early_termination)
    for sol in sols
        t = sol.t[end]
        avgₛ[1:t, :] += ρₛ(sol)
        avgₒ[1:t, :] += ρₒ(sol)
        for row in t+1:Tₑ
            avgₛ[row, :] .+= ρₛ(sol[end])
            # Due to the callback, the awareness is not necessarily stable and
            # the following approach is not correct.
            avgₒ[row, :] .+= ρₒ(sol[end])
        end
        max_t = max(max_t, t)
    end

    avgₛ ./= Nb
    avgₒ ./= Nb

    for row in max_t+1:Tₑ
        avgₒ[row, :] .= avgₒ[max_t, :]
    end

    return avgₛ, avgₒ 
end


"""
    solve_iter_final!(avgₛ, avgₒ, x₀, nd, x₀_gen!, p, Tₑ, Nb)

Calculate the final average densities of the epidemiological (`avgₛ`) and aware (`avgₒ`) situation.

# Arguments
- `x₀`: the initial condition scratch space
- `x₀_gen!(x₀)`: a function that reinitilasis `x₀`
- `p`: the parameter vector
- `Tₑ`: the maximum number of time steps to run the simulation for
- `Nb`: the number of simulations to run.
"""
function solve_iter_final!(avgₛ, avgₒ, x₀, nd, x₀_gen!, p, Tₑ, Nb; early_termination=true)
    avgₛ .= 0.0
    avgₒ .= 0.0
    T = 0.0

    output_func = (sol, _) -> ((sol[end], sol.t[end]), false)

    for (sol, t) in solve_ensemble(x₀, nd, x₀_gen!, p, Tₑ, Nb, output_func, false; early_termination=early_termination)
        T += t
        avgₛ[:] += ρₛ(sol)
        avgₒ[:] += ρₒ(sol)
    end

    avgₛ ./= Nb
    avgₒ ./= Nb
    return T / Nb
end

"""
    solve_iter_final!(avgₛ, avgₒ, x₀, nd, x₀_gen!, p, Tₑ, Nb, graph::AbstractGraph; early_termination=true, min_T=1)

Calculate the final average densities of the epidemiological (`avgₛ`) and aware (`avgₒ`) situation.
Also return

- `T`: time it took to reach the final state
- `Β`: the equivalent `β` for epidemics withouth awareness
- `max_I`: the maximum number of infected nodes
- `T_mI`: the time at which the maximum number of infected nodes was reached
- `max_A`: the maximum number of aware nodes
- `T_mA`: the time at which the maximum number of aware nodes was reached

# Arguments
- `x₀`: the initial condition scratch space
- `x₀_gen!(x₀)`: a function that reinitilasis `x₀`
- `p`: the parameter vector
- `Tₑ`: the maximum number of time steps to run the simulation for
- `Nb`: the number of simulations to run.
- `graph`: the graph on which the simulation is run
- `early_termination`: whether to terminate the simulation early if the epidemic dies out
- `min_T`: the timestep from which to run time statistics
"""
function solve_iter_final!(avgₛ, avgₒ, x₀, nd, x₀_gen!, p, Tₑ, Nb, graph::AbstractGraph; early_termination=true, min_T=1)
    avgₛ .= 0.0
    avgₒ .= 0.0
    T = 0.0
    Β = 0.0
    max_I = 0.0
    T_mI = 0.0
    max_A = 0.0
    T_mA = 0.0

    output_func = (sol, i) -> ((sol[end], sol.t[end], β_estimation(sol, graph), time_stats(sol; min_T=min_T)), false)

    for (sol, t, β, T_stats) in solve_ensemble(x₀, nd, x₀_gen!, p, Tₑ, Nb, output_func, true; early_termination=early_termination)
        T += t
        Β += β
        max_I += T_stats[1]
        T_mI += T_stats[2]
        max_A += T_stats[3]
        T_mA += T_stats[4]
        avgₛ[:] += ρₛ(sol)
        avgₒ[:] += ρₒ(sol)
    end

    avgₛ ./= Nb
    avgₒ ./= Nb
    return T / Nb, Β / Nb, max_I / Nb, T_mI / Nb, max_A / Nb, T_mA / Nb
end

###########################
# Solution interpretation #
###########################

ρₛ(x::AbstractArray) =
    [count(x[1:2:end] .== 1),
        count(x[1:2:end] .== 2),
        count(x[1:2:end] .== 3),
        count(x[1:2:end] .== 4)] ./ (length(x) / 2);

ρₒ(x::AbstractArray) =
    [count(x[2:2:end] .== 1),
        count(x[2:2:end] .== 2)] ./ (length(x) / 2);

ρₛ(sol::ODESolution) = reduce(hcat, [ρₛ(sol[i]) for i in eachindex(sol)])';
ρₒ(sol::ODESolution) = reduce(hcat, [ρₒ(sol[i]) for i in eachindex(sol)])';

function epi(sol, t)
    t < 0 ? sol[end][1:2:end] : sol[t][1:2:end]
end

function epi(sol)
    [epi(sol, t) for t in sol.t]
end

function opi(sol, t)
    t < 0 ? sol[end][2:2:end] : sol[t][2:2:end]
end

function opi(sol)
    [opi(sol, t) for t in sol.t]
end

"""
    time_stats(sol; min_T=1)

Given the solution of a coupled epidemic awareness dynamics, calculate the peak and
time to peak values of the densities of infected and aware agents.
Values are calculated from timestep `min_T` onwards.

# returns
- max ρᴵ(t)
- argmax ρᴵ(t)
- max ρᴬ(t)
- argmax ρᴬ(t)
"""
function time_stats(sol; min_T=1)
    max_I = 0
    T_mI = 0
    max_A = 0
    T_mA = 0
    for t in min_T:lastindex(sol)
        cI = 0
        cA = 0
        for i in 1:2:length(sol[t])
            if sol[t][i] == 2
                cI += 1
            end
            if sol[t][i+1] == 2
                cA += 1
            end
        end
        if cI > max_I
            max_I = cI
            T_mI = t
        end
        if cA > max_A
            max_A = cA
            T_mA = t
        end
    end

    N = length(sol[1])/2

    return max_I/N, T_mI, max_A/N, T_mA
end

# β estimation functionality
"""
    get_S_I_nghb_count!(nghb_I_count::Vector{Int}, state, g::AbstractGraph)

Calculate the number of infected neighbours that each susceptible node has and
store the result in `nghb_I_count` (size=`nv(g)`). The value 0 indictes both that the node is
non-susceptible or that it has no infected neighbours.
"""
function get_S_I_nghb_count!(nghb_I_count::Vector{Int}, epi_state, g::AbstractGraph)
    @assert length(nghb_I_count) == nv(g)
    for i in 1:nv(g)
        nghb_I_count[i] = epi_state[i] == 1 ?
            count(isequal(2), view(epi_state, neighbors(g,i))) :
            0
    end
end

"""
    infection_event_count(sol, g::AbstractGraph)

Return `nbs`, a matrix of size `(max_degree(g), 2)` where `nbs[i, j]` is the number of times
a susceptible vertex with `i` infected neighbours became infected (`j==2`) or remained
susceptible (`j==1`).
"""
function infection_event_count(sol, g::AbstractGraph)
    nghb_I_count = zeros(Int64, nv(g))
    nbs = zeros(Int64, (maximum(degree(g)), 2))

    for t in 2:lastindex(sol)
        get_S_I_nghb_count!(nghb_I_count, view(sol, 1:2:size(sol, 1), t-1), g)

        for (i, n_inf) in enumerate(nghb_I_count)
            if n_inf == 0
                continue
            end
            nbs[n_inf, sol[t][2*i-1]] += 1
        end
    end

    return nbs
end

"""
    β_estimation(sol::ODESolution, g::AbstractGraph)

Estimate the equivalent β parameter to obtain the solution `sol` on the graph `g` without awareness.
"""
function β_estimation(sol, g::AbstractGraph)
    nbs = infection_event_count(sol, g)

    # Calculate the equivalent β parameter based on the infection events
    β = 0.0
    total_w = 0.0
    for (n_inf, occurs) in enumerate(eachrow(nbs))
        if occurs[1] != 0
            nb = sum(occurs)
            total_w += nb
            β += nb * (1 - (1 - (occurs[2] / nb))^(1 / n_inf))
        end
    end

    return β / total_w
end

function β_critical(x₀, nd, x₀_gen!, p; prop_I=0.5, Tₑ=800, Nb=10)
    threshold = prop_I * length(x₀) / 2

    function F(β)
        p_curr = haskey(p, :α₂) ?
                 (β=β, μ=p.μ, ν=p.ν, α=p.α, α₂=p.α₂) :
                 (β=β, μ=p.μ, ν=p.ν, Θ=p.Θ, α=p.α)
        output_func = (sol, i) -> (count(sol[end][1:2:end] .> 1) >= threshold, false)
        sols = solve_ensemble(x₀, nd, x₀_gen!, p_curr, Tₑ, Nb, output_func, false)
        return count(sols) / Nb - 0.5
    end

    return find_zero(F, (0.0, 1.0); xrtol=1e-4)
end
