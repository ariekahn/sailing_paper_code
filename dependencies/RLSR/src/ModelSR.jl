abstract type AbstractSR <: AbstractStateModel end
mutable struct SRModel <: AbstractSR
    M::Matrix{Float64}
    R::Vector{Float64}
    V::Vector{Float64}
    Q::Vector{Float64}
    α::Float64
    αM::Float64
    γ::Float64
    λ::Float64
    trace::Vector{Float64}
    ident::Matrix{Float64}
end
function SRModel(env, α, αM, γ, λ)
    n = length(env)
    # Initialize M by inverting the transition matrix
    M = inv(I(n) - γ * stochastic_matrix(env))
    # Set M to zero for the dummy terminal states
    # Ensures that they won't affect our weight estimates or norming
    view(M, :, env.terminal_states) .= 0
    # Ident is used every step, save ourselves from reallocating
    ident = Matrix{Float64}(I, n, n)
    trace = zeros(n)
    SRModel(M, zeros(n), zeros(n), zeros(n), α, αM, γ, λ, trace, ident)
end
SRModel(env; α, αM, γ, λ)::SRModel = SRModel(env, α, αM, γ, λ)
function model_name(model::M) where {M <: AbstractSR} "SR" end

function SRSoftmax(env; α, αM, γ, λ, β)
    SR = SRModel(env, α, αM, γ, λ)
    policy = PolicySoftmax(β)
    StateAgent(env, SR, policy)
end

function SRTwoStepSoftmax(env; α, αM, γ, λ, β1, β2)
    SR = SRModel(env, α, αM, γ, λ)
    policy = PolicyTwoStepSoftmax(β1, β2)
    StateAgent(env, SR, policy)
end

function SR_ϵ_Greedy(env; α, αM, γ, λ, ϵ)
    SR = SRModel(env, α, αM, γ, λ)
    policy = Policy_ϵ_Greedy(ϵ)
    StateAgent(env, SR, policy)
end

function SRGreedy(env; α, αM, γ, λ)
    SR = SRModel(env, α, αM, γ, λ)
    policy = PolicyGreedy()
    StateAgent(env, SR, policy)
end

"""SR Non-Trace Update:

After a transition s → s′,

1. δ = R(s, a) + γV(s′) - V(s)
    - Error term
2. M[s, :] = (1 - αM) * M[s, :] + αM * (Iₛ + γ * M[s′, :])
    - M[s] should approach Iₛ plus the discounted M of the successor state
3. w(i) ← w(i) + αw δ M(s,i)
    - Each state that s could transition to (that s has as a feature) has its weight adjusted

- To maintain consistency, the M update should occur /before/ the w update
- Need to ensure that V is updated after both M and w updates


SR Trace Update:

After a transition s → s′,

1. trace = I_s + λ*γ*trace 
    - Discount the existing trace, and add 1 the current state
2. δM = (Iₛ + γ M_s') - (M_s)
    Observed state transition, versus prior prediction
3. For each state, M_sx = M_sx + α * trace[sx] * δM

NB should verify what trace M updates should look like?


"""
function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: AbstractSR, P}
    agent.model.trace .= 0
end

function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: AbstractSR, P}
    SR = agent.model

    # Update eligibility trace
    SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
    SR.trace[s] = SR.trace[s] + 1

    # Update M first
    # δM is our state misprediction: Iₛ + γ M_s' vs. M_s
    # Move each state's prediction in the direction of δM based on the trace
    δM = SR.ident[s, :] + SR.γ .* SR.M[s′, :] - SR.M[s, :]
    # The Iₛ cancels out, leaving us with γM_s′.
    # M_s should already be an average of γM_s′1, γM_s′2, etc.
    for sx in 1:length(SR)
        SR.M[sx, :] = SR.M[sx, :] + (SR.αM * SR.trace[sx]) .* δM
    end

    # Update V
    SR.V[:] = SR.M * SR.R

    # Update Q to include the discount
    # This isn't technically correct, since we're ignoring reward from the current state,
    # but as long as all rewards are terminal, it should be fine
    SR.Q[:] = SR.γ * SR.V
end

function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: AbstractSR, P}
    SR = agent.model

    # Update eligibility trace
    SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
    SR.trace[s] = SR.trace[s] + 1

    # Update M first
    # δM is our state misprediction: (Iₛ + γ M_s') vs. (M_s)
    # Move each state's prediction in the direction of δM based on the trace
    δM = SR.ident[s, :] + SR.γ .* SR.M[s′, :] - SR.M[s, :]
    # The Iₛ cancels out, leaving us with γM_s′.
    # M_s should already be an average of γM_s′1, γM_s′2, etc.
    for sx in 1:length(SR)
        SR.M[sx, :] = SR.M[sx, :] + (SR.αM * SR.trace[sx]) .* δM
    end

    # Update R - for proper discounting w/ the dummy terminal states,
    # we're updating the step before
    SR.R[s] = (SR.α * reward) + ((1 - SR.α) * SR.R[s])

    # Update V
    SR.V[:] = SR.M * SR.R

    # Update Q to include the discount
    # This isn't technically correct, since we're ignoring reward from the current state,
    # but as long as all rewards are terminal, it should be fine
    SR.Q[:] = SR.γ * SR.V
end

function update_model_end!(::StateAgent{E, M, P}, ::Episode) where {E, M <: AbstractSR, P} end

# Snapshot code
struct SRModelSnapshot <: AbstractModelSnapshot
    V::Vector{Float64}
    M::Matrix{Float64}
end
function SRModelSnapshot(model::SRModel)
    SRModelSnapshot(copy(model.V), copy(model.M))
end
mutable struct SRModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    M::Array{Float64, 3}
    n::Int
end
function SRModelRecord(agent::StateAgent{E,M,P}, maxsize::Int)::SRModelRecord where {E <: AbstractEnv, M <: SRModel, P <: AbstractPolicy}
    SRModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env), length(agent.env)),
        0)
end
Base.firstindex(record::SRModelRecord) = 1
Base.lastindex(record::SRModelRecord) = length(record)
Base.length(record::SRModelRecord) = record.n
function Base.push!(record::SRModelRecord, model::SRModel)
    record.n += 1
    (sx, sy) = size(record.V)
    if record.n > sx
        new_V = zeros(sx * 2, sy)
        view(new_V, 1:sx, :) .= record.V
        record.V = new_V

        new_M = zeros(sx * 2, sy, sy)
        new_M[1:sx, :, :] .= record.M
        record.M = new_M
    end
    record.V[record.n, :] = model.V[:]
    record.M[record.n, :, :] = model.M[:, :]
end
function Base.iterate(record::SRModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (SRModelSnapshot(record.V[state, :], record.M[state, :, :]), state+1)
    end
end
function Base.getindex(record::SRModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    SRModelSnapshot(record.V[i, :], record.M[i, :, :])
end
Base.getindex(record::SRModelRecord, I) = SRModelRecord(record.env, record.policy, record.V[I, :], record.M[I, I, :], length(I))

function Record(agent::StateAgent{E, M, P}, maxsize::Int)::SRModelRecord where {E, M <: SRModel, P}
    SRModelRecord(agent, maxsize)
end