
abstract AbstractStateSpaceModel
typealias AbstractSSM AbstractStateSpaceModel
abstract AbstractGaussianSSM <: AbstractStateSpaceModel

###########################################################################
# Linear Gaussian 
###########################################################################

immutable LinearGaussianSSM <: AbstractGaussianSSM
    # Process transition matrix, control matrix, and noise covariance
    F::Function
    B::Function
    V::Function

    # Observation matrix and noise covariance
    G::Function
    W::Function

    # Model dimensions
    nx::Int
    ny::Int
    nu::Int

    function LinearGaussianSSM(F::Function, B::Function, V::Function,
                                G::Function, W::Function)
        @assert ispossemidef(V(1))
        @assert ispossemidef(W(1))

        nx, ny, nu = confirm_matrix_sizes(F(1), B(1), V(1), G(1), W(1))
        new(F, B, V, G, W, nx, ny, nu)
    end
end

# Time-dependent constructor
LinearGaussianSSM{T}(F::Function, V::Function, G::Function, W::Function;
                          B::Function=_->zeros(T, size(V(1), 1), 1)) =
	  LinearGaussianSSM(F, B, V, G, W)

# Time-independent constructor
LinearGaussianSSM{T}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::Matrix{T};
                          B::Matrix{T}=zeros(T, size(F, 1), 1)) =
	  LinearGaussianSSM(_->F, _->B, _->V, _->G, _->W)

function show(io::IO, mod::LinearGaussianSSM)
    dx, dy = mod.nx, mod.ny 
    println("LinearGaussianSSM, $dx-D process x $dy-D observations")
    println("Process evolution matrix F:")
    show(mod.F(1))
    println("\n\nGontrol input matrix B:")
    show(mod.B(1))
    println("\n\nProcess error covariance V:")
    show(mod.V(1))
    println("\n\nObservation matrix G:")
    show(mod.G(1))
    println("\n\nObseration error covariance W:")
    show(mod.W(1))
end

## Core methods
process_matrix(m::LinearGaussianSSM, state::Vector, t::Int=1) = m.F(t)
process_matrix(m::LinearGaussianSSM, state::AbstractMvNormal, t::Int=1) = m.F(t)

observation_matrix(m::LinearGaussianSSM, state::Vector, t::Int=1) = m.G(t)
observation_matrix(m::LinearGaussianSSM, state::AbstractMvNormal, t::Int=1) = m.G(t)

control_input(m::LinearGaussianSSM, u, t::Int=1) = m.B(t) * u

###########################################################################
# Noninear Gaussian 
###########################################################################


immutable NonlinearGaussianSSM <: AbstractGaussianSSM
    # Process transition function, jacobian, and noise covariance matrix
    f::Function
    V::Function

    # Control input function
    b::Function

    # Observation function, 
    g::Function # actual observation function
    W::Function

    nx::Int
    ny::Int
    nu::Int

    function NonlinearGaussianSSM(f::Function, V::Function,
            b::Function, g::Function, W::Function, nx::Int, ny::Int, nu::Int)
        @assert ispossemidef(V(1))
        @assert ispossemidef(W(1))
        @assert (nx, nx) == size(V(1))
        @assert (ny, ny) == size(W(1))

        new(f, V, b, g, W, nx, ny, nu)
    end
end

function NonlinearGaussianSSM{T}(f::Function, V::Matrix{T}, g::Function,
        W::Matrix{T}; b::Function=_ -> 0, nu::Int=0)
    nx = size(V, 1)
    ny = size(W, 1)

    return NonlinearGaussianSSM(f, _-> V, b, g, _-> W, nx, ny, nu)
end


## Core methods
process_matrix(m::NonlinearGaussianSSM, x::Vector, t::Int=1) = jacobian(m.f, x)
function process_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal, t::Int=1)
    return process_matrix(m, mean(x), t)
end

observation_matrix(m::NonlinearGaussianSSM, x::Vector, t::Int=1) = jacobian(m.g, x)
function observation_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal, t::Int=1) 
    return process_matrix(m, mean(x), t)
end

control_input(m::NonlinearGaussianSSM, u, t::Int=1) = m.b(u)