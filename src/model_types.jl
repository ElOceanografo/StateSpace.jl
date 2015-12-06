
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

    # Observation matrix, feed-forward matrix, and noise covariance
    G::Function
    D::Function
    W::Function

    # Model dimensions
    nx::Int
    ny::Int
    nu::Int

    function LinearGaussianSSM(F::Function, B::Function, V::Function,
                                G::Function, D::Function, W::Function)
        @assert ispossemidef(V(1))
        @assert ispossemidef(W(1))

        nx, ny, nu = confirm_matrix_sizes(F(1), B(1), V(1), G(1), D(1), W(1))
        new(F, B, V, G, D, W, nx, ny, nu)
    end
end

# Time-dependent constructor
LinearGaussianSSM{T}(F::Function, V::Function, G::Function, W::Function;
                          B::Function=_->zeros(T, size(V(1), 1), 1),
                          D::Function=_->zeros(T, size(W(1), 1), 1)) =
	  LinearGaussianSSM(F, B, V, G, D, W)

# Time-independent constructor
LinearGaussianSSM{T}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T}, W::Matrix{T};
                          B::Matrix{T}=zeros(T, size(F, 1), 1),
                          D::Matrix{T}=zeros(T, size(G, 1), 1)) =
	  LinearGaussianSSM(_->F, _->B, _->V, _->G, _->D, _->W)

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
    println("\n\nFeed-forward matrix D:")
    show(mod.D(1))
    println("\n\nObseration error covariance W:")
    show(mod.W(1))
end

# Core methods
process_matrix(m::LinearGaussianSSM, state, t) = m.F(t)
process_matrix(m::LinearGaussianSSM, state) = m.F(1)

observation_matrix(m::LinearGaussianSSM, state, t) = m.G(t)
observation_matrix(m::LinearGaussianSSM, state) = m.G(1)

control_matrix(m::LinearGaussianSSM, t) = m.B(t)
control_matrix(m::LinearGaussianSSM) = m.B(1)

###########################################################################
# Noninear Gaussian 
###########################################################################


immutable NonlinearGaussianSSM{T} <: AbstractGaussianSSM
    f::Function # actual process function
    fjac::Function # function returning Jacobian of process
    V::Matrix{T}
    g::Function # actual observation function
    gjac::Function # function returning Jacobian of process
    W::Matrix{T}
end

function NonlinearGaussianSSM{T}(f::Function, V::Matrix{T}, 
        g::Function, W::Matrix{T})
    fjac = jacobian(f)
    gjac = jacobian(g)
    return NonlinearGaussianSSM{T}(f, fjac, V, g, gjac, W)
end


## Core methods
process_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal) = m.fjac(mean(x))
process_matrix(m::NonlinearGaussianSSM, x::Vector) = m.fjac(x)
observation_matrix(m::NonlinearGaussianSSM, x::AbstractMvNormal) = m.gjac(mean(x))
observation_matrix(m::NonlinearGaussianSSM, x::Vector) = m.gjac(x)

