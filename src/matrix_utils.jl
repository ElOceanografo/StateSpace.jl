
# Methods for checking matrix sizes etc.

issquare(x::Matrix) = size(x, 1) == size(x, 2)
ispossemidef(x::Matrix) = issymmetric(x) && (eigmin(x) >= 0)


function confirm_matrix_sizes(F, B, V, G, W)

    nx = size(F, 1)
    nu = size(B, 2)
    ny = size(G, 1)

    @assert size(F) == (nx, nx)
    @assert size(B) == (nx, nu)
    @assert size(V) == (nx, nx)

    @assert size(G) == (ny, nx)
    @assert size(W) == (ny, ny)

    return nx, ny, nu

end #confirm_matrix_sizes

# function confirm_matrix_sizes(A1, A2, A3, B1, B2, G, Q, C1, C2, C3, D1, D2, H, R, x1, P1)

#     @assert size(B2, 2) == size(D2, 2)

#     nx = size(A1, 1)
#     ny = size(C1, 1)
#     nu = size(B2, 2)

#     na1, na2  = size(A1, 2), size(A2, 2)
#     nb        = size(B1, 2)
#     nc1, nc2  = size(C1, 2), size(C2, 2)
#     nd        = size(D1, 2)

#     nq = size(Q, 1)
#     nr = size(R, 1)

#     @assert size(A1) == (nx, na1)
#     @assert size(A2) == (na1, na2)
#     @assert size(A3) == (na2, nx)

#     @assert size(B1) == (nx, nb)
#     @assert size(B2) == (nb, nu)

#     @assert size(G)  == (nx, nq)
#     @assert size(Q)  == (nq, nq)

#     @assert size(C1) == (ny, nc1)
#     @assert size(C2) == (nc1, nc2)
#     @assert size(C3) == (nc2, nx)

#     @assert size(D1) == (ny, nd)
#     @assert size(D2) == (nd, nu)

#     @assert size(H)  == (ny, nr)
#     @assert size(R)  == (nr, nr)

#     @assert length(x1)  == nx
#     @assert size(P1)    == (nx, nx)

#     return nx, ny, nu, nq, nr

# end #confirm_matrix_sizes
