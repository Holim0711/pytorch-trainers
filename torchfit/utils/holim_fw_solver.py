import torch


## this should be changed
def _min_norm(vv, vṽ, ṽṽ):
    if vṽ >= vv:
        return 0.999
    if vṽ >= ṽṽ:
        return 0.001
    return ((ṽṽ - vṽ) / (vv + ṽṽ - 2 * vṽ)).item()


def mgda_frank_wolfe_solver(V, max_iter=250, stop_crit=1e-5):
    """ Frank-Wolfe solver for MGDA

    Goal: find the smallest among convex combinations of T vectors.

    V ∈ R(T x D) contains T vectors.
    α ∈ R(T) is the ratio st. Vᵗα is the solution. (∵Σαᵢ = 1 && αᵢ ≥ 0)
    f(α) = |Vᵗα|² is the objective that we want to minimize.

    And also we know...
        ∇f(α) = 2Mα (∵M = VVᵗ)

    FW procedure is like this.
        1. initialize α
        2. find ś minimizing śᵗ∇f(α)
        3. find 𝛾 minimizing f(𝛾α + (1 - 𝛾)ś)
        4. update α ← (𝛾α + (1 - 𝛾)ś)
        5. repeat 2~4 until convergence

    This FW solver is like this.
        1. α = (1/T, 1/T, ...)
        2. ś = eᵢ (∵i = argminᵢ[(Mα)ᵢ])
        3. set v = Vᵗα, ṽ = Vᵢ and find 𝛾
        4. update α ← (𝛾α + (1 - 𝛾)ś)
        5. repeat 2~4 until convergence
    """
    T = len(V)

    if T < 2:
        return torch.ones(T)

    M = V.matmul(V.t()).cpu()

    if T == 2:
        𝛾 = _min_norm(M[0, 0], M[0, 1], M[1, 1])
        return torch.tensor([𝛾, 1 - 𝛾])

    α = torch.ones(T) / T

    for _ in range(max_iter):
        Mα = M.mv(α)
        ś = # min sMa that ss=1

        # this part should be changed
        vv = α.dot(Mα)
        vṽ = α.dot(M[i])
        ṽṽ = M[i, i]
        𝛾 = _min_norm(vv, vṽ, ṽṽ)

        ᾱ = 𝛾 * α
        ᾱ[i] += 1 - 𝛾

        Δα = ᾱ - α
        α = ᾱ

        if Δα.abs().sum() < stop_crit:
            break
    return α
