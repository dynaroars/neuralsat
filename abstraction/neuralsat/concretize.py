'''Constrained concretization of a linear bound (neuralsat abstraction).

Bounds ``coef x + off`` over ``{x : ||x - x0||_inf <= eps, h_j(x) <= 0}``
with a projected primal-dual method on the Lagrangian

    L(x, mu) = coef x + off + sum_j mu_j relu(h_j(x)),   mu_j >= 0.

``L`` is convex, so its tangent at any ``x`` under-estimates it and the
tangent's minimum over the ball has a closed form. That value is a sound
bound for every ``x`` and every ``mu >= 0``, so the search may stop anywhere.
'''

import torch

from abstraction.neuralsat.constraint import penalty_terms


def holder_min(g, x0, eps):
    # min over ||x - x0||_inf <= eps of g . x, for every row of g.
    return g @ x0 - (eps * g.abs()).sum(-1)


def lagrangian_grad(s_coef, mu, terms):
    # Subgradient s_coef + sum_j mu_j grad relu(h_j) of the Lagrangian.
    _, grads = terms
    return s_coef + sum(m.unsqueeze(-1) * g for m, g in zip(mu, grads))


def certified_bound(s_coef, s_off, x, x0, eps, mu, terms):
    '''Sound lower bound of ``min_B L(., mu)`` from the tangent of ``L`` at x.
    '''
    values, _ = terms
    grad = lagrangian_grad(s_coef, mu, terms)
    value = (s_coef * x).sum(-1) + s_off
    value = value + sum(m * torch.relu(v) for m, v in zip(mu, values))
    return value + (grad * (x0 - x)).sum(-1) - (eps * grad.abs()).sum(-1)


def best_multiplier(s_coef, grad_pen, slope0, eps):
    '''Exact argmax over ``mu >= 0`` of the concave function
    ``slope0 mu - sum eps |s_coef + mu g|``.

    The objective is concave piecewise linear in ``mu``. Its slope starts at
    ``slope0`` and drops by ``2 eps |g_k|`` at each breakpoint
    ``-s_coef_k / g_k``; the maximum is at the first breakpoint where the
    slope is no longer positive (0 if it never is).
    '''
    crossing = s_coef * grad_pen < 0
    safe_grad = torch.where(crossing, grad_pen, torch.ones_like(grad_pen))
    breakpoints = torch.where(crossing, -s_coef / safe_grad,
                              torch.full_like(s_coef, torch.inf))
    drops = torch.where(crossing, 2 * eps * grad_pen.abs(),
                        torch.zeros_like(s_coef))
    breakpoints, order = breakpoints.sort(-1)
    slopes = slope0.unsqueeze(-1) - drops.gather(-1, order).cumsum(-1)
    hit = (slopes <= 0) & breakpoints.isfinite()
    first = hit.to(torch.uint8).argmax(-1, keepdim=True)
    mu = breakpoints.gather(-1, first).squeeze(-1)
    return torch.where(hit.any(-1) & (slope0 > 0), mu, torch.zeros_like(mu))


def line_search_bounds(s_coef, s_off, x, x0, eps, terms):
    '''Certified bounds at ``x`` with the best multiplier per constraint.

    Every ``mu >= 0`` is sound, so pricing one constraint at a time (others
    at 0) gives valid bounds that need no step-size tuning.
    '''
    values, grads = terms
    zeros = [torch.zeros_like(v) for v in values]
    bounds = []
    for j, (value, grad) in enumerate(zip(values, grads)):
        slope0 = (torch.relu(value) + (grad * (x0 - x)).sum(-1)
                  - (eps * s_coef.sign() * grad).sum(-1))
        mu = list(zeros)
        mu[j] = best_multiplier(s_coef, grad, slope0, eps)
        bounds.append(certified_bound(s_coef, s_off, x, x0, eps, mu, terms))
    return bounds


def barzilai_borwein(eta, step, grad_change, tol):
    # Step size |dx . y| / |y|^2 clipped to (0, 1]; keep eta if y is ~0.
    yy = (grad_change ** 2).sum(-1, keepdim=True)
    xy = (step * grad_change).sum(-1, keepdim=True).abs()
    bb = (xy / yy.clamp(min=tol)).clamp(1e-8, 1.0)
    return torch.where(yy > tol, bb, eta)


def concretize(coef, off, x0, eps, constraints, sign,
               steps=100, lr=0.1, tau=0.1, tol=1e-6, patience=10):
    '''Bound ``coef x + off`` row-wise over the feasible set.

    Returns a lower bound for ``sign=+1`` and an upper bound for ``sign=-1``.
    ``lr`` and ``tau`` are the initial primal step and the dual step;
    ``patience`` iterations without ``tol`` gain stop the search.
    '''
    s_coef, s_off = sign * coef, sign * off
    best = holder_min(s_coef, x0, eps) + s_off  # mu = 0, sound alone
    if constraints:
        lower, upper = x0 - eps, x0 + eps
        x = torch.where(s_coef >= 0, lower, upper)  # minimizer of s_coef x
        mu = [s_coef.norm(dim=-1) for _ in constraints]
        terms = penalty_terms(x, constraints)
        eta = torch.full_like(best, lr).unsqueeze(-1)
        stall = 0
        for _ in range(steps):
            candidates = [
                certified_bound(s_coef, s_off, x, x0, eps, mu, terms)]
            candidates += line_search_bounds(s_coef, s_off, x, x0, eps, terms)
            gain = torch.stack(candidates).max(0).values
            stall = stall + 1 if (gain - best).max() <= tol else 0
            best = torch.maximum(best, gain)
            if stall >= patience:
                break
            grad = lagrangian_grad(s_coef, mu, terms)
            x_new = (x - eta * grad).clamp(lower, upper)  # project on ball
            terms_new = penalty_terms(x_new, constraints)
            grad_new = lagrangian_grad(s_coef, mu, terms_new)
            mu = [torch.relu(m + tau * h)
                  for m, h in zip(mu, terms_new[0])]  # dual ascent
            eta = barzilai_borwein(eta, x_new - x, grad_new - grad, tol)
            x, terms = x_new, terms_new
    return sign * best
