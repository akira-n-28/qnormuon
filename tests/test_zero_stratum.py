"""Isolated regular-quotient and zero-stratum research checks."""

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.rank_deficient import partial_polar
from experiments.zero_stratum import (
    balanced_birth, balanced_factors, birth_lmo, factor_metric, horizontal_lift,
    horizontal_projection, quotient_gradient, quotient_metric,
    quotient_unit_descent, stable_norm, tangent_map, tangent_projection,
)
from qnormuon import canonical_pair


DT = torch.float64


def rand(seed, *shape):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


def close(a, b, tol=3e-11):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def frame(n=4):
    a, b = rand(501, n), rand(502, n)
    return a / a.norm(), b / b.norm()


@pytest.mark.parametrize("c", [1e-6, 3., -2., -1e6])
def test_orbit_fiber_balanced_svd_and_positive_double_cover(c):
    u, d = rand(503, 4), rand(504, 4)
    x = torch.outer(d, u)
    u2, d2 = c * u, d / c
    close(torch.outer(d2, u2), x)
    recovered_c = (u2 @ u) / u.square().sum()
    close(u2 / recovered_c, u)
    close(d2 * recovered_c, d)
    ub, db = balanced_factors(u, d)
    ub2, db2 = balanced_factors(u2, d2)
    close(ub2, math.copysign(1., c) * ub)
    close(db2, math.copysign(1., c) * db)
    close(ub.norm(), db.norm())
    close(torch.outer(db, ub), x)
    left, singular, right = torch.linalg.svd(x, full_matrices=False)
    close(singular[0], u.norm() * d.norm())
    sign = torch.sign(right[0] @ ub)
    close(ub, sign * singular[0].sqrt() * right[0])
    close(db, sign * singular[0].sqrt() * left[:, 0])
    pu, pd = canonical_pair(u[None], d[:, None])
    close(ub, pu[0])
    close(db, pd[:, 0])


def test_balanced_sign_cover_has_loop_monodromy():
    ts = torch.linspace(0., math.pi, 101, dtype=DT)
    factors = torch.stack((ts.cos(), ts.sin()), dim=1)
    xs = torch.einsum("bi,bj->bij", factors, factors)
    close(xs[0], xs[-1])
    close(factors[0], -factors[-1])
    assert (factors[1:] - factors[:-1]).norm(dim=1).max() < 0.04
    # A lift followed continuously around this closed matrix loop changes sheet.
    assert (factors[-1] - factors[0]).norm() > 1.9


@pytest.mark.parametrize("kind", ["both", "up", "down"])
def test_zero_factor_types_have_same_swiglu_function_but_different_gradients(kind):
    n, batch = 3, 17
    inputs, gate, target = rand(510, batch, n), rand(511, n), rand(512, batch, n)
    u = (torch.zeros(n, dtype=DT) if kind in ("both", "up") else rand(513, n)).requires_grad_()
    d = (torch.zeros(n, dtype=DT) if kind in ("both", "down") else rand(514, n)).requires_grad_()
    activation = F.silu(inputs @ gate)
    output = activation[:, None] * (inputs @ u)[:, None] * d
    gu, gd = torch.autograd.grad(F.mse_loss(output, target), (u, d))
    x = torch.zeros(n, n, dtype=DT, requires_grad=True)
    y = activation[:, None] * (inputs @ x.T)
    g = torch.autograd.grad(F.mse_loss(y, target), x)[0]
    close(output, y)
    close(gu, g.T @ d)
    close(gd, g @ u)
    close(g, (2 / (batch * n)) * (activation[:, None] * (-target)).T @ inputs)
    assert g.norm() > 0.1
    if kind == "both":
        assert gu.norm() == 0 and gd.norm() == 0
    else:
        assert gu.norm() + gd.norm() > 0.01


def test_factor_gradients_do_not_determine_function_space_birth_gradient():
    g1, g2 = torch.diag(torch.tensor([1., 0.], dtype=DT)), torch.diag(torch.tensor([1., 2.], dtype=DT))
    u, d = torch.zeros(2, dtype=DT), torch.tensor([1., 0.], dtype=DT)
    close(g1.T @ d, g2.T @ d)
    close(g1 @ u, g2 @ u)
    z1 = birth_lmo(g1, gap_tolerance=1e-12).direction
    z2 = birth_lmo(g2, gap_tolerance=1e-12).direction
    assert (z1 - z2).norm() > 1.4


def test_regular_jacobian_kernel_and_singular_zero_images():
    n = 3
    u, d = rand(520, n), rand(521, n)

    def jacobian(u, d):
        return torch.autograd.functional.jacobian(
            lambda factors: torch.outer(factors[n:], factors[:n]).reshape(-1),
            torch.cat((u, d)))

    j = jacobian(u, d)
    assert torch.linalg.matrix_rank(j) == 2 * n - 1
    close(j @ torch.cat((u, -d)), torch.zeros(n*n, dtype=DT))
    du, dd = rand(522, n), rand(523, n)
    close((j @ torch.cat((du, dd))).reshape(n, n), tangent_map(u, d, du, dd))
    zero = torch.zeros(n, dtype=DT)
    assert torch.linalg.matrix_rank(jacobian(zero, d)) == n
    assert torch.linalg.matrix_rank(jacobian(u, zero)) == n
    assert torch.count_nonzero(jacobian(zero, zero)) == 0
    # More kernel directions than gauge alone at one-sided zeros.
    close(tangent_map(zero, d, zero, du), torch.zeros(n, n, dtype=DT))


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 3e-11), (torch.float32, 3e-5)])
def test_balanced_horizontal_lift_is_pseudoinverse_and_minimum_norm(dtype, tol):
    n, sigma = 4, 2.3
    a, b = [v.to(dtype) for v in frame(n)]
    u, d = math.sqrt(sigma) * b, math.sqrt(sigma) * a
    z = tangent_projection(a, b, rand(530, n, n).to(dtype))
    du, dd = horizontal_lift(u, d, z)
    close(tangent_map(u, d, du, dd), z, tol)
    close(u @ du, d @ dd, tol)
    j = torch.autograd.functional.jacobian(
        lambda factors: torch.outer(factors[n:], factors[:n]).reshape(-1), torch.cat((u, d)))
    inverse = torch.linalg.pinv(j, rtol=1e-5 if dtype == torch.float32 else 1e-12)
    close(torch.cat((du, dd)), inverse @ z.reshape(-1), tol)
    for t in (-2., -.1, .5, 3.):
        other = torch.cat((du + t * u, dd - t * d))
        close(other.square().sum(), torch.cat((du, dd)).square().sum() + 2 * sigma * t*t, tol)
    expected = torch.cat((torch.tensor([math.sqrt(2*sigma)], dtype=dtype),
                          torch.full((2*n-2,), math.sqrt(sigma), dtype=dtype),
                          torch.zeros(1, dtype=dtype)))
    close(torch.linalg.svdvals(j), expected, tol)
    error = (tangent_map(u, d, du, dd)-z).norm().item()
    print(f"horizontal {dtype} n=4 sigma=2.3: reconstruction Frobenius error={error:.3e}")


@pytest.mark.parametrize("c", [1e-6, 1e6, -7.])
def test_H_metric_lift_and_horizontal_projection_are_gauge_equivariant(c):
    u, d = rand(540, 4), rand(541, 4)
    a, b = d / d.norm(), u / u.norm()
    z = tangent_projection(a, b, rand(542, 4, 4))
    du, dd = horizontal_lift(u, d, z)
    du2, dd2 = horizontal_lift(c * u, d / c, z)
    close(du2 / c, du)
    close(dd2 * c, dd)
    close(factor_metric(u, d, du, dd), factor_metric(c*u, d/c, du2, dd2))
    vu, vd = rand(543, 4), rand(544, 4)
    hu, hd = horizontal_projection(u, d, vu, vd)
    close(tangent_map(u, d, hu, hd), tangent_map(u, d, vu, vd))
    h = u.norm() / d.norm()
    close(u @ hu / h - h * (d @ hd), torch.tensor(0., dtype=DT))
    hu2, hd2 = horizontal_projection(c*u, d/c, c*vu, vd/c)
    close(hu2/c, hu)
    close(hd2*c, hd)


def test_balanced_section_derivative_is_horizontal_projection_not_frozen_frame():
    u, d, vu, vd = [rand(550+i, 3) for i in range(4)]
    ub, db = balanced_factors(u, d)
    _, derivative = torch.autograd.functional.jvp(balanced_factors, (u, d), (vu, vd))
    h = u.norm() / d.norm()
    frozen_u, frozen_d = vu / h.sqrt(), vd * h.sqrt()
    expected_u, expected_d = horizontal_projection(ub, db, frozen_u, frozen_d)
    close(derivative[0], expected_u)
    close(derivative[1], expected_d)
    assert (derivative[0] - frozen_u).norm() > 0.01


def test_quotient_metric_gradient_and_exact_unit_ball_LMO():
    u, d = rand(560, 3), rand(561, 3)
    a, b = d/d.norm(), u/u.norm()
    g = rand(562, 3, 3)
    grad = quotient_gradient(u, d, g)
    gu, gd = horizontal_lift(u, d, grad)
    z = tangent_projection(a, b, rand(563, 3, 3))
    zu, zd = horizontal_lift(u, d, z)
    h = u.norm()/d.norm()
    close((gu @ zu)/h + h*(gd @ zd), (g*z).sum())
    direction = quotient_unit_descent(u, d, g)
    close(quotient_metric(u, d, direction), torch.tensor(1., dtype=DT))
    close((g*direction).sum(), -quotient_metric(u, d, grad).sqrt())
    for seed in range(10):
        candidate = tangent_projection(a, b, rand(570+seed, 3, 3))
        candidate /= quotient_metric(u, d, candidate).sqrt()
        assert (g*direction).sum() <= (g*candidate).sum() + 3e-11
    # Quotient geometry is not the ambient Frobenius matrix geometry.
    assert (grad - tangent_projection(a, b, g)).norm() > 0.1


def test_separate_polar_updates_need_not_be_horizontal_and_projection_changes_spectrum():
    m, n = 7, 3
    up, dt = rand(580, m, n), rand(581, m, n)
    pairs = [balanced_factors(u, d) for u, d in zip(up, dt)]
    up, dt = torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
    gs = rand(582, m, n, n)
    gu = torch.einsum("mji,mj->mi", gs, dt)
    gd = torch.einsum("mij,mj->mi", gs, up)
    close((up*gu).sum(1), (dt*gd).sum(1))
    pu, pd = partial_polar(gu, rank=n), partial_polar(gd, rank=n)
    residual = (up*pu).sum(1) - (dt*pd).sum(1)
    assert residual.abs().max() > 0.01
    projected = [horizontal_projection(u,d,vu,vd) for u,d,vu,vd in zip(up,dt,pu,pd)]
    hu, hd = torch.stack([v[0] for v in projected]), torch.stack([v[1] for v in projected])
    close((up*hu).sum(1), (dt*hd).sum(1))
    defect = (hu.T @ hu - torch.eye(n, dtype=DT)).norm().item()
    assert defect > 0.01
    print(f"separate polar float64 7x3: max horizontal residual={residual.abs().max().item():.6f}, "
          f"projected up Stiefel defect={defect:.6f}")


def test_signed_gauge_requires_covariant_canonical_state():
    a = rand(590, 7, 3)
    signs = torch.tensor([1., -1., 1., -1., -1., 1., 1.], dtype=DT)
    p = partial_polar(a, rank=3)
    close(partial_polar(signs[:, None]*a, rank=3), signs[:, None]*p)
    assert (p - signs[:, None]*p).norm() > 1
    # Same X does not justify leaving signed-sheet canonical momentum unchanged.


def test_partial_polar_preserves_zero_momentum_rows_instead_of_inventing_birth():
    momentum = torch.zeros(5, 3, dtype=DT)
    momentum[1:, :2] = rand(595, 4, 2)
    p = partial_polar(momentum, rank=2)
    close(p[0], torch.zeros(3, dtype=DT))
    # A nonzero matrix-space birth gradient can coexist with zero factor gradients.
    g = torch.diag(torch.tensor([3., 1., 0.], dtype=DT))
    assert birth_lmo(g, gap_tolerance=1e-12).direction.norm() > 0.9


def test_vertical_changes_leave_first_order_X_fixed_but_change_finite_step():
    u, d = rand(600, 3), rand(601, 3)
    eta = 0.1
    close(tangent_map(u,d,u,-d), torch.zeros(3,3,dtype=DT))
    close(torch.outer(d-eta*d, u+eta*u), (1-eta*eta)*torch.outer(d,u))


def test_horizontal_lift_blowup_and_vanishing_quotient_unit_direction():
    e = torch.eye(2, dtype=DT)
    z = torch.outer(e[0], e[0])
    norms = []
    for sigma in (1., 1e-4, 1e-8, 1e-12):
        u = d = math.sqrt(sigma)*e[0]
        du, dd = horizontal_lift(u,d,z)
        norm = torch.cat((du,dd)).norm().item()
        norms.append(norm)
        assert norm == pytest.approx(1/math.sqrt(2*sigma))
        velocity = quotient_unit_descent(u,d,z)
        assert velocity.norm().item() == pytest.approx(math.sqrt(2*sigma))
        close(quotient_gradient(u,d,z), 2*sigma*z)
    print(f"radial lift float64 n=2 sigma=1,1e-4,1e-8,1e-12: norms={norms}")


def test_zero_cone_not_vector_space_and_regular_tangent_intersection_zero():
    eye = torch.eye(2, dtype=DT)
    z1, z2 = torch.outer(eye[0],eye[0]), torch.outer(eye[1],eye[1])
    assert torch.linalg.matrix_rank(z1) == torch.linalg.matrix_rank(z2) == 1
    assert torch.linalg.matrix_rank(z1+z2) == 2
    normals = []
    for a in eye:
        for b in eye:
            # Every X=t a b^T tends to zero with its tangent plane unchanged.
            operator = torch.autograd.functional.jacobian(
                lambda v: (v.reshape(2,2)-tangent_projection(a,b,v.reshape(2,2))).reshape(-1),
                torch.zeros(4,dtype=DT))
            normals.append(operator)
    assert torch.linalg.matrix_rank(torch.cat(normals)) == 4


@pytest.mark.parametrize("seed", [610, 611, 612])
def test_birth_cone_LMO_and_square_root_factors(seed):
    g = rand(seed,4,4)
    result = birth_lmo(g, gap_tolerance=1e-12)
    z = result.direction
    assert result.separated_at_tolerance
    close(z.norm(), torch.tensor(1.,dtype=DT))
    close((g*z).sum(), torch.tensor(-result.largest_singular_value,dtype=DT))
    assert torch.linalg.matrix_rank(z) == 1
    for sample in range(10):
        a,b = rand(seed+20+sample,4),rand(seed+40+sample,4)
        candidate = torch.outer(a/a.norm(), b/b.norm())
        assert (g*z).sum() <= (g*candidate).sum()+3e-11
    for eta in (1e-2,1e-6,1e-12):
        u,d = balanced_birth(z,step=eta)
        close(torch.outer(d,u)/eta,z)
        close(u.norm(),d.norm())
        assert torch.cat((u,d)).norm().item() == pytest.approx(math.sqrt(2*eta))
    print(f"birth float64 4x4 seed={seed}: linear optimum={-result.largest_singular_value:.6f}, "
          f"spectral gap={result.gap:.6f}")


def test_birth_ties_and_zero_gradient_preclude_global_continuity():
    g = torch.eye(2,dtype=DT)
    tied = birth_lmo(g,gap_tolerance=1e-12)
    assert not tied.separated_at_tolerance
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=DT)
    close(rotation @ g @ rotation.T, g)
    assert (tied.direction - rotation @ tied.direction @ rotation.T).norm() > 1.4
    eps = 1e-10
    g1 = torch.diag(torch.tensor([1+eps,1.],dtype=DT))
    g2 = torch.diag(torch.tensor([1.,1+eps],dtype=DT))
    z1,z2 = (birth_lmo(v,gap_tolerance=1e-12).direction for v in (g1,g2))
    assert (z1-z2).norm() > 1.4
    zero = birth_lmo(torch.zeros_like(g),gap_tolerance=1e-12)
    assert zero.direction.norm() == 0
    for t in (1.,1e-6,1e-12):
        close(birth_lmo(t*g1,gap_tolerance=0).direction,z1)
    close(birth_lmo(-g1,gap_tolerance=0).direction,-z1)


def test_origin_factor_hessian_contains_birth_information_despite_zero_gradient():
    n=3
    g=rand(620,n,n)
    f=lambda v: (g*torch.outer(v[n:],v[:n])).sum()
    zero=torch.zeros(2*n,dtype=DT,requires_grad=True)
    assert torch.autograd.grad(f(zero),zero)[0].norm()==0
    hessian=torch.autograd.functional.hessian(f,zero)
    expected=torch.cat((torch.cat((torch.zeros_like(g),g.T),1),
                        torch.cat((g,torch.zeros_like(g)),1)),0)
    close(hessian,expected)
    s=torch.linalg.svdvals(g)
    close(torch.linalg.eigvalsh(hessian),torch.cat((-s,s.flip(0))))
    u,d=balanced_birth(birth_lmo(g,gap_tolerance=1e-12).direction,step=0.5)
    v=torch.cat((u,d))
    close(v @ hessian @ v,-s[0])


def test_equivariant_one_sided_birth_is_singular_and_depends_on_hidden_factor():
    g=torch.diag(torch.tensor([1.,2.],dtype=DT))
    zero=torch.zeros(2,dtype=DT)
    for t in (1.,1e-3,1e-6):
        d=torch.tensor([t,0.],dtype=DT)
        du=-(g.T @ d)/d.square().sum()
        assert du.norm().item()==pytest.approx(1/t)
        close(tangent_map(zero,d,du,zero),torch.diag(torch.tensor([-1.,0.],dtype=DT)))
    d=torch.tensor([0.,1.],dtype=DT)
    du=-(g.T @ d)/d.square().sum()
    close(tangent_map(zero,d,du,zero),torch.diag(torch.tensor([0.,-2.],dtype=DT)))


def test_no_continuous_regular_ambient_LMO_extension_can_match_birth():
    eye=torch.eye(2,dtype=DT)
    g=torch.outer(eye[0],eye[0])
    birth=birth_lmo(g,gap_tolerance=1e-12).direction
    for t in (1.,1e-6,1e-12):
        first=tangent_projection(eye[0],eye[0],g)
        second=tangent_projection(eye[1],eye[1],g)
        close(-first/first.norm(),birth)
        assert second.norm()==0
        # Birth is not even tangent along the second ray, regardless of tie-break.
        assert (birth-tangent_projection(eye[1],eye[1],birth)).norm()==1


def test_continuous_discrete_rank_one_proximal_birth_is_not_a_tangent_vector_field():
    g = torch.diag(torch.tensor([3., 1.], dtype=DT))
    eta = 0.2

    def proximal(x):
        left, singular, right_t = torch.linalg.svd(x-eta*g, full_matrices=False)
        return singular[0]*torch.outer(left[:,0],right_t[0])

    at_zero = proximal(torch.zeros_like(g))
    close(at_zero, torch.diag(torch.tensor([-0.6, 0.], dtype=DT)))
    a, b = frame(2)
    errors = []
    for t in (1e-3, 1e-6, 1e-9):
        errors.append((proximal(t*torch.outer(a,b))-at_zero).norm().item())
    assert errors[-1] < 2e-9 and errors[0] > 1000*errors[-1]
    e2 = torch.tensor([0.,1.],dtype=DT)
    x = 1e-6*torch.outer(e2,e2)
    delta = proximal(x)-x
    assert (delta-tangent_projection(e2,e2,delta)).norm() > 0.5


def test_invariant_amplitude_threshold_differs_from_individual_row_threshold():
    u=torch.tensor([1e-10,0.],dtype=DT)
    d=torch.tensor([1.,0.],dtype=DT)
    c=1e8
    close(stable_norm(u)*stable_norm(d),stable_norm(c*u)*stable_norm(d/c),1e-14)
    threshold=1e-9
    assert stable_norm(u)<threshold and stable_norm(c*u)>threshold
    assert stable_norm(d/c)>threshold
    with pytest.raises(ValueError,match="nonzero"):
        horizontal_lift(torch.zeros_like(u),d,torch.outer(d,u))


def test_horizontal_lift_rejects_nontangent_direction_instead_of_silent_projection():
    e=torch.eye(2,dtype=DT)
    with pytest.raises(ValueError,match="not in"):
        horizontal_lift(e[0],e[0],torch.outer(e[1],e[1]))
