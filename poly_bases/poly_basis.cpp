#include "poly_basis.hpp"

template <typename DerivedBasis, int dim>
poly_space_basis<DerivedBasis, dim>::poly_space_basis() : ready_2b_used(false)
{
}

/*
 * Constructor 1: For modal and arbitrary nodal
 */
template <typename DerivedBasis, int dim>
poly_space_basis<DerivedBasis, dim>::poly_space_basis(
  const std::vector<dealii::Point<dim> > &integration_points_,
  const std::vector<dealii::Point<1, double> > &support_points_,
  const int &domain_)
  : n_polys(pow(support_points_.size(), dim)),
    n_quads(integration_points_.size()),
    bases_grads_at_quads(n_polys, integration_points_.size()),
    basis_at_quads(integration_points_.size(), n_polys)
{
  poly_basis = std::move(DerivedBasis(support_points_, domain_));
  ready_2b_used = true;
  compute_quad_values(integration_points_);
}

/*
 * Constructor 2: For modal and LGL nodal
 */
template <typename DerivedBasis, int dim>
poly_space_basis<DerivedBasis, dim>::poly_space_basis(
  const std::vector<dealii::Point<dim> > &integration_points_,
  const unsigned &max_poly_order,
  const int &domain_)
  : n_polys(pow(max_poly_order + 1, dim)),
    n_quads(integration_points_.size()),
    bases_grads_at_quads(n_polys, integration_points_.size()),
    basis_at_quads(integration_points_.size(), n_polys)
{
  if (std::is_same<DerivedBasis, JacobiPolys<dim> >::value)
  {
    poly_basis = std::move(DerivedBasis(max_poly_order, domain_));
  }
  else if (std::is_same<DerivedBasis, LagrangePolys<dim> >::value)
  {
    const dealii::QGaussLobatto<1> LGL_quad_1D(max_poly_order + 1);
    poly_basis = std::move(DerivedBasis(LGL_quad_1D.get_points(), domain_));
  }
  ready_2b_used = true;
  compute_quad_values(integration_points_);
}

/*
 * Move constructor
 */
template <typename DerivedBasis, int dim>
poly_space_basis<DerivedBasis, dim>::poly_space_basis(
  poly_space_basis<DerivedBasis, dim> &&input) noexcept
  : n_polys(input.n_polys),
    n_quads(input.n_quads),
    bases_grads_at_quads(std::move(input.bases_grads_at_quads)),
    ready_2b_used(input.ready_2b_used),
    poly_basis(std::move(input.poly_basis)),
    basis_at_quads(std::move(input.basis_at_quads))
{
}

/*
 * Move assignment operator
 */
template <typename DerivedBasis, int dim>
poly_space_basis<DerivedBasis, dim> &poly_space_basis<DerivedBasis, dim>::
operator=(poly_space_basis<DerivedBasis, dim> &&input) noexcept
{
  n_polys = input.n_polys;
  n_quads = input.n_quads;
  basis_at_quads = std::move(input.basis_at_quads);
  bases_grads_at_quads = std::move(input.bases_grads_at_quads);
  ready_2b_used = input.ready_2b_used;
  poly_basis = std::move(input.poly_basis);
  return *this;
}

template <typename DerivedBasis, int dim>
void poly_space_basis<DerivedBasis, dim>::compute_quad_values(
  const std::vector<dealii::Point<dim> > &integration_points_)
{
  assert(ready_2b_used);
  for (unsigned i1 = 0; i1 < integration_points_.size(); ++i1)
  {
    dealii::Point<dim, double> p0 = integration_points_[i1];
    std::vector<double> Ni = value(p0);
    std::vector<dealii::Tensor<1, dim> > Ni_grad = grad(p0);
    bases_grads_at_quads[mtl::iall][i1] =
      mtl::dense_vector<dealii::Tensor<1, dim> >(Ni_grad);
    for (unsigned i_poly = 0; i_poly < Ni.size(); ++i_poly)
      basis_at_quads(i1, i_poly) = Ni[i_poly];
  }
}

template <typename Derived_Basis, int dim>
std::vector<double> poly_space_basis<Derived_Basis, dim>::value(
  const dealii::Point<dim, double> &P0, const unsigned half_range)
{
  assert(ready_2b_used);
  return poly_basis.value(P0, half_range);
}

template <typename Derived_Basis, int dim>
eigen3mat poly_space_basis<Derived_Basis, dim>::get_func_vals_at_iquad(
  const unsigned i_quad) const
{
  return basis_at_quads.block(i_quad, 0, 1, n_polys);
}

template <typename Derived_Basis, int dim>
eigen3mat poly_space_basis<Derived_Basis, dim>::get_dof_vals_at_quads(
  const eigen3mat &in_dof) const
{
  return basis_at_quads * in_dof;
}

template <typename Derived_Basis, int dim>
std::vector<dealii::Tensor<1, dim> >
poly_space_basis<Derived_Basis, dim>::grad(const dealii::Point<dim, double> &P0)
{
  assert(ready_2b_used);
  return poly_basis.grad(P0);
}

template <typename Derived_Basis, int dim>
poly_space_basis<Derived_Basis, dim>::~poly_space_basis()
{
}
