#include "gn_eps_0_beta_0.hpp"
#include "support_classes.hpp"

template <int dim>
q1_func_class<dim, dealii::Tensor<1, dim> > GN_eps_0_beta_0<dim>::q1_func{};
template <int dim>
q2_func_class<dim, double> GN_eps_0_beta_0<dim>::q2_func{};
template <int dim>
zeta_func_class<dim, double> GN_eps_0_beta_0<dim>::zeta_func{};
template <int dim>
grad_zeta_func_class<dim, dealii::Tensor<1, dim> >
  GN_eps_0_beta_0<dim>::grad_zeta_func{};

template <int dim>
solver_options GN_eps_0_beta_0<dim>::required_solver_options()
{
  return solver_options::spd_matrix;
}

template <int dim>
solver_type GN_eps_0_beta_0<dim>::required_solver_type()
{
  return solver_type::implicit_petsc_aij;
}

template <int dim>
unsigned GN_eps_0_beta_0<dim>::get_num_dofs_per_node()
{
  return dim;
}

/*!
 * The move constrcutor of the derived class should call the move
 * constructor of the base class using std::move. Otherwise the copy
 * constructor will be called.
 */
template <int dim>
GN_eps_0_beta_0<dim>::GN_eps_0_beta_0(GN_eps_0_beta_0 &&inp_cell) noexcept
  : GenericCell<dim>(std::move(inp_cell)),
    taus(std::move(inp_cell.taus)),
    model(inp_cell.model)
{
}

template <int dim>
GN_eps_0_beta_0<dim>::GN_eps_0_beta_0(
  typename GenericCell<dim>::dealiiCell &inp_cell,
  const unsigned &id_num_,
  const unsigned &poly_order_,
  hdg_model<dim, GN_eps_0_beta_0> *model_)
  : GenericCell<dim>(inp_cell, id_num_, poly_order_),
    taus(this->n_faces, -10.0),
    model(model_)
{
}

template <int dim>
void GN_eps_0_beta_0<dim>::assign_BCs(const bool &at_boundary,
                                      const unsigned &i_face,
                                      const dealii::Point<dim> &face_center)
{
  /* Example 1 */
  if (at_boundary)
  {
    if (face_center[0] < -1 + 1.0E-4)
    {
      this->BCs[i_face] = GenericCell<dim>::BC::essential;
      this->dof_names_on_faces[i_face].resize(dim, 0);
    }
    else if (face_center[0] > 1 - 1.0E-4)
    {
      this->BCs[i_face] = GenericCell<dim>::BC::essential;
      this->dof_names_on_faces[i_face].resize(dim, 0);
    }
    else
    {
      this->BCs[i_face] = GenericCell<dim>::BC::essential;
      this->dof_names_on_faces[i_face].resize(dim, 0);
    }
  }
  else
  {
    this->dof_names_on_faces[i_face].resize(dim, 1);
  }
  /* End of example 1 */
}

template <int dim>
GN_eps_0_beta_0<dim>::~GN_eps_0_beta_0()
{
}

template <int dim>
void GN_eps_0_beta_0<dim>::assign_initial_data()
{
}

template <int dim>
void GN_eps_0_beta_0<dim>::calculate_matrices()
{
  const unsigned n_faces = this->n_faces;
  const unsigned n_cell_bases = this->n_cell_bases;
  const unsigned n_face_bases = this->n_face_bases;
  const unsigned elem_quad_size = this->elem_quad_bundle->size();
  std::vector<dealii::DerivativeForm<1, dim, dim> > d_forms =
    this->cell_quad_fe_vals->get_inverse_jacobians();
  std::vector<dealii::Point<dim> > quad_pt_locs =
    this->cell_quad_fe_vals->get_quadrature_points();
  std::vector<double> cell_JxW = this->cell_quad_fe_vals->get_JxW_values();
  mtl::mat::compressed2D<dealii::Tensor<2, dim> > d_forms_mat(
    elem_quad_size, elem_quad_size, elem_quad_size);
  {
    mtl::mat::inserter<mtl::compressed2D<dealii::Tensor<2, dim> > > ins(
      d_forms_mat);
    for (unsigned int i1 = 0; i1 < elem_quad_size; ++i1)
      ins[i1][i1] << d_forms[i1];
  }
  mtl::mat::dense2D<dealii::Tensor<1, dim> > grad_N_x(
    this->the_elem_basis->bases_grads_at_quads * d_forms_mat);
  A1 = eigen3mat::Zero(dim * n_cell_bases, dim * n_cell_bases);
  A2 = eigen3mat::Zero(n_cell_bases, n_cell_bases);
  B1 = eigen3mat::Zero(dim * n_cell_bases, n_cell_bases);
  B2 = eigen3mat::Zero(dim * n_cell_bases, n_cell_bases);
  D1 = eigen3mat::Zero(dim * n_cell_bases, dim * n_cell_bases);
  C2 = eigen3mat::Zero(dim * n_cell_bases, dim * n_faces * n_face_bases);
  C1 = eigen3mat::Zero(n_cell_bases, dim * n_faces * n_face_bases);
  D2 = eigen3mat::Zero(dim * n_cell_bases, n_cell_bases);
  E1 =
    eigen3mat::Zero(dim * n_faces * n_face_bases, dim * n_faces * n_face_bases);
  eigen3mat grad_NT, NT, N_vec, div_N;
  for (unsigned i_quad = 0; i_quad < this->elem_quad_bundle->size(); ++i_quad)
  {
    grad_NT = eigen3mat::Zero(dim, n_cell_bases);
    div_N = eigen3mat::Zero(dim * n_cell_bases, 1);
    NT = this->the_elem_basis->get_func_vals_at_iquad(i_quad);
    N_vec = eigen3mat::Zero(dim * n_cell_bases, dim);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      N_vec.block(n_cell_bases * i_dim, i_dim, n_cell_bases, 1) =
        NT.transpose();
    for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
    {
      dealii::Tensor<1, dim> grad_N_at_point = grad_N_x[i_poly][i_quad];
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        grad_NT(i_dim, i_poly) = grad_N_at_point[i_dim];
        div_N(n_cell_bases * i_dim + i_poly, 0) = grad_N_at_point[i_dim];
      }
    }
    A1 += cell_JxW[i_quad] * N_vec * N_vec.transpose();
    A2 += cell_JxW[i_quad] * NT.transpose() * NT;
    B1 += cell_JxW[i_quad] * N_vec * grad_NT;
    B2 += cell_JxW[i_quad] * div_N * NT;
  }
  eigen3mat normal(dim, 1);
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    eigen3mat C1_on_face = eigen3mat::Zero(n_cell_bases, dim * n_face_bases);
    eigen3mat C2_on_face =
      eigen3mat::Zero(dim * n_cell_bases, dim * n_face_bases);
    eigen3mat E1_on_face =
      eigen3mat::Zero(dim * n_face_bases, dim * n_face_bases);
    /*
     * Here, we project face quadratue points to the element space.
     * So that we can integrate a function which is defined on the element
     * domain, can be integrated on face.
     */
    std::vector<dealii::Point<dim> > projected_quad_points(
      this->face_quad_bundle->size());
    dealii::QProjector<dim>::project_to_face(
      *(this->face_quad_bundle), i_face, projected_quad_points);
    std::vector<dealii::Point<dim> > normals =
      this->face_quad_fe_vals->get_normal_vectors();
    std::vector<double> face_JxW = this->face_quad_fe_vals->get_JxW_values();
    eigen3mat NT_face = eigen3mat::Zero(1, n_face_bases);
    eigen3mat NT_face_vec = eigen3mat::Zero(dim, dim * n_face_bases);
    eigen3mat N_vec;
    eigen3mat Nj = eigen3mat::Zero(n_cell_bases, 1);
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      N_vec = eigen3mat::Zero(dim * n_cell_bases, dim);
      std::vector<double> N_at_projected_quad_point =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);
      const std::vector<double> &face_basis = this->the_face_basis->value(
        face_quad_points[i_face_quad], this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        NT_face(0, i_polyface) = face_basis[i_polyface];
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        NT_face_vec.block(i_dim, i_dim * n_face_bases, 1, n_face_bases) =
          NT_face;
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        Nj(i_poly, 0) = N_at_projected_quad_point[i_poly];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          N_vec(i_dim * n_cell_bases + i_poly, i_dim) =
            N_at_projected_quad_point[i_poly];
        }
      }
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_face_quad](i_dim);
      D1 += taus[i_face] * face_JxW[i_face_quad] * N_vec * N_vec.transpose();
      C2_on_face += taus[i_face] * face_JxW[i_face_quad] * N_vec * NT_face_vec;
      C1_on_face +=
        face_JxW[i_face_quad] * Nj * normal.transpose() * NT_face_vec;
      D2 += face_JxW[i_face_quad] * N_vec * normal * Nj.transpose();
      E1_on_face += taus[i_face] * face_JxW[i_face_quad] *
                    NT_face_vec.transpose() * NT_face_vec;
    }
    C2.block(
      0, i_face * dim * n_face_bases, dim * n_cell_bases, dim * n_face_bases) =
      C2_on_face;
    C1.block(0, i_face * dim * n_face_bases, n_cell_bases, dim * n_face_bases) =
      C1_on_face;
    E1.block(i_face * dim * n_face_bases,
             i_face * dim * n_face_bases,
             dim * n_face_bases,
             dim * n_face_bases) = E1_on_face;
  }
}

template <int dim>
void GN_eps_0_beta_0<dim>::assemble_globals(const solver_update_keys &keys_)
{
  const std::vector<double> &quad_weights =
    this->elem_quad_bundle->get_weights();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();
  std::vector<dealii::Point<dim> > q_points_loc =
    this->cell_quad_fe_vals->get_quadrature_points();
  std::vector<double> quad_JxWs = this->cell_quad_fe_vals->get_JxW_values();

  this->reinit_cell_fe_vals();
  calculate_matrices();
  eigen3mat A2_inv = std::move(A2.inverse());
  eigen3mat mat1 =
    std::move(A1 + mu / 3.0 * (B1 * A2_inv * B1.transpose() - D1));
  eigen3mat mat2 = std::move(-mu / 3.0 * B1 * A2_inv * C1 + mu / 3.0 * C2);
  eigen3ldlt mat1_ldlt = std::move(mat1.ldlt());
  std::vector<double> cell_mat;
  std::vector<int> row_nums(this->n_faces * dim * this->n_face_bases, -1);
  std::vector<int> col_nums(this->n_faces * dim * this->n_face_bases, -1);

  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    unsigned dof_count = 0;
    for (unsigned i_dof = 0; i_dof < this->dof_names_on_faces[i_face].size();
         ++i_dof)
    {
      if (this->dof_names_on_faces[i_face][i_dof])
      {
        for (unsigned i_polyface = 0; i_polyface < this->n_face_bases;
             ++i_polyface)
        {
          unsigned global_dof_number =
            this->dofs_ID_in_all_ranks[i_face][dof_count] * this->n_face_bases +
            i_polyface;
          unsigned i_num = i_face * dim * this->n_face_bases +
                           dof_count * this->n_face_bases + i_polyface;
          col_nums[i_num] = global_dof_number;
          row_nums[i_num] = global_dof_number;
        }
        ++dof_count;
      }
    }
  }

  eigen3mat zeta(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> zeta_mtl;
  this->project_to_elem_basis(
    zeta_func, *(this->the_elem_basis), quad_weights, zeta_mtl);
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    zeta(i_poly, 0) = zeta_mtl[i_poly];

  eigen3mat grad_zeta(this->n_cell_bases * dim, 1);
  mtl::vec::dense_vector<dealii::Tensor<1, dim> > grad_zeta_mtl;
  this->project_to_elem_basis(
    grad_zeta_func, *(this->the_elem_basis), quad_weights, grad_zeta_mtl);
  for (unsigned i2 = 0; i2 < dim; ++i2)
    for (unsigned i1 = 0; i1 < this->n_cell_bases; ++i1)
      grad_zeta(i2 * this->n_cell_bases + i1, 0) = grad_zeta_mtl[i1][i2];

  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    for (unsigned i_dof = 0; i_dof < dim; ++i_dof)
    {
      for (unsigned i_polyface = 0; i_polyface < this->n_face_bases;
           ++i_polyface)
      {
        unsigned i_num = i_face * dim * this->n_face_bases +
                         i_dof * this->n_face_bases + i_polyface;
        eigen3mat q1_hat = std::move(
          eigen3mat::Zero(this->n_faces * dim * this->n_face_bases, 1));
        q1_hat(i_num, 0) = 1.0;
        eigen3mat q1 = mat1_ldlt.solve(-mat2 * q1_hat);
        eigen3mat q2 = std::move(A2_inv * (-B1.transpose() * q1 + C1 * q1_hat));
        eigen3mat jth_col =
          (C1.transpose() * q2 + C2.transpose() * q1 - E1 * q1_hat);
        cell_mat.insert(
          cell_mat.end(), jth_col.data(), jth_col.data() + jth_col.rows());
      }
    }
  }
  if (keys_ & update_mat)
    model->solver->push_to_global_mat(row_nums, col_nums, cell_mat, ADD_VALUES);

  if (keys_ & update_rhs)
  {
    eigen3mat q1_hat =
      eigen3mat::Zero(this->n_faces * dim * this->n_face_bases, 1);
    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      unsigned n_dofs = this->dofs_ID_in_all_ranks[i_face].size();
      if (this->BCs[i_face] == GenericCell<dim>::essential)
      {
        mtl::vec::dense_vector<dealii::Tensor<1, dim> > q1_hat_mtl;
        this->reinit_face_fe_vals(i_face);
        this->project_essential_BC_to_face(
          q1_func, *(this->the_face_basis), face_quad_weights, q1_hat_mtl, 0.0);
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
          for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
            q1_hat((i_face * dim + i_dim) * this->n_face_bases + i_poly) =
              q1_hat_mtl[i_poly][i_dim];
      }
      if (this->BCs[i_face] == GenericCell<dim>::flux_bc)
      {
        for (unsigned i_dof = 0; i_dof < n_dofs; ++i_dof)
        {
        }
      }
    }
    eigen3mat q1 = mat1_ldlt.solve(B1 * zeta - mat2 * q1_hat);
    eigen3mat q2 = std::move(A2_inv * (-B1.transpose() * q1 + C1 * q1_hat));
    eigen3mat jth_col_vec =
      -(C1.transpose() * q2 + C2.transpose() * q1 - E1 * q1_hat);
    std::vector<double> rhs_col(jth_col_vec.data(),
                                jth_col_vec.data() + jth_col_vec.rows());
    model->solver->push_to_rhs_vec(row_nums, rhs_col, ADD_VALUES);
  }

  if (keys_ & update_sol)
  {
    eigen3mat q1_hat =
      eigen3mat::Zero(dim * this->n_faces * this->n_face_bases, 1);
    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      mtl::vec::dense_vector<dealii::Tensor<1, dim> > q1_hat_mtl;
      this->reinit_face_fe_vals(i_face);
      this->project_essential_BC_to_face(
        q1_func, *(this->the_face_basis), face_quad_weights, q1_hat_mtl, 0.0);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
          q1_hat((i_face * dim + i_dim) * this->n_face_bases + i_poly) =
            q1_hat_mtl[i_poly][i_dim];
    }
    std::vector<double> exact_uhat_vec(q1_hat.data(),
                                       q1_hat.data() + q1_hat.rows());
    model->solver->push_to_exact_sol(row_nums, exact_uhat_vec, INSERT_VALUES);
  }
  wreck_it_Ralph(A1);
  wreck_it_Ralph(A2);
  wreck_it_Ralph(B1);
  wreck_it_Ralph(B2);
  wreck_it_Ralph(D1);
  wreck_it_Ralph(C2);
  wreck_it_Ralph(C1);
  wreck_it_Ralph(D2);
  wreck_it_Ralph(E1);
}

template <int dim>
template <typename T>
double GN_eps_0_beta_0<dim>::compute_internal_dofs(
  const double *const local_uhat_vec,
  eigen3mat &q2,
  eigen3mat &q1,
  const poly_space_basis<T, dim> &output_basis)
{
  const std::vector<double> &quad_weights =
    this->elem_quad_bundle->get_weights();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();

  this->reinit_cell_fe_vals();
  calculate_matrices();
  eigen3mat A2_inv = std::move(A2.inverse());
  eigen3mat mat1 =
    std::move(A1 + mu / 3.0 * (B1 * A2_inv * B1.transpose() - D1));
  eigen3mat mat2 = std::move(-mu / 3.0 * B1 * A2_inv * C1 + mu / 3.0 * C2);
  eigen3ldlt mat1_ldlt = std::move(mat1.ldlt());

  eigen3mat zeta(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> zeta_mtl;
  this->project_to_elem_basis(
    zeta_func, *(this->the_elem_basis), quad_weights, zeta_mtl);
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    zeta(i_poly, 0) = zeta_mtl[i_poly];
  eigen3mat grad_zeta = B1 * zeta;

  eigen3mat exact_q1_hat(dim * this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim> > q1_hat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(
      q1_func, *(this->the_face_basis), face_quad_weights, q1_hat_mtl, 0.0);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
        exact_q1_hat((i_face * dim + i_dim) * this->n_face_bases + i_poly) =
          q1_hat_mtl[i_poly][i_dim];
  }

  eigen3mat solved_q1_hat = exact_q1_hat;
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    for (unsigned i_dof = 0; i_dof < this->dofs_ID_in_this_rank[i_face].size();
         ++i_dof)
    {
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
      {
        int global_dof_number =
          this->dofs_ID_in_this_rank[i_face][i_dof] * this->n_face_bases +
          i_poly;
        solved_q1_hat((i_face * dim + i_dof) * this->n_face_bases + i_poly, 0) =
          local_uhat_vec[global_dof_number];
      }
    }
  }

  q1 = mat1_ldlt.solve(grad_zeta - mat2 * solved_q1_hat);
  q2 = std::move(A2_inv * (-B1.transpose() * q1 + C1 * solved_q1_hat));
  eigen3mat jth_col_vec =
    (C1.transpose() * q2 + C2.transpose() * q1 - E1 * solved_q1_hat);

  eigen3mat nodal_u = output_basis.get_dof_vals_at_quads(q2);
  eigen3mat nodal_q(dim * this->n_cell_bases, 1);
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    nodal_q.block(i_dim * this->n_cell_bases, 0, this->n_cell_bases, 1) =
      output_basis.get_dof_vals_at_quads(
        q1.block(i_dim * this->n_cell_bases, 0, this->n_cell_bases, 1));
  }
  unsigned n_local_dofs = nodal_u.rows();
  /* Now we calculate the refinement critera */
  unsigned i_cell = this->id_num;
  for (unsigned i_local_dofs = 0; i_local_dofs < n_local_dofs; ++i_local_dofs)
  {
    double temp_val = 0;
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      temp_val += std::pow(nodal_q(i_dim * n_local_dofs + i_local_dofs, 0), 2);
    }
    this->refn_local_nodal->assemble(i_cell * n_local_dofs + i_local_dofs,
                                     sqrt(temp_val));
  }
  for (unsigned i_local_unknown = 0; i_local_unknown < n_local_dofs;
       ++i_local_unknown)
  {
    {
      unsigned idx1 = (i_cell * n_local_dofs) * (dim + 1) + i_local_unknown;
      this->cell_local_nodal->assemble(idx1, nodal_u(i_local_unknown, 0));
    }
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      unsigned idx1 = (i_cell * n_local_dofs) * (dim + 1) +
                      (i_dim + 1) * n_local_dofs + i_local_unknown;
      this->cell_local_nodal->assemble(
        idx1, nodal_q(i_dim * n_local_dofs + i_local_unknown, 0));
    }
  }

  wreck_it_Ralph(A1);
  wreck_it_Ralph(A2);
  wreck_it_Ralph(B1);
  wreck_it_Ralph(B2);
  wreck_it_Ralph(D1);
  wreck_it_Ralph(C2);
  wreck_it_Ralph(C1);
  wreck_it_Ralph(D2);
  wreck_it_Ralph(E1);
  return 0.;
}

template <int dim>
void GN_eps_0_beta_0<dim>::internal_vars_errors(const eigen3mat &u_vec,
                                                const eigen3mat &q_vec,
                                                double &u_error,
                                                double &q_error)
{
  double error_u2 = this->get_error_in_cell(q2_func, u_vec);
  double error_q2 = this->get_error_in_cell(q1_func, q_vec);
  u_error += error_u2;
  q_error += error_q2;
}

template <int dim>
void GN_eps_0_beta_0<dim>::ready_for_next_iteration()
{
}

template <int dim>
void GN_eps_0_beta_0<dim>::ready_for_next_time_step()
{
}
