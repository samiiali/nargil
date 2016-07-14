#include "advection_diffusion.hpp"
#include "support_classes.hpp"

template <int dim>
adv_c_vec_func_class<dim, eigen3mat> AdvectionDiffusion<dim>::c_vec_func{};
template <int dim>
adv_u_func_class<dim, double> AdvectionDiffusion<dim>::u_func{};
template <int dim>
adv_f1_func_class<dim, double> AdvectionDiffusion<dim>::f1_func{};
template <int dim>
adv_uinf_func_class<dim, double> AdvectionDiffusion<dim>::uinf_func{};

template <int dim>
solver_options AdvectionDiffusion<dim>::required_solver_options()
{
  return solver_options::default_option;
}

template <int dim>
solver_type AdvectionDiffusion<dim>::required_solver_type()
{
  return solver_type::implicit_petsc_aij;
}

template <int dim>
unsigned AdvectionDiffusion<dim>::get_num_dofs_per_node()
{
  return 1;
}

/*!
 * The move constrcutor of the derived class should call the move
 * constructor of the base class using std::move. Otherwise the copy
 * constructor will be called.
 */
template <int dim>
AdvectionDiffusion<dim>::AdvectionDiffusion(AdvectionDiffusion &&inp_cell)
  noexcept : GenericCell<dim>(std::move(inp_cell)),
             taus(std::move(inp_cell.taus)),
             model(inp_cell.model),
             u_i_s(std::move(inp_cell.u_i_s))
{
}

template <int dim>
AdvectionDiffusion<dim>::AdvectionDiffusion(
  typename GenericCell<dim>::dealiiCell &inp_cell,
  const unsigned &id_num_,
  const unsigned &poly_order_,
  hdg_model<dim, AdvectionDiffusion> *model_)
  : GenericCell<dim>(inp_cell, id_num_, poly_order_),
    taus(this->n_faces, 100.0),
    model(model_),
    time_integrator(model_->time_integrator),
    u_i_s(time_integrator->order, eigen3mat::Zero(this->n_cell_bases, 1))
{
}

template <int dim>
void AdvectionDiffusion<dim>::assign_BCs(const bool &at_boundary,
                                         const unsigned &i_face,
                                         const dealii::Point<dim> &face_center)
{
  /* Example 1 */
  if (at_boundary)
  {
    if (face_center[0] < -1 + 1.0E-4)
    {
      this->BCs[i_face] = GenericCell<dim>::BC::flux_bc;
      this->dof_names_on_faces[i_face].resize(1, 1);
    }
    else if (face_center[0] > 1 - 1.0E-4)
    {
      this->BCs[i_face] = GenericCell<dim>::BC::flux_bc;
      this->dof_names_on_faces[i_face].resize(1, 1);
    }
    else
    {
      this->BCs[i_face] = GenericCell<dim>::BC::flux_bc;
      this->dof_names_on_faces[i_face].resize(1, 1);
    }
  }
  else
  {
    this->dof_names_on_faces[i_face].resize(1, 1);
  }
  /* End of example 1 */
}

template <int dim>
AdvectionDiffusion<dim>::~AdvectionDiffusion()
{
}

template <int dim>
void AdvectionDiffusion<dim>::assign_initial_data()
{
  this->reinit_cell_fe_vals();
  const std::vector<double> &quad_weights =
    this->elem_quad_bundle->get_weights();
  mtl::vec::dense_vector<double> u0_mtl;
  this->project_to_elem_basis(
    u_func, *(this->the_elem_basis), quad_weights, u0_mtl);
  for (unsigned i_dof = 0; i_dof < this->n_cell_bases; ++i_dof)
    u_i_s[u_i_s.size() - 1](i_dof, 0) = u0_mtl[i_dof];
}

template <int dim>
void AdvectionDiffusion<dim>::compute_next_time_step_rhs()
{
}

template <int dim>
void AdvectionDiffusion<dim>::calculate_matrices()
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

  A2 = eigen3mat::Zero(n_cell_bases, n_cell_bases);
  B3 = eigen3mat::Zero(n_cell_bases, n_cell_bases);
  C3 = eigen3mat::Zero(n_cell_bases, n_faces * n_face_bases);
  C4 = eigen3mat::Zero(n_cell_bases, n_faces * n_face_bases);
  C5 = eigen3mat::Zero(n_cell_bases, n_faces * n_face_bases);
  D3 = eigen3mat::Zero(n_cell_bases, n_cell_bases);
  E2 = eigen3mat::Zero(n_faces * n_face_bases, n_faces * n_face_bases);
  E3 = eigen3mat::Zero(n_faces * n_face_bases, n_faces * n_face_bases);
  U8 = eigen3mat::Zero(n_faces * n_face_bases, n_faces * n_face_bases);

  eigen3mat grad_NT, NT;
  for (unsigned i_quad = 0; i_quad < this->elem_quad_bundle->size(); ++i_quad)
  {
    grad_NT = eigen3mat::Zero(dim, n_cell_bases);
    NT = this->the_elem_basis->get_func_vals_at_iquad(i_quad);
    for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
    {
      dealii::Tensor<1, dim> grad_N_at_point = grad_N_x[i_poly][i_quad];
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        grad_NT(i_dim, i_poly) = grad_N_at_point[i_dim];
      }
    }
    const eigen3mat &c_vec = c_vec_func.value(
      quad_pt_locs[i_quad], quad_pt_locs[i_quad], time_integrator->next_time);
    A2 += cell_JxW[i_quad] * NT.transpose() * NT;
    B3 += cell_JxW[i_quad] * grad_NT.transpose() * c_vec * NT;
  }
  eigen3mat normal(dim, 1);
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    eigen3mat C3_on_face = eigen3mat::Zero(n_cell_bases, n_face_bases);
    eigen3mat C4_on_face = eigen3mat::Zero(n_cell_bases, n_face_bases);
    eigen3mat C5_on_face = eigen3mat::Zero(n_cell_bases, n_face_bases);
    eigen3mat E2_on_face = eigen3mat::Zero(n_face_bases, n_face_bases);
    eigen3mat E3_on_face = eigen3mat::Zero(n_face_bases, n_face_bases);
    eigen3mat U8_on_face = eigen3mat::Zero(n_face_bases, n_face_bases);
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
    std::vector<dealii::Point<dim> > face_quad_points_loc =
      this->face_quad_fe_vals->get_quadrature_points();
    eigen3mat NT_face = eigen3mat::Zero(1, n_face_bases);
    eigen3mat Nj = eigen3mat::Zero(n_cell_bases, 1);
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      std::vector<double> N_at_projected_quad_point =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);
      const std::vector<double> &face_basis = this->the_face_basis->value(
        face_quad_points[i_face_quad], this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        NT_face(0, i_polyface) = face_basis[i_polyface];
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        Nj(i_poly, 0) = N_at_projected_quad_point[i_poly];
      }
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_face_quad](i_dim);
      eigen3mat c_vec = c_vec_func.value(face_quad_points_loc[i_face_quad],
                                         face_quad_points_loc[i_face_quad],
                                         time_integrator->next_time);
      double c_dot_n = (c_vec.transpose() * normal)(0, 0);
      double A_n_plus = (c_dot_n + fabs(c_dot_n)) / 2.;
      double A_n_minus = (c_dot_n - fabs(c_dot_n)) / 2.;
      double abs_A_n = fabs(c_dot_n);
      /* Now we start multiplying the basis functions to get the matrices. */
      C3_on_face +=
        (c_dot_n - taus[i_face]) * face_JxW[i_face_quad] * Nj * NT_face;
      D3 += taus[i_face] * face_JxW[i_face_quad] * Nj * Nj.transpose();
      if (this->BCs[i_face] != GenericCell<dim>::flux_bc)
      {
        C4_on_face += taus[i_face] * face_JxW[i_face_quad] * Nj * NT_face;
        E2_on_face += (c_dot_n - taus[i_face]) * face_JxW[i_face_quad] *
                      NT_face.transpose() * NT_face;
      }
      if (this->BCs[i_face] == GenericCell<dim>::flux_bc)
      {
        C5_on_face += A_n_plus * face_JxW[i_face_quad] * Nj * NT_face;
        E3_on_face +=
          abs_A_n * face_JxW[i_face_quad] * NT_face.transpose() * NT_face;
        U8_on_face +=
          A_n_minus * face_JxW[i_face_quad] * NT_face.transpose() * NT_face;
      }
    }
    C3.block(0, i_face * n_face_bases, n_cell_bases, n_face_bases) = C3_on_face;
    C4.block(0, i_face * n_face_bases, n_cell_bases, n_face_bases) = C4_on_face;
    C5.block(0, i_face * n_face_bases, n_cell_bases, n_face_bases) = C5_on_face;
    E2.block(i_face * n_face_bases,
             i_face * n_face_bases,
             n_face_bases,
             n_face_bases) = E2_on_face;
    E3.block(i_face * n_face_bases,
             i_face * n_face_bases,
             n_face_bases,
             n_face_bases) = E3_on_face;
    U8.block(i_face * n_face_bases,
             i_face * n_face_bases,
             n_face_bases,
             n_face_bases) = U8_on_face;
  }
}

template <int dim>
void AdvectionDiffusion<dim>::assemble_globals(const solver_update_keys &keys_)
{
  const std::vector<double> &quad_weights =
    this->elem_quad_bundle->get_weights();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();
  /* Now, we want to obtain the rhs forcing, due to previous time steps */
  eigen3mat sum_u_i_s = time_integrator->compute_sum_y_i(u_i_s);
  double beta_h = time_integrator->get_beta_h();
  /* Recalculate FE Values for this element */
  this->reinit_cell_fe_vals();
  calculate_matrices();
  eigen3mat mat1 = A2 + beta_h * (D3 - B3);
  Eigen::PartialPivLU<eigen3mat> mat1_lu(mat1);
  std::vector<int> row_nums(this->n_faces * this->n_face_bases, -1);
  std::vector<int> col_nums(this->n_faces * this->n_face_bases, -1);
  std::vector<double> cell_mat;

  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    for (unsigned i_polyface = 0; i_polyface < this->n_face_bases; ++i_polyface)
    {
      unsigned i_num = i_face * this->n_face_bases + i_polyface;
      int global_dof_number;
      if (this->dofs_ID_in_all_ranks[i_face].size() > 0)
      {
        global_dof_number =
          this->dofs_ID_in_all_ranks[i_face][0] * this->n_face_bases +
          i_polyface;
        row_nums[i_num] = global_dof_number;
        col_nums[i_num] = global_dof_number;
      }
    }

  eigen3mat u_inf(this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<double> u_hat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(uinf_func,
                                       *(this->the_face_basis),
                                       face_quad_weights,
                                       u_hat_mtl,
                                       time_integrator->next_time);
    for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
      u_inf(i_face * this->n_face_bases + i_poly) = u_hat_mtl[i_poly];
  }

  for (unsigned j_face = 0; j_face < this->n_faces && keys_; ++j_face)
  {
    for (unsigned j_poly = 0; j_poly < this->n_face_bases; ++j_poly)
    {
      unsigned j_num = j_face * this->n_face_bases + j_poly;
      eigen3mat u_hat = eigen3mat::Zero(this->n_faces * this->n_face_bases, 1);
      u_hat(j_num, 0) = 1.0;
      eigen3mat u = mat1_lu.solve(-beta_h * C3 * u_hat);
      eigen3mat jth_col = ((C4 + C5).transpose() * u + (E2 - E3) * u_hat);
      cell_mat.insert(
        cell_mat.end(), jth_col.data(), jth_col.data() + jth_col.rows());
    }
  }
  if (keys_ & update_mat)
    this->model->solver->push_to_global_mat(
      row_nums, col_nums, cell_mat, ADD_VALUES);

  eigen3mat f1(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> f1_mtl;
  this->project_to_elem_basis(f1_func,
                              *(this->the_elem_basis),
                              quad_weights,
                              f1_mtl,
                              time_integrator->next_time);
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    f1(i_poly, 0) = f1_mtl[i_poly];

  /*
  eigen3mat u_t(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> u_t_mtl;
  this->project_to_elem_basis(u_t_func, *(this->the_elem_basis), quad_weights,
  u_t_mtl);
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    u_t(i_poly, 0) = u_t_mtl[i_poly];
  */

  if (keys_ & update_rhs)
  {
    eigen3mat u_hat = eigen3mat::Zero(this->n_faces * this->n_face_bases, 1);
    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      if (this->BCs[i_face] == GenericCell<dim>::essential)
        u_hat.block(i_face * this->n_face_bases, 0, this->n_face_bases, 1) =
          u_inf.block(i_face * this->n_face_bases, 0, this->n_face_bases, 1);
    }
    eigen3mat u =
      mat1_lu.solve(beta_h * A2 * f1 - A2 * sum_u_i_s - beta_h * C3 * u_hat);
    eigen3mat jth_col_vec =
      -((C4 + C5).transpose() * u + (E2 - E3) * u_hat - U8 * u_inf);
    std::vector<double> rhs_col(jth_col_vec.data(),
                                jth_col_vec.data() + jth_col_vec.rows());
    model->solver->push_to_rhs_vec(row_nums, rhs_col, ADD_VALUES);
  }

  if (keys_ & update_sol)
  {
    std::vector<double> exact_uhat_vec(u_inf.data(),
                                       u_inf.data() + u_inf.rows());
    eigen3mat u =
      mat1_lu.solve(beta_h * A2 * f1 - A2 * sum_u_i_s - beta_h * C3 * u_inf);
    eigen3mat jth_col_vec =
      (C4 + C5).transpose() * u + (E2 - E3) * u_inf - U8 * u_inf;
    model->solver->push_to_exact_sol(row_nums, exact_uhat_vec, ADD_VALUES);
  }

  wreck_it_Ralph(A2);
  wreck_it_Ralph(B3);
  wreck_it_Ralph(C3);
  wreck_it_Ralph(C4);
  wreck_it_Ralph(C5);
  wreck_it_Ralph(D3);
  wreck_it_Ralph(E2);
  wreck_it_Ralph(E3);
  wreck_it_Ralph(U8);
}

template <int dim>
template <typename T>
double
AdvectionDiffusion<dim>::compute_internal_dofs(const double *const local_uhat,
                                               eigen3mat &u,
                                               eigen3mat &q,
                                               const poly_space_basis<T, dim>
                                                 output_basis)
{
  const std::vector<double> &quad_weights =
    this->elem_quad_bundle->get_weights();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();
  /* Now, we want to obtain the rhs forcing, due to previous time steps */
  eigen3mat sum_u_i_s = time_integrator->compute_sum_y_i(u_i_s);
  double beta_h = time_integrator->get_beta_h();
  /* Next, we update the FE values for the current cell. */
  this->reinit_cell_fe_vals();
  calculate_matrices();
  eigen3mat mat1 = A2 + beta_h * (D3 - B3);
  Eigen::PartialPivLU<eigen3mat> mat1_lu(mat1);

  eigen3mat exact_u_hat(this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<double> u_hat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(u_func,
                                       *(this->the_face_basis),
                                       face_quad_weights,
                                       u_hat_mtl,
                                       time_integrator->next_time);
    for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
      exact_u_hat(i_face * this->n_face_bases + i_poly) = u_hat_mtl[i_poly];
  }

  eigen3mat f1(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> f1_mtl;
  this->project_to_elem_basis(f1_func,
                              *(this->the_elem_basis),
                              quad_weights,
                              f1_mtl,
                              time_integrator->next_time);
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    f1(i_poly, 0) = f1_mtl[i_poly];

  eigen3mat solved_u_hat = exact_u_hat;
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
        solved_u_hat(i_face * this->n_face_bases + i_poly, 0) =
          local_uhat[global_dof_number];
      }
    }
  }
  u = mat1_lu.solve(beta_h * A2 * f1 - A2 * sum_u_i_s -
                    beta_h * C3 * solved_u_hat);
  time_integrator->back_substitute_y_i_s(u_i_s, u);

  eigen3mat nodal_u = output_basis.get_dof_vals_at_quads(u);
  //  eigen3mat nodal_u = mode2supp_mat * u;
  q = std::move(eigen3mat::Zero(dim * this->n_cell_bases, 1));
  eigen3mat nodal_q = std::move(eigen3mat::Zero(dim * this->n_cell_bases, 1));
  unsigned n_local_dofs = nodal_u.rows();
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
  wreck_it_Ralph(A2);
  wreck_it_Ralph(B3);
  wreck_it_Ralph(C3);
  wreck_it_Ralph(C4);
  wreck_it_Ralph(C5);
  wreck_it_Ralph(D3);
  wreck_it_Ralph(E2);
  wreck_it_Ralph(E3);
  wreck_it_Ralph(U8);
  return 0.;
}

template <int dim>
void AdvectionDiffusion<dim>::internal_vars_errors(const eigen3mat &u_vec,
                                                   const eigen3mat &,
                                                   double &u_error,
                                                   double &q_error)
{
  double error_u2 = this->get_error_in_cell(u_func, u_vec);

  u_error += error_u2;
  q_error += 0;
}

template <int dim>
void AdvectionDiffusion<dim>::ready_for_next_iteration()
{
}

template <int dim>
void AdvectionDiffusion<dim>::ready_for_next_time_step()
{
}
