#include "diffusion.hpp"
#include "support_classes.hpp"

template <int dim>
kappa_inv_class<dim, Eigen::MatrixXd> Diffusion<dim>::kappa_inv{};
template <int dim>
tau_func_class<dim> Diffusion<dim>::tau_func{};
template <int dim>
u_func_class<dim> Diffusion<dim>::u_func{};
template <int dim>
q_func_class<dim, dealii::Tensor<1, dim> > Diffusion<dim>::q_func{};
template <int dim>
divq_func_class<dim> Diffusion<dim>::divq_func{};
template <int dim>
f_func_class<dim> Diffusion<dim>::f_func{};
template <int dim>
dirichlet_BC_func_class<dim> Diffusion<dim>::dirichlet_bc_func{};
template <int dim>
neumann_BC_func_class<dim> Diffusion<dim>::Neumann_BC_func{};

template <int dim>
solver_options Diffusion<dim>::required_solver_options()
{
  return solver_options::spd_matrix;
}

template <int dim>
solver_type Diffusion<dim>::required_solver_type()
{
  return solver_type::implicit_petsc_aij;
}

template <int dim>
unsigned Diffusion<dim>::get_num_dofs_per_node()
{
  return 1;
}

/*!
 * The move constrcutor of the derived class should call the move
 * constructor of the base class using std::move. Otherwise the copy
 * constructor will be called.
 */
template <int dim>
Diffusion<dim>::Diffusion(Diffusion &&inp_cell) noexcept
  : GenericCell<dim>(std::move(inp_cell)),
    model(inp_cell.model)
{
}

template <int dim>
Diffusion<dim>::Diffusion(typename GenericCell<dim>::dealiiCell &inp_cell,
                          const unsigned &id_num_,
                          const unsigned &poly_order_,
                          hdg_model<dim, Diffusion> *model_)
  : GenericCell<dim>(inp_cell, id_num_, poly_order_), model(model_)
{
}

template <int dim>
void Diffusion<dim>::assign_BCs(const bool &at_boundary,
                                const unsigned &i_face,
                                const dealii::Point<dim> &face_center)
{
  /* Example 1 */
  /*
  if (at_boundary)
  {
    if (fabs(face_center[0]) > 1. - 1.e-4)
    {
      this->BCs[i_face] = GenericCell<dim>::BC::essential;
      this->dof_names_on_faces[i_face].resize(1, 0);
    }
    else
    {
      this->BCs[i_face] = GenericCell<dim>::BC::essential;
      this->dof_names_on_faces[i_face].resize(1, 0);
    }
  }
  else
  {
    this->dof_names_on_faces[i_face].resize(1, 1);
  }
  */
  /* End of example 1 */
  /* Francois's Example 1 */
  if (at_boundary && face_center[0] < 1000)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::essential;
    this->dof_names_on_faces[i_face].resize(1, 0);
  }
  else
  {
    this->dof_names_on_faces[i_face].resize(1, 1);
  }
  /* End of example 1 */
}

template <int dim>
Diffusion<dim>::~Diffusion()
{
}

template <int dim>
void Diffusion<dim>::assign_initial_data()
{
}

template <int dim>
void Diffusion<dim>::calculate_matrices()
{
  const unsigned n_faces = this->n_faces;
  const unsigned n_cell_basis = this->n_cell_bases;
  const unsigned n_face_basis = this->n_face_bases;
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
  mtl::mat::dense2D<dealii::Tensor<1, dim> > grad_Ni_x(
    this->the_elem_basis->bases_grads_at_quads * d_forms_mat);

  A = eigen3mat::Zero(dim * n_cell_basis, dim * n_cell_basis);
  B = eigen3mat::Zero(dim * n_cell_basis, n_cell_basis);
  C = eigen3mat::Zero(dim * n_cell_basis, n_faces * n_face_basis);
  D = eigen3mat::Zero(n_cell_basis, n_cell_basis);
  E = eigen3mat::Zero(n_cell_basis, n_faces * n_face_basis);
  H = eigen3mat::Zero(n_faces * n_face_basis, n_faces * n_face_basis);
  H2 = eigen3mat::Zero(n_faces * n_face_basis, n_faces * n_face_basis);
  M = eigen3mat::Zero(n_cell_basis, n_cell_basis);

  eigen3mat Ni_div, NjT, Ni_vec;
  for (unsigned i1 = 0; i1 < this->elem_quad_bundle->size(); ++i1)
  {
    Ni_div = eigen3mat::Zero(dim * n_cell_basis, 1);
    NjT = this->the_elem_basis->get_func_vals_at_iquad(i1);
    Ni_vec = eigen3mat::Zero(dim * n_cell_basis, dim);
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      Ni_vec.block(n_cell_basis * i_dim, i_dim, n_cell_basis, 1) =
        NjT.transpose();
    for (unsigned i_poly = 0; i_poly < n_cell_basis; ++i_poly)
    {
      dealii::Tensor<1, dim> N_grads_X = grad_Ni_x[i_poly][i1];
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        Ni_div(n_cell_basis * i_dim + i_poly, 0) = N_grads_X[i_dim];
    }
    eigen3mat kappa_inv_ = kappa_inv.value(quad_pt_locs[i1], quad_pt_locs[i1]);
    A += cell_JxW[i1] * Ni_vec * kappa_inv_ * Ni_vec.transpose();
    M += cell_JxW[i1] * NjT.transpose() * NjT;
    B += cell_JxW[i1] * Ni_div * NjT;
  }
  eigen3mat normal(dim, 1);
  std::vector<dealii::Point<dim - 1> > Face_Q_Points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    eigen3mat C_on_face = eigen3mat::Zero(dim * n_cell_basis, n_face_basis);
    eigen3mat E_on_face = eigen3mat::Zero(n_cell_basis, n_face_basis);
    eigen3mat H_on_face = eigen3mat::Zero(n_face_basis, n_face_basis);
    eigen3mat H2_on_face = eigen3mat::Zero(n_face_basis, n_face_basis);
    std::vector<dealii::Point<dim> > projected_face_Q_points(
      this->face_quad_bundle->size());
    dealii::QProjector<dim>::project_to_face(
      *(this->face_quad_bundle), i_face, projected_face_Q_points);
    std::vector<dealii::Point<dim> > normals =
      this->face_quad_fe_vals->get_normal_vectors();
    std::vector<double> Face_JxW = this->face_quad_fe_vals->get_JxW_values();
    std::vector<dealii::Point<2> > quads_loc =
      this->face_quad_fe_vals->get_quadrature_points();
    eigen3mat NjT_Face = eigen3mat::Zero(1, n_face_basis);
    eigen3mat Nj_vec;
    eigen3mat Nj = eigen3mat::Zero(n_cell_basis, 1);
    for (unsigned i_Q_face = 0; i_Q_face < this->face_quad_bundle->size();
         ++i_Q_face)
    {
      Nj_vec = eigen3mat::Zero(dim * n_cell_basis, dim);
      std::vector<double> N_valus =
        this->the_elem_basis->value(projected_face_Q_points[i_Q_face]);
      const std::vector<double> &half_range_face_basis =
        this->the_face_basis->value(Face_Q_Points[i_Q_face],
                                    this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_basis; ++i_polyface)
        NjT_Face(0, i_polyface) = half_range_face_basis[i_polyface];
      for (unsigned i_poly = 0; i_poly < n_cell_basis; ++i_poly)
      {
        Nj(i_poly, 0) = N_valus[i_poly];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
          Nj_vec(i_dim * n_cell_basis + i_poly, i_dim) = N_valus[i_poly];
      }
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_Q_face](i_dim);
      double tau_ = tau_func.value(quads_loc[i_Q_face], normals[i_Q_face]);
      C_on_face += Face_JxW[i_Q_face] * Nj_vec * normal * NjT_Face;
      D += Face_JxW[i_Q_face] * tau_ * Nj * Nj.transpose();
      E_on_face += Face_JxW[i_Q_face] * tau_ * Nj * NjT_Face;
      H_on_face += Face_JxW[i_Q_face] * tau_ * NjT_Face.transpose() * NjT_Face;
      H2_on_face += Face_JxW[i_Q_face] * NjT_Face.transpose() * NjT_Face;
    }
    H.block(i_face * n_face_basis,
            i_face * n_face_basis,
            n_face_basis,
            n_face_basis) = H_on_face;
    H2.block(i_face * n_face_basis,
             i_face * n_face_basis,
             n_face_basis,
             n_face_basis) = H2_on_face;
    C.block(0, i_face * n_face_basis, dim * n_cell_basis, n_face_basis) =
      C_on_face;
    E.block(0, i_face * n_face_basis, n_cell_basis, n_face_basis) = E_on_face;
  }
}

template <int dim>
void Diffusion<dim>::calculate_postprocess_matrices()
{
  const unsigned n_cell_basis = this->n_cell_bases;
  const unsigned elem_quad_size = this->elem_quad_bundle->size();
  const unsigned n_cell_basis1 = pow(this->poly_order + 2, dim);

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
  mtl::mat::dense2D<dealii::Tensor<1, dim> > grad_Ni_x(
    model->manager->postprocess_cell_basis.bases_grads_at_quads * d_forms_mat);

  DM_star = eigen3mat::Zero(n_cell_basis1, n_cell_basis1);
  DB2 = eigen3mat::Zero(n_cell_basis1, dim * n_cell_basis);

  Eigen::MatrixXd Ni_grad, Ni_vecT;
  for (unsigned i_quad = 0; i_quad < elem_quad_size; ++i_quad)
  {
    Ni_grad = Eigen::MatrixXd::Zero(n_cell_basis1, dim);
    Ni_vecT = Eigen::MatrixXd::Zero(dim, dim * n_cell_basis);
    for (unsigned i_poly = 0; i_poly < n_cell_basis1; ++i_poly)
    {
      dealii::Tensor<1, dim> grad_Ni_at_iquad = grad_Ni_x[i_poly][i_quad];
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        Ni_grad(i_poly, i_dim) = grad_Ni_at_iquad[i_dim];
    }
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      Ni_vecT.block(i_dim, i_dim * n_cell_basis, 1, n_cell_basis) =
        this->the_elem_basis->get_func_vals_at_iquad(i_quad);
    DM_star += cell_JxW[i_quad] * Ni_grad * Ni_grad.transpose();
    Eigen::MatrixXd kappa_inv_ =
      kappa_inv.value(quad_pt_locs[i_quad], quad_pt_locs[i_quad]);
    DB2 += cell_JxW[i_quad] * Ni_grad * kappa_inv_ * Ni_vecT;
  }
}

template <int dim>
void Diffusion<dim>::assemble_globals(const solver_update_keys &keys_)
{
  unsigned n_polys = this->n_cell_bases;
  unsigned n_polyfaces = this->n_face_bases;
  const std::vector<double> &Q_Weights = this->elem_quad_bundle->get_weights();
  const std::vector<double> &Face_Q_Weights =
    this->face_quad_bundle->get_weights();

  this->reinit_cell_fe_vals();
  calculate_matrices();
  Eigen::FullPivLU<eigen3mat> lu_of_A(A);
  eigen3mat Ainv = lu_of_A.inverse();
  eigen3mat BT_Ainv = B.transpose() * Ainv;
  Eigen::FullPivLU<eigen3mat> lu_of_BT_Ainv_B_plus_D(BT_Ainv * B + D);
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

  eigen3mat f_vec = eigen3mat::Zero(n_polys, 1);
  for (unsigned i_face = 0; i_face < this->n_faces && keys_; ++i_face)
  {
    for (unsigned i_polyface = 0; i_polyface < n_polyfaces; ++i_polyface)
    {
      eigen3mat uhat_vec = eigen3mat::Zero(this->n_faces * n_polyfaces, 1);
      uhat_vec(i_face * n_polyfaces + i_polyface, 0) = 1.0;
      eigen3mat u_vec = lu_of_BT_Ainv_B_plus_D.solve(
        M * f_vec + BT_Ainv * C * uhat_vec + E * uhat_vec);
      eigen3mat q_vec = lu_of_A.solve(B * u_vec - C * uhat_vec);
      eigen3mat jth_col =
        -1 * (C.transpose() * q_vec + E.transpose() * u_vec - H * uhat_vec);
      cell_mat.insert(
        cell_mat.end(), jth_col.data(), jth_col.data() + jth_col.size());
    }
  }
  if (keys_ & update_mat)
    model->solver->push_to_global_mat(row_nums, col_nums, cell_mat, ADD_VALUES);

  if (keys_ & update_rhs)
  {
    eigen3mat gD_vec(this->n_face_bases, 1);
    eigen3mat gN_vec = eigen3mat::Zero(n_polyfaces * this->n_faces, 1);
    eigen3mat uhat_vec = eigen3mat::Zero(n_polyfaces * this->n_faces, 1);
    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      if (this->BCs[i_face] == GenericCell<dim>::essential)
      {
        this->reinit_face_fe_vals(i_face);
        if (this->half_range_flag[i_face] == 0)
        {
          mtl::vec::dense_vector<double> gD_mtl;
          this->project_essential_BC_to_face(
            dirichlet_bc_func, *(this->the_face_basis), Face_Q_Weights, gD_mtl);
          for (unsigned i_dof = 0; i_dof < this->n_face_bases; ++i_dof)
            gD_vec(i_dof, 0) = gD_mtl[i_dof];
        }
        else
          std::cout << "There is something wrong dude!\n";
        uhat_vec.block(i_face * n_polyfaces, 0, n_polyfaces, 1) = gD_vec;
      }
      if (this->BCs[i_face] == GenericCell<dim>::flux_bc)
      {
        this->reinit_face_fe_vals(i_face);
        eigen3mat gN_vec_face(n_polyfaces, 1);
        if (this->half_range_flag[i_face] == 0)
        {
          mtl::vec::dense_vector<double> gN_mtl;
          this->project_flux_BC_to_face(
            Neumann_BC_func, *(this->the_face_basis), Face_Q_Weights, gN_mtl);
          for (unsigned i_dof = 0; i_dof < this->n_face_bases; ++i_dof)
            gN_vec_face(i_dof, 0) = gN_mtl[i_dof];
          gN_vec.block(i_face * n_polyfaces, 0, n_polyfaces, 1) = gN_vec_face;
        }
      }
    }

    eigen3mat f_vec(this->n_cell_bases, 1);
    mtl::vec::dense_vector<double> f_mtl;
    this->project_to_elem_basis(
      f_func, *(this->the_elem_basis), Q_Weights, f_mtl);
    for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
      f_vec(i_poly, 0) = f_mtl[i_poly];

    std::vector<double> rhs_col;
    eigen3mat u_vec = lu_of_BT_Ainv_B_plus_D.solve(
      M * f_vec + BT_Ainv * C * uhat_vec + E * uhat_vec);
    eigen3mat q_vec = lu_of_A.solve(B * u_vec - C * uhat_vec);
    eigen3mat jth_col =
      1 * (C.transpose() * q_vec + E.transpose() * u_vec - H * uhat_vec) -
      H2 * gN_vec;
    rhs_col.assign(jth_col.data(), jth_col.data() + jth_col.rows());
    /* Now, we assemble the calculated column. */
    model->solver->push_to_rhs_vec(row_nums, rhs_col, ADD_VALUES);
  }

  if (keys_ & update_sol)
  {
    std::vector<double> exact_uhat_vec;
    eigen3mat face_exact_uhat_vec(this->n_face_bases, 1);
    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      this->reinit_face_fe_vals(i_face);
      mtl::vec::dense_vector<double> face_exact_uhat_mtl;
      this->project_essential_BC_to_face(dirichlet_bc_func,
                                         *(this->the_face_basis),
                                         Face_Q_Weights,
                                         face_exact_uhat_mtl);
      for (unsigned i_dof = 0; i_dof < this->n_face_bases; ++i_dof)
        face_exact_uhat_vec(i_dof, 0) = face_exact_uhat_mtl[i_dof];
      exact_uhat_vec.insert(exact_uhat_vec.end(),
                            face_exact_uhat_vec.data(),
                            face_exact_uhat_vec.data() +
                              face_exact_uhat_vec.rows());
      /* Now, we assemble the exact solution. */
      model->solver->push_to_exact_sol(row_nums, exact_uhat_vec, INSERT_VALUES);
    }
  }
  wreck_it_Ralph(A);
  wreck_it_Ralph(B);
  wreck_it_Ralph(C);
  wreck_it_Ralph(D);
  wreck_it_Ralph(E);
  wreck_it_Ralph(H);
  wreck_it_Ralph(H2);
  wreck_it_Ralph(M);
}

template <int dim>
template <typename T>
double Diffusion<dim>::compute_internal_dofs(
  const double *const local_uhat_vec,
  eigen3mat &u_vec,
  eigen3mat &q_vec,
  const poly_space_basis<T, dim> &output_basis)
{
  unsigned n_polys = this->n_cell_bases;
  unsigned n_polyfaces = this->n_face_bases;
  const std::vector<double> &Q_Weights = this->elem_quad_bundle->get_weights();
  const std::vector<double> &Face_Q_Weights =
    this->face_quad_bundle->get_weights();
  /* Now we attach the fe_values to the current object. */
  this->reinit_cell_fe_vals();
  calculate_matrices();
  Eigen::FullPivLU<eigen3mat> lu_of_A(A);
  eigen3mat Ainv = lu_of_A.inverse();
  eigen3mat BT_Ainv = B.transpose() * Ainv;
  Eigen::FullPivLU<eigen3mat> lu_of_BT_Ainv_B_plus_D(BT_Ainv * B + D);

  eigen3mat exact_f_vec(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> exact_f_mtl;
  this->project_to_elem_basis(
    f_func, *(this->the_elem_basis), Q_Weights, exact_f_mtl);
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    exact_f_vec(i_poly, 0) = exact_f_mtl[i_poly];
  eigen3mat solved_uhat_vec = eigen3mat::Zero(n_polyfaces * this->n_faces, 1);

  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    if (this->dofs_ID_in_this_rank[i_face].size() == 0)
    {
      eigen3mat face_uhat_vec(this->n_face_bases, 1);
      this->reinit_face_fe_vals(i_face);
      mtl::vec::dense_vector<double> face_uhat_mtl;
      this->project_essential_BC_to_face(dirichlet_bc_func,
                                         *(this->the_face_basis),
                                         Face_Q_Weights,
                                         face_uhat_mtl);
      for (unsigned i_dof = 0; i_dof < this->n_face_bases; ++i_dof)
        face_uhat_vec(i_dof, 0) = face_uhat_mtl[i_dof];
      solved_uhat_vec.block(i_face * n_polyfaces, 0, n_polyfaces, 1) =
        face_uhat_vec;
    }
    else
    {
      for (unsigned i_polyface = 0; i_polyface < n_polyfaces; ++i_polyface)
      {
        int global_dof_number =
          this->dofs_ID_in_this_rank[i_face][0] * n_polyfaces + i_polyface;
        solved_uhat_vec(i_face * n_polyfaces + i_polyface, 0) =
          local_uhat_vec[global_dof_number];
      }
    }
  }
  u_vec = lu_of_BT_Ainv_B_plus_D.solve(
    M * exact_f_vec + BT_Ainv * C * solved_uhat_vec + E * solved_uhat_vec);
  q_vec = B * u_vec - C * solved_uhat_vec;
  q_vec = lu_of_A.solve(q_vec);

  /*
   * Here, we use a postprocessing technique to obtain a higher order
   * approximation to u.
   */
  calculate_postprocess_matrices();
  eigen3mat RHS_vec_of_ustar = -DB2 * q_vec;
  DM_star(0, 0) = 1;
  RHS_vec_of_ustar(0, 0) = u_vec(0);
  ustar = DM_star.ldlt().solve(RHS_vec_of_ustar);

  eigen3mat nodal_u = output_basis.get_dof_vals_at_quads(u_vec);
  eigen3mat nodal_q(dim * n_polys, 1);
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    nodal_q.block(i_dim * n_polys, 0, n_polys, 1) =
      output_basis.get_dof_vals_at_quads(
        q_vec.block(i_dim * n_polys, 0, n_polys, 1));
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

  wreck_it_Ralph(A);
  wreck_it_Ralph(B);
  wreck_it_Ralph(C);
  wreck_it_Ralph(D);
  wreck_it_Ralph(E);
  wreck_it_Ralph(H);
  wreck_it_Ralph(H2);
  wreck_it_Ralph(M);
  wreck_it_Ralph(DM_star);
  wreck_it_Ralph(DB2);
  return 0.;
}

template <int dim>
void Diffusion<dim>::internal_vars_errors(const eigen3mat &u_vec,
                                          const eigen3mat &q_vec,
                                          double &u_error,
                                          double &q_error)
{
  double error_q3 = this->get_error_in_cell(q_func, q_vec);
  double error_u3 = this->get_error_in_cell(u_func, u_vec);
  error_u3 = this->get_postprocessed_error_in_cell(u_func, ustar);

  u_error += error_u3;
  q_error += error_q3;
}

template <int dim>
void Diffusion<dim>::ready_for_next_iteration()
{
}

template <int dim>
void Diffusion<dim>::ready_for_next_time_step()
{
}

template <int dim>
double Diffusion<dim>::get_postprocessed_error_in_cell(
  const TimeFunction<dim, double> &func,
  const Eigen::MatrixXd &input_vector,
  const double &time)
{
  double error = 0;
  const std::vector<dealii::Point<dim> > &points_loc =
    this->cell_quad_fe_vals->get_quadrature_points();
  const std::vector<double> &JxWs = this->cell_quad_fe_vals->get_JxW_values();
  assert(points_loc.size() == JxWs.size());
  assert(input_vector.rows() == model->manager->postprocess_cell_basis.n_polys);
  Eigen::MatrixXd values_at_Nodes =
    model->manager->postprocess_cell_basis.get_dof_vals_at_quads(input_vector);
  for (unsigned i_point = 0; i_point < JxWs.size(); ++i_point)
  {
    error += (func.value(points_loc[i_point], points_loc[i_point], time) -
              values_at_Nodes(i_point, 0)) *
             (func.value(points_loc[i_point], points_loc[i_point], time) -
              values_at_Nodes(i_point, 0)) *
             JxWs[i_point];
  }
  return error;
}
