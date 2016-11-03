#include "explicit_gn_dispersive.hpp"
#include "support_classes.hpp"

/*
ResultPacket::ResultPacket(eigen3mat *last_q_, eigen3mat *last_qhat_)
{
  last_q = last_q_;
  last_qhat = last_qhat_;
}
*/

template <int dim>
explicit_gn_dispersive_g_h_grad_zeta_class<dim, dealii::Tensor<1, dim> >
  explicit_gn_dispersive<dim>::g_h_grad_zeta_func{};

template <int dim>
explicit_nswe_grad_b_func_class<dim, dealii::Tensor<1, dim> >
  explicit_gn_dispersive<dim>::explicit_nswe_grad_b_func{};

template <int dim>
explicit_gn_dispersive_hVinf_t_class<dim, dealii::Tensor<1, dim> >
  explicit_gn_dispersive<dim>::hVinf_t_func{};

template <int dim>
explicit_gn_dispersive_grad_grad_b_class<dim, dealii::Tensor<2, dim> >
  explicit_gn_dispersive<dim>::grad_grad_b_func{};

template <int dim>
explicit_gn_dispersive_qis_class<dim, dealii::Tensor<1, dim + 1> >
  explicit_gn_dispersive<dim>::explicit_gn_dispersive_qs{};

template <int dim>
explicit_gn_dispersive_L_class<dim, dealii::Tensor<1, dim> >
  explicit_gn_dispersive<dim>::explicit_gn_dispersive_L{};

template <int dim>
explicit_gn_dispersive_W1_class<dim, dealii::Tensor<1, dim> >
  explicit_gn_dispersive<dim>::explicit_gn_dispersive_W1{};

template <int dim>
explicit_gn_dispersive_W2_class<dim, double>
  explicit_gn_dispersive<dim>::explicit_gn_dispersive_W2{};

template <int dim>
solver_options explicit_gn_dispersive<dim>::required_solver_options()
{
  return solver_options::default_option;
}

template <int dim>
solver_type explicit_gn_dispersive<dim>::required_solver_type()
{
  return solver_type::implicit_petsc_aij;
}

template <int dim>
unsigned explicit_gn_dispersive<dim>::get_num_dofs_per_node()
{
  return dim;
}

/**
 * The move constrcutor of the derived class should call the move
 * constructor of the base class using std::move. Otherwise the copy
 * constructor will be called.
 */
template <int dim>
explicit_gn_dispersive<dim>::explicit_gn_dispersive(
  explicit_gn_dispersive &&inp_cell) noexcept
  : GenericCell<dim>(std::move(inp_cell)),
    taus(std::move(inp_cell.taus)),
    model(inp_cell.model),
    time_integrator(model->time_integrator),
    last_step_q(std::move(inp_cell.last_step_q)),
    last_stage_q(std::move(inp_cell.last_stage_q)),
    ki_s(std::move(inp_cell.ki_s))
{
}

template <int dim>
explicit_gn_dispersive<dim>::explicit_gn_dispersive(
  typename GenericCell<dim>::dealiiCell &inp_cell,
  const unsigned &id_num_,
  const unsigned &poly_order_,
  hdg_model_with_explicit_rk<dim, explicit_gn_dispersive> *model_)
  : GenericCell<dim>(inp_cell, id_num_, poly_order_),
    taus(this->n_faces, -10.0),
    model(model_),
    time_integrator(model_->time_integrator),
    last_step_q((dim + 1) * this->n_cell_bases, 1),
    last_stage_q((dim + 1) * this->n_cell_bases, 1),
    ki_s(time_integrator->order,
         eigen3mat::Zero((dim + 1) * this->n_cell_bases, 1))
{
}

template <int dim>
void explicit_gn_dispersive<dim>::assign_BCs(
  const bool &at_boundary,
  const unsigned &i_face,
  const dealii::Point<dim> &face_center)
{
  // Green-Naghdi first example: Flat bottom
  if (at_boundary && (face_center[0] < -9.99 || face_center[0] > 9.99))
  {
    this->BCs[i_face] = GenericCell<dim>::BC::essential;
    this->dof_names_on_faces[i_face].resize(dim, 1);
    this->dof_names_on_faces[i_face][1] = 0;
  }
  /*
  else if (at_boundary)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::essential;
    this->dof_names_on_faces[i_face].resize(dim, 1);
    this->dof_names_on_faces[i_face][1] = 0;
  }
  */
  /*
  else if (at_boundary)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::solid_wall;
    this->dof_names_on_faces[i_face].resize(dim, 1);
    this->dof_names_on_faces[i_face][1] = 0;
  }
  */
  // End of Green-Naghdi first example: Flat bottom
  // Dissertation example 2
  /*
  if (at_boundary && face_center[0] < -5 + 1.e-6)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::essential;
    this->dof_names_on_faces[i_face].resize(dim, 1);
  }
  else if (at_boundary)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::solid_wall;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  */
  // End of dissertation example 2
  else
  {
    this->BCs[i_face] = GenericCell<dim>::BC::not_set;
    this->dof_names_on_faces[i_face].resize(dim, 1);
    this->dof_names_on_faces[i_face][1] = 0;
  }
}

template <int dim>
explicit_gn_dispersive<dim>::~explicit_gn_dispersive<dim>()
{
}

template <int dim>
void explicit_gn_dispersive<dim>::assign_initial_data()
{
}

template <int dim>
void explicit_gn_dispersive<dim>::set_previous_step_results(
  eigen3mat *last_step_q_)
{
  last_step_q = std::move(*last_step_q_);
}

template <int dim>
eigen3mat *explicit_gn_dispersive<dim>::get_previous_step_results()
{
  return &last_step_q;
}

/*
 * Here we only need the last step qhat (not the last stage qhat).
 * Actually the last step h is enough. Even though q = (h,hv1,hv2),
 * we do not use hv1 and hv2 in this element and we only use h from
 * qhat.
template <int dim>
ResultPacket explicit_gn_dispersive<dim>::get_previous_step_results1()
{
  ResultPacket result1(&last_step_q, &last_step_qhat);
  return result1;
}

template <int dim>
void explicit_gn_dispersive<dim>::set_previous_step_results1(
  ResultPacket result_)
{
  last_step_q = std::move(*(result_.last_q));
  last_step_qhat = std::move(*(result_.last_qhat));
  last_stage_q = last_step_q;
}
*/

template <int dim>
void explicit_gn_dispersive<dim>::calculate_matrices()
{
  static_assert(dim == 2, "The problem dimension should be 2.");
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
  A001 = eigen3mat::Zero(dim * n_cell_bases, dim * n_cell_bases);
  A01 = eigen3mat::Zero(dim * n_cell_bases, dim * n_cell_bases);
  A02 = eigen3mat::Zero(n_cell_bases, n_cell_bases);
  A03 = eigen3mat::Zero(dim * n_cell_bases, n_cell_bases);
  B01T = eigen3mat::Zero(n_cell_bases, dim * n_cell_bases);
  B02 = eigen3mat::Zero(dim * n_cell_bases, n_cell_bases);
  B03T = eigen3mat::Zero(dim * n_cell_bases, dim * n_cell_bases);
  D01 = eigen3mat::Zero(dim * n_cell_bases, dim * n_cell_bases);
  C02 = eigen3mat::Zero(dim * n_cell_bases, dim * n_faces * n_face_bases);
  C01 = eigen3mat::Zero(n_cell_bases, dim * n_faces * n_face_bases);
  C03T = eigen3mat::Zero(dim * n_faces * n_face_bases, n_cell_bases);
  E01 =
    eigen3mat::Zero(dim * n_faces * n_face_bases, dim * n_faces * n_face_bases);
  C04T = eigen3mat::Zero(dim * n_faces * n_face_bases, dim * n_cell_bases);
  L01 = eigen3mat::Zero(dim * n_cell_bases, 1);
  L10 = eigen3mat::Zero(dim * n_cell_bases, 1);
  L11 = eigen3mat::Zero(dim * n_cell_bases, 1);
  L12 = eigen3mat::Zero(dim * n_cell_bases, 1);
  L21 = eigen3mat::Zero(dim * n_cell_bases, 1);
  E31 =
    eigen3mat::Zero(dim * n_faces * n_face_bases, dim * n_faces * n_face_bases);
  C34T = eigen3mat::Zero(dim * n_faces * n_face_bases, dim * n_cell_bases);
  L31 = eigen3mat::Zero(dim * n_faces * n_face_bases, 1);

  /*
  {
    const std::vector<double> &cell_quad_weights =
      this->elem_quad_bundle->get_weights();
    mtl::vec::dense_vector<dealii::Tensor<1, dim + 1> > last_stage_qs_mtl;
    this->project_to_elem_basis(explicit_gn_dispersive_qs,
                                *(this->the_elem_basis),
                                cell_quad_weights,
                                last_stage_qs_mtl,
                                time_integrator->get_current_time());
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
        last_stage_q(i_nswe_dim * this->n_cell_bases + i_poly) =
          last_stage_qs_mtl[i_poly][i_nswe_dim];
  }
  */

  /*
   * On quadrature points, we need \partial_x V. But we have h, hv1, hv2.
   * Since, we use modal basis, we cannot say v1 = hv1 / h. We have to first
   * transfer everything to points and then obtain the following:
   *
   *        (hV)_x  -  (hV)/h * h_x
   * V_x = --------------------------
   *                  h
   *
   * In the above relation, we have h and hV. We need (hV)_x and h_x.
   */
  eigen3mat L11_arg = eigen3mat::Zero(n_cell_bases, 1);
  {
    const std::vector<double> &cell_quad_weights =
      this->elem_quad_bundle->get_weights();
    eigen3mat grad_NT, NT;
    unsigned n_int_pts = this->elem_quad_bundle->size();
    for (unsigned i_quad = 0; i_quad < n_int_pts; ++i_quad)
    {
      grad_NT = eigen3mat::Zero(dim, n_cell_bases);
      NT = this->the_elem_basis->get_func_vals_at_iquad(i_quad);
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        dealii::Tensor<1, dim> grad_N_at_point = grad_N_x[i_poly][i_quad];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
          grad_NT(i_dim, i_poly) = grad_N_at_point[i_dim];
      }

      double h_h = (NT * last_stage_q.block(0, 0, n_cell_bases, 1))(0, 0);
      double hv1_h =
        (NT * last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double hv2_h =
        (NT * last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      Eigen::Matrix<double, 2, 1> grad_h_h =
        grad_NT * last_stage_q.block(0, 0, n_cell_bases, 1);
      Eigen::Matrix<double, 2, 1> grad_hv1_h =
        grad_NT * last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1);
      Eigen::Matrix<double, 2, 1> grad_hv2_h =
        grad_NT * last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1);
      double v1 = hv1_h / h_h;
      double v2 = hv2_h / h_h;
      double v1_x = (grad_hv1_h(0, 0) - v1 * grad_h_h(0, 0)) / h_h;
      double v1_y = (grad_hv1_h(1, 0) - v1 * grad_h_h(1, 0)) / h_h;
      double v2_x = (grad_hv2_h(0, 0) - v2 * grad_h_h(0, 0)) / h_h;
      double v2_y = (grad_hv2_h(1, 0) - v2 * grad_h_h(1, 0)) / h_h;

      Eigen::Matrix<double, dim, 1> grad_b_at_quad;
      dealii::Tensor<1, dim> grad_b_at_quad_tensor =
        explicit_nswe_grad_b_func.value(quad_pt_locs[i_quad],
                                        quad_pt_locs[i_quad]);
      dealii::Tensor<2, dim> grad_grad_b =
        grad_grad_b_func.value(quad_pt_locs[i_quad], quad_pt_locs[i_quad]);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        grad_b_at_quad(i_dim, 0) = grad_b_at_quad_tensor[i_dim];

      double b_11 = grad_grad_b[0][0];
      double b_12 = grad_grad_b[1][0];
      double b_21 = grad_grad_b[0][1];
      double b_22 = grad_grad_b[1][1];

      double L11_val_at_iquad =
        2. / 3. * h_h * h_h * h_h *
          (-v1_x * v2_y + v2_x * v1_y + (v1_x + v2_y) * (v1_x + v2_y)) +
        1. / 2. * h_h * h_h *
          (v1 * v1 * b_11 + v1 * v2 * b_12 + v2 * v1 * b_21 + v2 * v2 * b_22);
      L11_arg += cell_quad_weights[i_quad] * L11_val_at_iquad * NT.transpose();
    }
  }

  {
    eigen3mat grad_NT, div_N, NT, N_vec;
    unsigned n_int_pts = this->elem_quad_bundle->size();
    for (unsigned i_quad = 0; i_quad < n_int_pts; ++i_quad)
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
          div_N(i_dim * n_cell_bases + i_poly) = grad_N_at_point[i_dim];
          grad_NT(i_dim, i_poly) = grad_N_at_point[i_dim];
        }
      }

      double h_h = (NT * last_stage_q.block(0, 0, n_cell_bases, 1))(0, 0);
      double hv1_h =
        (NT * last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double hv2_h =
        (NT * last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      Eigen::Matrix<double, 2, 1> grad_h_h =
        grad_NT * last_stage_q.block(0, 0, n_cell_bases, 1);
      Eigen::Matrix<double, 2, 1> grad_hv1_h =
        grad_NT * last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1);
      Eigen::Matrix<double, 2, 1> grad_hv2_h =
        grad_NT * last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1);
      double v1 = hv1_h / h_h;
      double v2 = hv2_h / h_h;
      double v1_x = (grad_hv1_h(0, 0) - v1 * grad_h_h(0, 0)) / h_h;
      double v1_y = (grad_hv1_h(1, 0) - v1 * grad_h_h(1, 0)) / h_h;
      double v2_x = (grad_hv2_h(0, 0) - v2 * grad_h_h(0, 0)) / h_h;
      double v2_y = (grad_hv2_h(1, 0) - v2 * grad_h_h(1, 0)) / h_h;

      Eigen::Matrix<double, dim, 1> L01_at_quad, grad_b_at_quad, L10_at_quad;
      dealii::Tensor<1, dim> L01_at_quad_tensor =
        explicit_gn_dispersive_L.value(quad_pt_locs[i_quad],
                                       quad_pt_locs[i_quad],
                                       time_integrator->get_current_time() +
                                         time_integrator->get_cih());
      dealii::Tensor<1, dim> grad_b_at_quad_tensor =
        explicit_nswe_grad_b_func.value(quad_pt_locs[i_quad],
                                        quad_pt_locs[i_quad]);
      dealii::Tensor<1, dim> L10_at_quad_tensor = g_h_grad_zeta_func.value(
        quad_pt_locs[i_quad],
        quad_pt_locs[i_quad],
        time_integrator->get_current_time() + time_integrator->get_cih());
      dealii::Tensor<2, dim> grad_grad_b =
        grad_grad_b_func.value(quad_pt_locs[i_quad], quad_pt_locs[i_quad]);

      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        L01_at_quad(i_dim, 0) = L01_at_quad_tensor[i_dim];
        L10_at_quad(i_dim, 0) = L10_at_quad_tensor[i_dim];
        grad_b_at_quad(i_dim, 0) = grad_b_at_quad_tensor[i_dim];
      }

      /*
      std::cout << 4 * cos(4. * quad_pt_locs[i_quad][0]) << grad_h_h[0]
                << std::endl;
      std::cout << 5. + sin(4. * quad_pt_locs[i_quad][0]) - h_h << std::endl;
      */

      double b_11 = grad_grad_b[0][0];
      double b_12 = grad_grad_b[1][0];
      double b_21 = grad_grad_b[0][1];
      double b_22 = grad_grad_b[1][1];
      Eigen::Matrix<double, 2, 2> grad_grad_b_at_quad;
      grad_grad_b_at_quad << b_11, b_12, b_21, b_22;

      A001 += cell_JxW[i_quad] * N_vec * N_vec.transpose();
      A01 += cell_JxW[i_quad] * N_vec * N_vec.transpose() +
             cell_JxW[i_quad] * alpha * N_vec * grad_b_at_quad *
               grad_b_at_quad.transpose() * N_vec.transpose();
      A02 += cell_JxW[i_quad] / h_h / h_h / h_h * NT.transpose() * NT;
      A03 += cell_JxW[i_quad] / h_h * N_vec * grad_b_at_quad * NT;
      B01T += cell_JxW[i_quad] / h_h * grad_NT.transpose() * N_vec.transpose();
      B02 += cell_JxW[i_quad] * N_vec * grad_NT;
      B03T += cell_JxW[i_quad] * h_h * div_N * grad_b_at_quad.transpose() *
              N_vec.transpose();
      L01 += cell_JxW[i_quad] * N_vec * L01_at_quad;

      L10 += cell_JxW[i_quad] / alpha * gravity * h_h * N_vec *
             (grad_h_h + grad_b_at_quad);
      //      L10 += cell_JxW[i_quad] * N_vec * L10_at_quad;

      L11 += cell_JxW[i_quad] *
             (-2. / 3. * h_h * h_h * h_h *
                (-v1_x * v2_y + v2_x * v1_y + (v1_x + v2_y) * (v1_x + v2_y)) -
              1. / 2. * h_h * h_h * (v1 * v1 * b_11 + v1 * v2 * b_12 +
                                     v2 * v1 * b_21 + v2 * v2 * b_22)) *
             div_N;
      //      L11 += cell_JxW[i_quad] * N_vec * grad_NT * L11_arg;
      L12 += cell_JxW[i_quad] * (h_h * h_h * (-v1_x * v2_y + v2_x * v1_y +
                                              (v1_x + v2_y) * (v1_x + v2_y)) +
                                 h_h * (v1 * v1 * b_11 + v1 * v2 * b_12 +
                                        v2 * v1 * b_21 + v2 * v2 * b_22)) *
             N_vec * grad_b_at_quad;
    }
  }

  eigen3mat normal(dim, 1);
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    eigen3mat C01_on_face = eigen3mat::Zero(n_cell_bases, dim * n_face_bases);
    eigen3mat C02_on_face =
      eigen3mat::Zero(dim * n_cell_bases, dim * n_face_bases);
    eigen3mat C03T_on_face = eigen3mat::Zero(dim * n_face_bases, n_cell_bases);
    eigen3mat E01_on_face =
      eigen3mat::Zero(dim * n_face_bases, dim * n_face_bases);
    eigen3mat C04T_on_face =
      eigen3mat::Zero(dim * n_face_bases, dim * n_cell_bases);
    eigen3mat C34T_on_face =
      eigen3mat::Zero(dim * n_face_bases, dim * n_cell_bases);
    eigen3mat E31_on_face =
      eigen3mat::Zero(dim * n_face_bases, dim * n_face_bases);
    eigen3mat L31_on_face = eigen3mat::Zero(dim * n_face_bases, 1);
    std::vector<dealii::DerivativeForm<1, dim, dim> > face_d_forms =
      this->face_quad_fe_vals->get_inverse_jacobians();
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
    std::vector<dealii::Point<dim> > face_quad_pt_locs =
      this->face_quad_fe_vals->get_quadrature_points();
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
      std::vector<dealii::Tensor<1, dim> > grad_N_at_face_quad =
        this->the_elem_basis->grad(projected_quad_points[i_face_quad]);
      Eigen::MatrixXd grad_NxT_at_face_quads(dim, this->n_cell_bases);
      for (unsigned i_cell_basis = 0; i_cell_basis < this->n_cell_bases;
           ++i_cell_basis)
      {
        dealii::Tensor<1, dim> grad_Nx_at_face_iquad =
          grad_N_at_face_quad[i_cell_basis] *
          static_cast<dealii::Tensor<2, dim> >(face_d_forms[i_face_quad]);
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
          grad_NxT_at_face_quads(i_dim, i_cell_basis) =
            grad_Nx_at_face_iquad[i_dim];
      }
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
      // Now, we get the value of h, hv1, hv2 at the current quad point.
      std::vector<double> qs_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
        qs_at_iquad[i_nswe_dim] =
          (Nj.transpose() *
           last_stage_q.block(i_nswe_dim * n_cell_bases, 0, n_cell_bases, 1))(
            0, 0);
      double h_h = qs_at_iquad[0];
      double hv1_h = qs_at_iquad[1];
      double hv2_h = qs_at_iquad[2];
      Eigen::Matrix<double, 2, 1> grad_h_h =
        grad_NxT_at_face_quads * last_stage_q.block(0, 0, n_cell_bases, 1);
      Eigen::Matrix<double, 2, 1> grad_hv1_h =
        grad_NxT_at_face_quads *
        last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1);
      Eigen::Matrix<double, 2, 1> grad_hv2_h =
        grad_NxT_at_face_quads *
        last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1);
      double v1 = hv1_h / h_h;
      double v2 = hv2_h / h_h;
      double v1_x = (grad_hv1_h(0, 0) - v1 * grad_h_h(0, 0)) / h_h;
      double v1_y = (grad_hv1_h(1, 0) - v1 * grad_h_h(1, 0)) / h_h;
      double v2_x = (grad_hv2_h(0, 0) - v2 * grad_h_h(0, 0)) / h_h;
      double v2_y = (grad_hv2_h(1, 0) - v2 * grad_h_h(1, 0)) / h_h;

      Eigen::Matrix<double, dim, 1> grad_b_at_quad;
      dealii::Tensor<1, dim> grad_b_at_quad_tensor =
        explicit_nswe_grad_b_func.value(face_quad_pt_locs[i_face_quad],
                                        face_quad_pt_locs[i_face_quad]);
      dealii::Tensor<2, dim> grad_grad_b = grad_grad_b_func.value(
        face_quad_pt_locs[i_face_quad], face_quad_pt_locs[i_face_quad]);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        grad_b_at_quad(i_dim, 0) = grad_b_at_quad_tensor[i_dim];
      double b_11 = grad_grad_b[0][0];
      double b_12 = grad_grad_b[1][0];
      double b_21 = grad_grad_b[0][1];
      double b_22 = grad_grad_b[1][1];
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_face_quad](i_dim);
      dealii::Tensor<1, dim> hVinf_t_tensor = hVinf_t_func.value(
        face_quad_pt_locs[i_face_quad],
        face_quad_pt_locs[i_face_quad],
        time_integrator->get_current_time() + time_integrator->get_cih());
      Eigen::Matrix<double, dim, 1> hVinf_t_at_quad;
      hVinf_t_at_quad(0, 0) = hVinf_t_tensor[0];
      hVinf_t_at_quad(1, 0) = hVinf_t_tensor[1];

      Eigen::Matrix<double, 2, 1> W1_val_vec;
      dealii::Tensor<1, dim> W1_vec_tensor = explicit_gn_dispersive_W1.value(
        face_quad_pt_locs[i_face_quad],
        face_quad_pt_locs[i_face_quad],
        time_integrator->get_current_time() + time_integrator->get_cih());
      W1_val_vec(0, 0) = W1_vec_tensor[0];
      W1_val_vec(1, 0) = W1_vec_tensor[1];

      double tau_on_face = -20.;

      D01 += tau_on_face * face_JxW[i_face_quad] * N_vec * N_vec.transpose();
      C01_on_face +=
        face_JxW[i_face_quad] / h_h * Nj * normal.transpose() * NT_face_vec;
      C02_on_face += tau_on_face * face_JxW[i_face_quad] * N_vec * NT_face_vec +
                     3. / 2. * face_JxW[i_face_quad] * h_h * N_vec * normal *
                       grad_b_at_quad.transpose() * NT_face_vec;
      L21 += face_JxW[i_face_quad] *
             (2. / 3. * h_h * h_h * h_h *
                (-v1_x * v2_y + v2_x * v1_y + (v1_x + v2_y) * (v1_x + v2_y)) +
              1. / 2. * h_h * h_h * (v1 * v1 * b_11 + v1 * v2 * b_12 +
                                     v2 * v1 * b_21 + v2 * v2 * b_22)) *
             N_vec * normal;
      if (this->BCs[i_face] == GenericCell<dim>::BC::not_set)
      {
        C03T_on_face += face_JxW[i_face_quad] * NT_face_vec.transpose() *
                        normal * Nj.transpose();
        C04T_on_face += tau_on_face * face_JxW[i_face_quad] *
                        NT_face_vec.transpose() * N_vec.transpose();
        E01_on_face += tau_on_face * face_JxW[i_face_quad] *
                       NT_face_vec.transpose() * NT_face_vec;
      }
      if (this->BCs[i_face] == GenericCell<dim>::BC::essential)
      {
        E31_on_face -=
          face_JxW[i_face_quad] * NT_face_vec.transpose() * NT_face_vec;
        L31_on_face +=
          face_JxW[i_face_quad] * NT_face_vec.transpose() *
          (1. / alpha * gravity * h_h * (grad_h_h + grad_b_at_quad) -
           hVinf_t_at_quad);
      }
      if (this->BCs[i_face] == GenericCell<dim>::BC::solid_wall)
      {
        C34T_on_face +=
          face_JxW[i_face_quad] * NT_face_vec.transpose() * N_vec.transpose() -
          face_JxW[i_face_quad] * NT_face_vec.transpose() * normal *
            normal.transpose() * N_vec.transpose();
        E31_on_face +=
          face_JxW[i_face_quad] * NT_face_vec.transpose() * NT_face_vec;
        L31_on_face += face_JxW[i_face_quad] / alpha * gravity * h_h *
                       NT_face_vec.transpose() * normal * normal.transpose() *
                       (grad_h_h + grad_b_at_quad);
      }
      if (this->BCs[i_face] == GenericCell<dim>::BC::in_out_BC)
      {
        double vn = v1 * normal(0, 0) + v2 * normal(1, 0);
        double vn_pls = (vn > 0) ? vn : 0.;
        double vn_neg = (vn < 0) ? vn : 0.;
        double vn_abs = std::abs(vn);
        /*
         * This is correct but using this type of boundary condition is
         * dangerous for stability ! So skip this part and use the one
         * which is explained downstairs.
        C34T_on_face += face_JxW[i_face_quad] * vn_pls *
                        NT_face_vec.transpose() * N_vec.transpose();
        E31_on_face += face_JxW[i_face_quad] * vn_abs *
                       NT_face_vec.transpose() * NT_face_vec;
        L31_on_face +=
          vn_neg * face_JxW[i_face_quad] * NT_face_vec.transpose() *
          (1. / alpha * gravity * h_h * (grad_h_h + grad_b_at_quad) -
           Vinf_t_at_quad);
        */
        E31_on_face += face_JxW[i_face_quad] * vn_abs *
                       NT_face_vec.transpose() * NT_face_vec;
        L31_on_face +=
          vn_neg * face_JxW[i_face_quad] * NT_face_vec.transpose() *
            (1. / alpha * gravity * h_h * (grad_h_h + grad_b_at_quad) -
             hVinf_t_at_quad) -
          vn_pls * face_JxW[i_face_quad] * NT_face_vec.transpose() * 1. /
            alpha * gravity * h_h * (grad_h_h + grad_b_at_quad);
      }
    }
    C02.block(
      0, i_face * dim * n_face_bases, dim * n_cell_bases, dim * n_face_bases) =
      C02_on_face;
    C01.block(
      0, i_face * dim * n_face_bases, n_cell_bases, dim * n_face_bases) =
      C01_on_face;
    C03T.block(
      i_face * dim * n_face_bases, 0, dim * n_face_bases, n_cell_bases) =
      C03T_on_face;
    C04T.block(
      i_face * dim * n_face_bases, 0, dim * n_face_bases, dim * n_cell_bases) =
      C04T_on_face;
    E01.block(i_face * dim * n_face_bases,
              i_face * dim * n_face_bases,
              dim * n_face_bases,
              dim * n_face_bases) = E01_on_face;
    C34T.block(
      i_face * dim * n_face_bases, 0, dim * n_face_bases, dim * n_cell_bases) =
      C34T_on_face;
    E31.block(i_face * dim * n_face_bases,
              i_face * dim * n_face_bases,
              dim * n_face_bases,
              dim * n_face_bases) = E31_on_face;
    L31.block(i_face * dim * n_face_bases, 0, dim * n_face_bases, 1) =
      L31_on_face;
  }
}

template <int dim>
void explicit_gn_dispersive<dim>::assemble_globals(
  const solver_update_keys &keys_)
{
  const std::vector<double> &quad_weights =
    this->elem_quad_bundle->get_weights();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();

  this->reinit_cell_fe_vals();
  /*
  {
    const std::vector<double> &cell_quad_weights =
      this->elem_quad_bundle->get_weights();
    mtl::vec::dense_vector<dealii::Tensor<1, dim + 1> > last_step_qs_mtl;
    this->project_to_elem_basis(explicit_gn_dispersive_qs,
                                *(this->the_elem_basis),
                                cell_quad_weights,
                                last_step_qs_mtl,
                                time_integrator->get_current_time() +
                                  time_integrator->get_h() / 2.);
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
        last_step_q(i_nswe_dim * this->n_cell_bases + i_poly) =
          last_step_qs_mtl[i_poly][i_nswe_dim];
  }
  */

  last_stage_q = last_step_q + time_integrator->get_sum_h_aij_kj(ki_s);
  //  last_stage_q = last_step_q;
  calculate_matrices();
  eigen3mat A2_inv = std::move(A02.inverse());
  eigen3mat mat1 = std::move(
    A01 - alpha / 2. * B03T +
    (alpha / 2. * A03 + alpha / 3. * B02) * A2_inv * B01T - alpha / 3. * D01);
  eigen3mat mat2 = std::move(
    (alpha / 2. * A03 + alpha / 3. * B02) * A2_inv * C01 - alpha / 3. * C02);
  Eigen::PartialPivLU<eigen3mat> mat1_lu = std::move(mat1.partialPivLu());

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

  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    for (unsigned i_dof = 0; i_dof < dim; ++i_dof)
    {
      for (unsigned i_polyface = 0; i_polyface < this->n_face_bases;
           ++i_polyface)
      {
        unsigned i_num = i_face * dim * this->n_face_bases +
                         i_dof * this->n_face_bases + i_polyface;
        eigen3mat w1_hat = std::move(
          eigen3mat::Zero(this->n_faces * dim * this->n_face_bases, 1));
        w1_hat(i_num, 0) = 1.0;
        eigen3mat w1 = mat1_lu.solve(mat2 * w1_hat);
        eigen3mat w2 = std::move(A2_inv * (-B01T * w1 + C01 * w1_hat));
        eigen3mat jth_col =
          (C03T * w2 + (C04T + C34T) * w1 - (E01 + E31) * w1_hat);
        cell_mat.insert(
          cell_mat.end(), jth_col.data(), jth_col.data() + jth_col.rows());
      }
    }
  }
  if (keys_ & update_mat)
    model->solver->push_to_global_mat(row_nums, col_nums, cell_mat, ADD_VALUES);

  if (keys_ & update_rhs)
  {
    eigen3mat w1_hat =
      eigen3mat::Zero(this->n_faces * dim * this->n_face_bases, 1);
    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      unsigned n_dofs = this->dofs_ID_in_all_ranks[i_face].size();
      if (this->BCs[i_face] == GenericCell<dim>::essential)
      {
        mtl::vec::dense_vector<dealii::Tensor<1, dim> > w1_hat_mtl;
        this->reinit_face_fe_vals(i_face);
        this->project_essential_BC_to_face(explicit_gn_dispersive_W1,
                                           *(this->the_face_basis),
                                           face_quad_weights,
                                           w1_hat_mtl,
                                           time_integrator->get_current_time() +
                                             time_integrator->get_cih());
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
          for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
            w1_hat((i_face * dim + i_dim) * this->n_face_bases + i_poly) =
              w1_hat_mtl[i_poly][i_dim];
      }
      if (this->BCs[i_face] == GenericCell<dim>::flux_bc)
      {
        for (unsigned i_dof = 0; i_dof < n_dofs; ++i_dof)
        {
        }
      }
    }

    w1_hat = eigen3mat::Zero(this->n_faces * dim * this->n_face_bases, 1);

    //    eigen3mat w1 = mat1_lu.solve(L01 + L10 + L11 + L12 + L21 + mat2 *
    //    w1_hat);

    eigen3mat w1 = mat1_lu.solve(L01 + L10 + mat2 * w1_hat);

    eigen3mat w2 = std::move(A2_inv * (-B01T * w1 + C01 * w1_hat));
    eigen3mat jth_col_vec =
      -(C03T * w2 + (C04T + C34T) * w1 - (E01 + E31) * w1_hat) + L31;
    std::vector<double> rhs_col(jth_col_vec.data(),
                                jth_col_vec.data() + jth_col_vec.rows());
    model->solver->push_to_rhs_vec(row_nums, rhs_col, ADD_VALUES);
  }

  wreck_it_Ralph(A001);
  wreck_it_Ralph(A01);
  wreck_it_Ralph(A02);
  wreck_it_Ralph(A03);
  wreck_it_Ralph(B01T);
  wreck_it_Ralph(B02);
  wreck_it_Ralph(B03T);
  wreck_it_Ralph(C03T);
  wreck_it_Ralph(C04T);
  wreck_it_Ralph(D01);
  wreck_it_Ralph(C02);
  wreck_it_Ralph(C01);
  wreck_it_Ralph(E01);
  wreck_it_Ralph(L01);
  wreck_it_Ralph(L10);
  wreck_it_Ralph(L11);
  wreck_it_Ralph(L12);
  wreck_it_Ralph(L21);
  wreck_it_Ralph(C34T);
  wreck_it_Ralph(E31);
  wreck_it_Ralph(L31);
}

template <int dim>
template <typename T>
double explicit_gn_dispersive<dim>::compute_internal_dofs(
  const double *const,
  eigen3mat &W2,
  eigen3mat &W1,
  const poly_space_basis<T, dim> &output_basis)
{
  last_step_q += time_integrator->get_sum_h_bi_ki(ki_s);

  W2 = last_step_q.block(0, 0, this->n_cell_bases, 1);
  W1 = last_step_q.block(this->n_cell_bases, 0, dim * this->n_cell_bases, 1);
  //  W1 = stored_W1;

  eigen3mat nodal_u = output_basis.get_dof_vals_at_quads(W2);
  eigen3mat nodal_q(dim * this->n_cell_bases, 1);
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    nodal_q.block(i_dim * this->n_cell_bases, 0, this->n_cell_bases, 1) =
      output_basis.get_dof_vals_at_quads(
        W1.block(i_dim * this->n_cell_bases, 0, this->n_cell_bases, 1));
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

  /*
  wreck_it_Ralph(A01);
  wreck_it_Ralph(A02);
  wreck_it_Ralph(A03);
  wreck_it_Ralph(B01T);
  wreck_it_Ralph(B02);
  wreck_it_Ralph(B03T);
  wreck_it_Ralph(C03T);
  wreck_it_Ralph(C04T);
  wreck_it_Ralph(D01);
  wreck_it_Ralph(C02);
  wreck_it_Ralph(C01);
  wreck_it_Ralph(E01);
  wreck_it_Ralph(L01);
  wreck_it_Ralph(L10);
  wreck_it_Ralph(L11);
  wreck_it_Ralph(L12);
  wreck_it_Ralph(L21);
  wreck_it_Ralph(C34T);
  wreck_it_Ralph(E31);
  wreck_it_Ralph(L31);
  */
  return 0.;
}

template <int dim>
void explicit_gn_dispersive<dim>::internal_vars_errors(const eigen3mat &u_vec,
                                                       const eigen3mat &q_vec,
                                                       double &u_error,
                                                       double &q_error)
{
  this->reinit_cell_fe_vals();

  eigen3mat total_vec((dim + 1) * this->n_cell_bases, 1);
  total_vec << u_vec, q_vec;

  /*
  double error_u2 = this->get_error_in_cell(
    explicit_gn_dispersive_qs,
    total_vec,
    time_integrator->get_current_time() + time_integrator->get_h() / 2.);
  */

  double error_u2 =
    this->get_error_in_cell(explicit_gn_dispersive_qs,
                            last_step_q,
                            time_integrator->get_current_time());

  /*
  double error_u2 = this->get_error_in_cell(
    explicit_gn_dispersive_W1, stored_W1, time_integrator->get_current_time());
  */
  //  double error_q2 = this->get_error_in_cell(
  //    explicit_gn_dispersive_W2, u_vec, time_integrator->get_current_time());
  u_error += error_u2;
  q_error += 0;
}

template <int dim>
double
explicit_gn_dispersive<dim>::get_iteration_increment_norm(const double *const)
{
  return 0.0;
}

template <int dim>
void explicit_gn_dispersive<dim>::calculate_stage_matrices()
{
  calculate_matrices();
}

template <int dim>
void explicit_gn_dispersive<dim>::ready_for_next_stage(
  double *const local_hat_vec)
{
  const std::vector<double> &quad_weights =
    this->elem_quad_bundle->get_weights();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();

  this->reinit_cell_fe_vals();
  calculate_matrices();
  eigen3mat A2_inv = std::move(A02.inverse());
  eigen3mat mat1 = std::move(
    A01 - alpha / 2. * B03T +
    (alpha / 2. * A03 + alpha / 3. * B02) * A2_inv * B01T - alpha / 3. * D01);
  eigen3mat mat2 = std::move(
    (alpha / 2. * A03 + alpha / 3. * B02) * A2_inv * C01 - alpha / 3. * C02);
  Eigen::PartialPivLU<eigen3mat> mat1_lu = std::move(mat1.partialPivLu());

  eigen3mat exact_W1_hat(dim * this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim> > W1_hat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(explicit_gn_dispersive_W1,
                                       *(this->the_face_basis),
                                       face_quad_weights,
                                       W1_hat_mtl,
                                       time_integrator->get_current_time() +
                                         time_integrator->get_cih());
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
        exact_W1_hat((i_face * dim + i_dim) * this->n_face_bases + i_poly) =
          W1_hat_mtl[i_poly][i_dim];
  }

  eigen3mat solved_W1_hat = exact_W1_hat;
  /*
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
        solved_W1_hat((i_face * dim + i_dof) * this->n_face_bases + i_poly, 0) =
          local_hat_vec[global_dof_number];
      }
    }
  }
  */

  eigen3mat W1 = stored_W1 =
    mat1_lu.solve(L01 + L10 + L11 + L12 + L21 + mat2 * solved_W1_hat);
  //  eigen3mat W1 = stored_W1 = mat1_lu.solve(L01 + L10 + mat2 *
  //  solved_W1_hat);

  /*
  mtl::vec::dense_vector<dealii::Tensor<1, dim> > W1_exact;
  this->project_to_elem_basis(explicit_gn_dispersive_W1,
                              *(this->the_elem_basis),
                              quad_weights,
                              W1_exact,
                              time_integrator->get_current_time() +
                                time_integrator->get_cih());
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    {
      W1(i_dim * this->n_cell_bases + i_poly, 0) = W1_exact[i_poly][i_dim];
    }
  */

  //  eigen3mat W2 = std::move(A2_inv * (-B01T * W1 + C01 * solved_W1_hat));
  Eigen::FullPivLU<eigen3mat> A001_lu(A001);
  eigen3mat ki = eigen3mat::Zero((dim + 1) * this->n_cell_bases, 1);
  ki.block(this->n_cell_bases, 0, 2 * this->n_cell_bases, 1) =
    A001_lu.solve(L10) - W1;
  ki_s[time_integrator->get_current_stage() - 1] = ki;

  wreck_it_Ralph(A001);
  wreck_it_Ralph(A01);
  wreck_it_Ralph(A02);
  wreck_it_Ralph(A03);
  wreck_it_Ralph(B01T);
  wreck_it_Ralph(B02);
  wreck_it_Ralph(B03T);
  wreck_it_Ralph(C03T);
  wreck_it_Ralph(C04T);
  wreck_it_Ralph(D01);
  wreck_it_Ralph(C02);
  wreck_it_Ralph(C01);
  wreck_it_Ralph(E01);
  wreck_it_Ralph(L01);
  wreck_it_Ralph(L10);
  wreck_it_Ralph(L11);
  wreck_it_Ralph(L12);
  wreck_it_Ralph(L21);
  wreck_it_Ralph(C34T);
  wreck_it_Ralph(E31);
  wreck_it_Ralph(L31);
}

template <int dim>
void explicit_gn_dispersive<dim>::ready_for_next_time_step()
{
}

template <int dim>
void explicit_gn_dispersive<dim>::ready_for_next_iteration()
{
}
