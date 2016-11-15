#include "explicit_gn_dispersive.hpp"
#include "support_classes.hpp"

template <int dim>
explicit_gn_dispersive_h_t_class<dim, double>
  explicit_gn_dispersive<dim>::h_t_func{};

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
explicit_gn_dispersive_grad_grad_grad_b_class<dim, dealii::Tensor<3, dim> >
  explicit_gn_dispersive<dim>::grad_grad_grad_b_func{};

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
    last_step_qhat(std::move(inp_cell.last_step_qhat)),
    last_stage_q(std::move(inp_cell.last_stage_q)),
    rhs_of_momentum_eq(std::move(inp_cell.rhs_of_momentum_eq)),
    connected_face_count(std::move(inp_cell.connected_face_count)),
    avg_prim_vars_flux(std::move(inp_cell.avg_prim_vars_flux)),
    jump_V_dot_n(std::move(inp_cell.jump_V_dot_n)),
    grad_h(std::move(inp_cell.grad_h)),
    div_V(std::move(inp_cell.div_V)),
    grad_V(std::move(inp_cell.grad_V)),
    avg_grad_V_flux(std::move(inp_cell.avg_grad_V_flux)),
    grad_grad_V(std::move(grad_grad_V)),
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
    last_step_qhat((dim + 1) * this->n_faces * this->n_face_bases, 1),
    last_stage_q((dim + 1) * this->n_cell_bases, 1),
    connected_face_count((dim + 1) * this->n_faces * this->n_face_bases, 0),
    avg_prim_vars_flux((dim + 1) * this->n_faces * this->n_face_bases, 1),
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
  const explicit_nswe<dim> *const src_cell)
{
  last_step_q = src_cell->last_step_q;
}

template <int dim>
eigen3mat *explicit_gn_dispersive<dim>::get_previous_step_results()
{
  return &last_step_q;
}

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
  // Here, we have projected the argument of the gradients corresponding to
  // the R1 and R2 to the element bases.
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
  */

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

      dealii::Tensor<2, dim> grad_grad_b =
        grad_grad_b_func.value(quad_pt_locs[i_quad], quad_pt_locs[i_quad]);
      dealii::Tensor<3, dim> grad_grad_grad_b =
        grad_grad_grad_b_func.value(quad_pt_locs[i_quad], quad_pt_locs[i_quad]);

      double h_h = (NT * last_stage_q.block(0, 0, n_cell_bases, 1))(0, 0);
      double hv1_h =
        (NT * last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double hv2_h =
        (NT * last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v1 = hv1_h / h_h;
      double v2 = hv2_h / h_h;

      //
      // This is a stupid way of computing the derivatives h, v1, v2
      //
      // Eigen::Matrix<double, 2, 1> grad_h_at_quad =
      //  grad_NT * last_stage_q.block(0, 0, n_cell_bases, 1);
      // Eigen::Matrix<double, 2, 1> grad_hv1_h =
      //   grad_NT * last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1);
      // Eigen::Matrix<double, 2, 1> grad_hv2_h =
      //   grad_NT * last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases,
      //   1);
      // double v1_x = (grad_hv1_h(0, 0) - v1 * grad_h_h(0, 0)) / h_h;
      // double v1_y = (grad_hv1_h(1, 0) - v1 * grad_h_h(1, 0)) / h_h;
      // double v2_x = (grad_hv2_h(0, 0) - v2 * grad_h_h(0, 0)) / h_h;
      // double v2_y = (grad_hv2_h(1, 0) - v2 * grad_h_h(1, 0)) / h_h;

      //
      // This is the smarter way of computing the derivatives of h, V
      // Here, we get the values of V_x, V_y, V_xx, V_yy, V_xy for later
      // computations.
      //
      double h = (NT * last_stage_q.block(0, 0, n_cell_bases, 1))(0, 0);
      double hx = (NT * grad_h.block(0, 0, n_cell_bases, 1))(0, 0);
      double hy = (NT * grad_h.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      Eigen::Matrix<double, 2, 1> grad_h_at_quad;
      grad_h_at_quad << hx, hy;
      double v1x = (NT * grad_V.block(0, 0, n_cell_bases, 1))(0, 0);
      double v1y = (NT * grad_V.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2x =
        (NT * grad_V.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2y =
        (NT * grad_V.block(3 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v1xx = (NT * grad_grad_V.block(0, 0, n_cell_bases, 1))(0, 0);
      double v1xy =
        (NT * grad_grad_V.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v1yx =
        (NT * grad_grad_V.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v1yy =
        (NT * grad_grad_V.block(3 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2xx =
        (NT * grad_grad_V.block(4 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2xy =
        (NT * grad_grad_V.block(5 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2yx =
        (NT * grad_grad_V.block(6 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2yy =
        (NT * grad_grad_V.block(7 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double bxx = grad_grad_b[0][0];
      double bxy = grad_grad_b[1][0];
      double byx = grad_grad_b[0][1];
      double byy = grad_grad_b[1][1];
      double bxxx = grad_grad_grad_b[0][0][0];
      double bxxy = grad_grad_grad_b[0][0][1];
      double bxyx = grad_grad_grad_b[0][1][0];
      double bxyy = grad_grad_grad_b[0][1][1];
      double byxx = grad_grad_grad_b[1][0][0];
      double byxy = grad_grad_grad_b[1][0][1];
      double byyx = grad_grad_grad_b[1][1][0];
      double byyy = grad_grad_grad_b[1][1][1];
      // Now we compute the first term in the R1(W), which is
      // grad(h^3 W). It will have two components.
      double grad_W_in_R1_1 =
        3. * h * h * hx * (-v1x * v2y + v2x * v1y + (v1x + v2y) * (v1x + v2y)) +
        h * h * h * (-v1xx * v2y + v2xx * v1y +
                     2 * (v1x + v2y) * (v1xx + v2yx) - v1x * v2yx + v2x * v1yx);
      double grad_W_in_R1_2 =
        3. * h * h * hy * (-v1x * v2y + v2x * v1y + (v1x + v2y) * (v1x + v2y)) +
        h * h * h * (-v1xy * v2y + v2xy * v1y +
                     2 * (v1x + v2y) * (v1xy + v2yy) - v1x * v2yy + v2x * v1yy);
      // Next, we obtain the first term in R2(W), which is
      // grad(h^2 W). It will have two components as well.
      double grad_W_in_R2_1 =
        2 * h * hx * (v1 * v1 * bxx + 2 * v1 * v2 * bxy + v2 * v2 * byy) +
        h * h * (2 * v1 * v1x * bxx + v1 * v1 * bxxx + 2 * v1x * v2 * bxy +
                 2 * v1 * v2x * bxy + 2 * v1 * v2 * bxyx + 2 * v2 * v2x * byy +
                 v2 * v2 * byyx);
      double grad_W_in_R2_2 =
        2 * h * hy * (v1 * v1 * bxx + 2 * v1 * v2 * bxy + v2 * v2 * byy) +
        h * h * (2 * v1 * v1y * bxx + v1 * v1 * bxxy + 2 * v1y * v2 * bxy +
                 2 * v1 * v2y * bxy + 2 * v1 * v2 * bxyy + 2 * v2 * v2y * byy +
                 v2 * v2 * byyy);

      Eigen::Matrix<double, 2, 1> L11_arg_new;
      L11_arg_new << 2. / 3. * grad_W_in_R1_1 + 1. / 2. * grad_W_in_R2_1,
        2. / 3. * grad_W_in_R1_2 + 1. / 2. * grad_W_in_R2_2;
      // Now we compare this with the analytical solution and the results
      // from L11_arg.

      // Eigen::Matrix<double, 2, 1> grad_L11_arg = grad_NT * L11_arg;
      // std::cout << L11_arg2(0, 0) << " " << grad_L11_arg(0, 0) <<
      // std::endl;
      // std::cout << grad_h_h << " " << grad_h << std::endl;

      Eigen::Matrix<double, dim, 1> L01_at_quad, grad_b_at_quad, L10_at_quad;
      dealii::Tensor<1, dim> L01_at_quad_tensor =
        explicit_gn_dispersive_L.value(
          quad_pt_locs[i_quad],
          quad_pt_locs[i_quad],
          time_integrator->get_current_stage_time());
      dealii::Tensor<1, dim> grad_b_at_quad_tensor =
        explicit_nswe_grad_b_func.value(quad_pt_locs[i_quad],
                                        quad_pt_locs[i_quad]);
      dealii::Tensor<1, dim> L10_at_quad_tensor =
        g_h_grad_zeta_func.value(quad_pt_locs[i_quad],
                                 quad_pt_locs[i_quad],
                                 time_integrator->get_current_stage_time());

      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        L01_at_quad(i_dim, 0) = L01_at_quad_tensor[i_dim];
        L10_at_quad(i_dim, 0) = L10_at_quad_tensor[i_dim];
        grad_b_at_quad(i_dim, 0) = grad_b_at_quad_tensor[i_dim];
      }

      /*
      double x1 = quad_pt_locs[i_quad][0];
      double t1 = time_integrator->get_current_stage_time();
      std::cout << ((3 * cos(5 * x1) +
                     5 * (cos(2 * t1 + 3 * x1) - 2 * sin(t1 - x1))) *
                    (54 * cos(t1 - x1) + 5 * cos(3 * t1 + 7 * x1) +
                     9 * cos(t1 + 9 * x1) -
                     40 * (4 * sin(5 * x1) + 3 * sin(2 * t1 + 3 * x1)))) /
                     (30. * pow(5 + sin(t1 + 4 * x1), 2))
                << std::endl;
      */

      /*
      std::cout << 4 * cos(4. * quad_pt_locs[i_quad][0]) << grad_h_h[0]
                << std::endl;
      std::cout << 5. + sin(4. * quad_pt_locs[i_quad][0]) - h_h << std::endl;
      */

      // This is to make the problem 1D.
      grad_NT.block(1, 0, 1, n_cell_bases) = eigen3mat::Zero(1, n_cell_bases);
      div_N.block(n_cell_bases, 0, n_cell_bases, 1) =
        eigen3mat::Zero(n_cell_bases, 1);
      grad_h_at_quad(1, 0) = 0.;
      // End of 1D modification.

      Eigen::Matrix<double, 2, 1> V2_x;
      V2_x << (2 * h * h * hx * v1x * v1x + 4. / 3. * h * h * h * v1x * v1xx),
        0.;

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
             (grad_h_at_quad + grad_b_at_quad);

      L11 += cell_JxW[i_quad] * N_vec * V2_x;

      //      L11 += cell_JxW[i_quad] * N_vec * L11_arg_new;
      //      L12 += cell_JxW[i_quad] *
      //             (h_h * h_h * (-v1x * v2y + v2x * v1y + (v1x + v2y) * (v1x +
      //             v2y)) +
      //              h_h * (v1 * v1 * bxx + v1 * v2 * bxy + v2 * v1 * byx +
      //                     v2 * v2 * byy)) *
      //             N_vec * grad_b_at_quad;

      /*
      L11 += cell_JxW[i_quad] *
             (-2. / 3. * h_h * h_h * h_h *
                (-v1_x * v2_y + v2_x * v1_y + (v1_x + v2_y) * (v1_x + v2_y)) -
              1. / 2. * h_h * h_h * (v1 * v1 * bxx + v1 * v2 * bxy +
                                     v2 * v1 * byx + v2 * v2 * byy)) *
             div_N;
      L12 += cell_JxW[i_quad] * (h_h * h_h * (-v1_x * v2_y + v2_x * v1_y +
                                              (v1_x + v2_y) * (v1_x + v2_y)) +
                                 h_h * (v1 * v1 * bxx + v1 * v2 * bxy +
                                        v2 * v1 * byx + v2 * v2 * byy)) *
             N_vec * grad_b_at_quad;
      */
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
      //
      // Now, we get the value of h, hv1, hv2 at the current quad point.
      //
      std::vector<double> qs_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
        qs_at_iquad[i_nswe_dim] =
          (Nj.transpose() *
           last_stage_q.block(i_nswe_dim * n_cell_bases, 0, n_cell_bases, 1))(
            0, 0);
      //      double h_hat = qs_at_iquad[0];
      //      double hv1_hat = qs_at_iquad[1];
      //      double hv2_hat = qs_at_iquad[2];
      //      double v1_hat = hv1_hat / h_hat;
      //      double v2_hat = hv2_hat / h_hat;

      std::vector<double> avg_prim_flux_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        int row1 = (i_face * (dim + 1) + i_nswe_dim) * n_face_bases;
        avg_prim_flux_at_iquad[i_nswe_dim] =
          (NT_face * avg_prim_vars_flux.block(row1, 0, n_face_bases, 1))(0, 0);
      }
      double h_hat = avg_prim_flux_at_iquad[0];
      double v1_hat = avg_prim_flux_at_iquad[1];
      double v2_hat = avg_prim_flux_at_iquad[2];

      //
      // Next, we get the gradient of h_h by a stupid method.
      //
      // Eigen::Matrix<double, 2, 1> grad_h_h =
      //   grad_NxT_at_face_quads * last_stage_q.block(0, 0, n_cell_bases,
      //   1);
      // Eigen::Matrix<double, 2, 1> grad_hv1_h =
      //   grad_NxT_at_face_quads *
      //   last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1);
      // Eigen::Matrix<double, 2, 1> grad_hv2_h =
      //   grad_NxT_at_face_quads *
      //   last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1);
      // double v1_x = (grad_hv1_h(0, 0) - v1 * grad_h_h(0, 0)) / h_h;
      // double v1_y = (grad_hv1_h(1, 0) - v1 * grad_h_h(1, 0)) / h_h;
      // double v2_x = (grad_hv2_h(0, 0) - v2 * grad_h_h(0, 0)) / h_h;
      // double v2_y = (grad_hv2_h(1, 0) - v2 * grad_h_h(1, 0)) / h_h;

      double hx = (Nj.transpose() * grad_h.block(0, 0, n_cell_bases, 1))(0, 0);
      double hy =
        (Nj.transpose() * grad_h.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      Eigen::Matrix<double, 2, 1> grad_h_at_quad;
      grad_h_at_quad << hx, hy;

      Eigen::Matrix<double, dim, 1> grad_b_at_quad;
      dealii::Tensor<1, dim> grad_b_at_quad_tensor =
        explicit_nswe_grad_b_func.value(face_quad_pt_locs[i_face_quad],
                                        face_quad_pt_locs[i_face_quad]);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        grad_b_at_quad(i_dim, 0) = grad_b_at_quad_tensor[i_dim];

      Eigen::Matrix<double, dim, 1> w1_at_quad;
      dealii::Tensor<1, dim> w1_at_quad_tensor =
        explicit_gn_dispersive_W1.value(
          face_quad_pt_locs[i_face_quad],
          face_quad_pt_locs[i_face_quad],
          time_integrator->get_current_stage_time());
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        w1_at_quad(i_dim, 0) = w1_at_quad_tensor[i_dim];

      // dealii::Tensor<2, dim> grad_grad_b = grad_grad_b_func.value(
      //   face_quad_pt_locs[i_face_quad], face_quad_pt_locs[i_face_quad]);
      // double b_11 = grad_grad_b[0][0];
      // double b_12 = grad_grad_b[1][0];
      // double b_21 = grad_grad_b[0][1];
      // double b_22 = grad_grad_b[1][1];
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_face_quad](i_dim);
      dealii::Tensor<1, dim> hVinf_t_tensor =
        hVinf_t_func.value(face_quad_pt_locs[i_face_quad],
                           face_quad_pt_locs[i_face_quad],
                           time_integrator->get_current_stage_time());
      Eigen::Matrix<double, dim, 1> hVinf_t_at_quad;
      hVinf_t_at_quad(0, 0) = hVinf_t_tensor[0];
      hVinf_t_at_quad(1, 0) = hVinf_t_tensor[1];

      Eigen::Matrix<double, 2, 1> W1_val_vec;
      dealii::Tensor<1, dim> W1_vec_tensor = explicit_gn_dispersive_W1.value(
        face_quad_pt_locs[i_face_quad],
        face_quad_pt_locs[i_face_quad],
        time_integrator->get_current_stage_time());
      W1_val_vec(0, 0) = W1_vec_tensor[0];
      W1_val_vec(1, 0) = W1_vec_tensor[1];

      double tau_on_face = -20.;

      // Make the problem 1D
      grad_h_at_quad(1, 0) = 0.;
      // End of 1D modification of the problem

      D01 += tau_on_face * face_JxW[i_face_quad] * N_vec * N_vec.transpose();
      C01_on_face +=
        face_JxW[i_face_quad] / h_hat * Nj * normal.transpose() * NT_face_vec;
      C02_on_face += tau_on_face * face_JxW[i_face_quad] * N_vec * NT_face_vec +
                     3. / 2. * face_JxW[i_face_quad] * h_hat * N_vec * normal *
                       grad_b_at_quad.transpose() * NT_face_vec;
      /*
      L21 += face_JxW[i_face_quad] *
             (2. / 3. * h_h * h_h * h_h *
                (-v1_x * v2_y + v2_x * v1_y + (v1_x + v2_y) * (v1_x + v2_y)) +
              1. / 2. * h_h * h_h * (v1 * v1 * b_11 + v1 * v2 * b_12 +
                                     v2 * v1 * b_21 + v2 * v2 * b_22)) *
             N_vec * normal;
      */

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

        //        L31_on_face +=
        //          face_JxW[i_face_quad] * NT_face_vec.transpose() *
        //          w1_at_quad;

        L31_on_face +=
          face_JxW[i_face_quad] * NT_face_vec.transpose() *
          (1. / alpha * gravity * h_hat * (grad_h_at_quad + grad_b_at_quad) -
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
        L31_on_face += face_JxW[i_face_quad] / alpha * gravity * h_hat *
                       NT_face_vec.transpose() * normal * normal.transpose() *
                       (grad_h_at_quad + grad_b_at_quad);
      }
      if (this->BCs[i_face] == GenericCell<dim>::BC::in_out_BC)
      {
        double vn = v1_hat * normal(0, 0) + v2_hat * normal(1, 0);
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
            (1. / alpha * gravity * h_hat * (grad_h_at_quad + grad_b_at_quad) -
             hVinf_t_at_quad) -
          vn_pls * face_JxW[i_face_quad] * NT_face_vec.transpose() * 1. /
            alpha * gravity * h_hat * (grad_h_at_quad + grad_b_at_quad);
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

  // The next line is moved to produce_h_trace(...).
  //  last_stage_q = last_step_q + time_integrator->get_sum_h_aij_kj(ki_s);

  compute_grad_grad_V();
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
        this->project_essential_BC_to_face(
          explicit_gn_dispersive_W1,
          *(this->the_face_basis),
          face_quad_weights,
          w1_hat_mtl,
          time_integrator->get_current_stage_time());
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

    eigen3mat w1 = mat1_lu.solve(L01 + L10 + L11 + L12 + L21 + mat2 * w1_hat);

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

  //  double error_u2 = this->get_error_in_cell(
  //    explicit_gn_dispersive_W1, stored_W1,
  //    time_integrator->get_current_time());

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
void explicit_gn_dispersive<dim>::get_RHS_of_momentum_eq(
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

  eigen3mat exact_h_t(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> exact_h_t_mtl;
  this->project_to_elem_basis(h_t_func,
                              *(this->the_elem_basis),
                              quad_weights,
                              exact_h_t_mtl,
                              time_integrator->get_current_stage_time());
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    exact_h_t(i_poly) = exact_h_t_mtl[i_poly];

  eigen3mat exact_W1_hat(dim * this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim> > W1_hat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(
      explicit_gn_dispersive_W1,
      *(this->the_face_basis),
      face_quad_weights,
      W1_hat_mtl,
      time_integrator->get_current_stage_time());
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
        exact_W1_hat((i_face * dim + i_dim) * this->n_face_bases + i_poly) =
          W1_hat_mtl[i_poly][i_dim];
  }

  eigen3mat solved_W1_hat = exact_W1_hat;
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

  rhs_of_momentum_eq = eigen3mat::Zero(dim * this->n_cell_bases, 1);
  stored_W1 = mat1_lu.solve(L01 + L10 + L11 + L12 + L21 + mat2 * solved_W1_hat);
  //  eigen3mat W2 = std::move(A2_inv * (-B01T * W1 + C01 * solved_W1_hat));
  Eigen::FullPivLU<eigen3mat> A001_lu(A001);

  //  ki.block(this->n_cell_bases, 0, 2 * this->n_cell_bases, 1) =
  //    A001_lu.solve(L10) - W1;

  rhs_of_momentum_eq.block(this->n_cell_bases, 0, this->n_cell_bases, 1) =
    (A001_lu.solve(L10) - stored_W1).block(0, 0, this->n_cell_bases, 1);

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

  eigen3mat exact_h_t(this->n_cell_bases, 1);
  mtl::vec::dense_vector<double> exact_h_t_mtl;
  this->project_to_elem_basis(h_t_func,
                              *(this->the_elem_basis),
                              quad_weights,
                              exact_h_t_mtl,
                              time_integrator->get_current_stage_time());
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    exact_h_t(i_poly) = exact_h_t_mtl[i_poly];

  eigen3mat exact_W1_hat(dim * this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim> > W1_hat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(
      explicit_gn_dispersive_W1,
      *(this->the_face_basis),
      face_quad_weights,
      W1_hat_mtl,
      time_integrator->get_current_stage_time());
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
        exact_W1_hat((i_face * dim + i_dim) * this->n_face_bases + i_poly) =
          W1_hat_mtl[i_poly][i_dim];
  }

  eigen3mat solved_W1_hat = exact_W1_hat;
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

  eigen3mat W1 = stored_W1 =
    mat1_lu.solve(L01 + L10 + L11 + L12 + L21 + mat2 * solved_W1_hat);
  //  eigen3mat W1 = stored_W1 = mat1_lu.solve(L01 + L10 + mat2 *
  //  solved_W1_hat);

  Eigen::FullPivLU<eigen3mat> A001_lu(A001);
  eigen3mat ki = eigen3mat::Zero((dim + 1) * this->n_cell_bases, 1);

  ki.block(0, 0, this->n_cell_bases, 1) = exact_h_t;

  //  ki.block(this->n_cell_bases, 0, 2 * this->n_cell_bases, 1) =
  //    A001_lu.solve(L10) - W1;

  ki.block(this->n_cell_bases, 0, this->n_cell_bases, 1) =
    (A001_lu.solve(L10) - W1).block(0, 0, this->n_cell_bases, 1);

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

template <int dim>
void explicit_gn_dispersive<dim>::produce_trace_of_conserved_vars(
  const explicit_nswe<dim> *const src_cell)
{
  this->reinit_cell_fe_vals();
  last_step_q.block(2 * this->n_cell_bases, 0, this->n_cell_bases, 1) =
    eigen3mat::Zero(this->n_cell_bases, 1);
  last_stage_q = last_step_q + time_integrator->get_sum_h_aij_kj(ki_s);

  std::ios_base::openmode mode1;
  mode1 = (model->sorry_for_this_boolshit)
            ? std::ofstream::out | std::ofstream::trunc
            : std::ofstream::out | std::ofstream::app;
  model->sorry_for_this_boolshit = false;

  //  std::ofstream prim_vars_test;
  //  prim_vars_test.open("prim_vars_test.txt", mode1);

  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  const unsigned n_faces = this->n_faces;
  const unsigned n_face_bases = this->n_face_bases;
  const unsigned n_cell_bases = this->n_cell_bases;
  eigen3mat conserved_vars_trace =
    eigen3mat::Zero((dim + 1) * n_faces * n_face_bases, 1); // <h,\mu>
  std::vector<double> V_dot_n((dim + 1) * n_faces * n_face_bases, 0);
  std::vector<double> face_count((dim + 1) * n_faces * n_face_bases, 1.);

  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    std::vector<double> face_JxW = this->face_quad_fe_vals->get_JxW_values();
    std::vector<dealii::Point<dim> > normals =
      this->face_quad_fe_vals->get_normal_vectors();
    eigen3mat Mat1_face = eigen3mat::Zero(n_face_bases, n_face_bases);
    eigen3mat RHS_h = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat RHS_v1 = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat RHS_v2 = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat RHS_V_dot_n = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat Nj = eigen3mat::Zero(n_cell_bases, 1);
    eigen3mat N_face = eigen3mat::Zero(n_face_bases, 1);
    std::vector<dealii::Point<dim> > projected_quad_points(
      this->face_quad_bundle->size());
    dealii::QProjector<dim>::project_to_face(
      *(this->face_quad_bundle), i_face, projected_quad_points);

    std::vector<dealii::Point<dim> > face_quad_pt_locs =
      this->face_quad_fe_vals->get_quadrature_points();
    std::vector<double> h_h_at_quads(face_quad_pt_locs.size());

    //    prim_vars_test << " face:" << i_face;

    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      std::vector<double> N_at_projected_quad_point =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);
      const std::vector<double> &face_basis = this->the_face_basis->value(
        face_quad_points[i_face_quad], this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        N_face(i_polyface, 0) = face_basis[i_polyface];
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
        Nj(i_poly, 0) = N_at_projected_quad_point[i_poly];
      double h_h =
        (Nj.transpose() * last_stage_q.block(0, 0, n_cell_bases, 1))(0, 0);

      h_h_at_quads[i_face_quad] = h_h;

      double hv1_h =
        (Nj.transpose() *
         last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double hv2_h =
        (Nj.transpose() *
         last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);

      //      if (i_face_quad == 0)
      //        prim_vars_test << " " << h_h << " " << hv1_h << " " << hv2_h;

      double v1_h = hv1_h / h_h;
      double v2_h = hv2_h / h_h;
      double V_dot_n =
        v1_h * normals[i_face_quad][0] + v2_h * normals[i_face_quad][1];

      RHS_h += face_JxW[i_face_quad] * h_h * N_face;
      RHS_v1 += face_JxW[i_face_quad] * v1_h * N_face;
      RHS_v2 += face_JxW[i_face_quad] * v2_h * N_face;
      RHS_V_dot_n += face_JxW[i_face_quad] * V_dot_n * N_face;
      Mat1_face += face_JxW[i_face_quad] * N_face * N_face.transpose();
    }
    eigen3ldlt Mat1_face_ldlt = Mat1_face.ldlt();
    eigen3mat h_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_h));
    eigen3mat v1_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v1));
    eigen3mat v2_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v2));
    eigen3mat V_dot_n_trace_on_face =
      std::move(Mat1_face_ldlt.solve(RHS_V_dot_n));

    //    prim_vars_test << std::endl;
    //    const Eigen::IOFormat fmt1(
    //      2, Eigen::DontAlignCols, " ", " ", "", "", "", "");
    //    prim_vars_test << "Cell: " << this->id_num << " Face: " << i_face << "
    //    "
    //                   << h_trace_on_face.format(fmt1) << " | "
    //                   << hv1_trace_on_face.format(fmt1) << " | "
    //                   << hv2_trace_on_face.format(fmt1) << std::endl;

    //
    // Printing some results in the quadrature points //
    //
    //    for (auto &&point : face_quad_pt_locs)
    //      prim_vars_test << point << " ";
    //    for (auto &&h1 : h_h_at_quads)
    //      prim_vars_test << h1 << " ";
    //    prim_vars_test << std::endl;

    for (unsigned i1 = 0; i1 < n_face_bases; ++i1)
    {
      unsigned row1 = i_face * (dim + 1) * this->n_face_bases;
      V_dot_n[row1 + i1] = V_dot_n_trace_on_face(i1, 0);
    }
    conserved_vars_trace.block(
      i_face * (dim + 1) * n_face_bases, 0, n_face_bases, 1) = h_trace_on_face;
    conserved_vars_trace.block(
      i_face * (dim + 1) * n_face_bases + n_face_bases, 0, n_face_bases, 1) =
      v1_trace_on_face;
    conserved_vars_trace.block(i_face * (dim + 1) * n_face_bases +
                                 2 * n_face_bases,
                               0,
                               n_face_bases,
                               1) = v2_trace_on_face;
  }
  //  prim_vars_test.close();

  std::vector<double> conserved_vars_trace_vec(conserved_vars_trace.data(),
                                               conserved_vars_trace.data() +
                                                 conserved_vars_trace.rows());

  std::vector<int> row_nums((dim + 1) * n_faces * n_face_bases, -1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    unsigned dof_count = 0;
    for (unsigned i_dof = 0; i_dof < dim + 1; ++i_dof)
      if (src_cell->dof_names_on_faces[i_face][i_dof])
      {
        for (unsigned i_polyface = 0; i_polyface < src_cell->n_face_bases;
             ++i_polyface)
        {
          unsigned global_dof_number =
            src_cell->dofs_ID_in_all_ranks[i_face][dof_count] *
              src_cell->n_face_bases +
            i_polyface;
          unsigned i_num = i_face * (dim + 1) * src_cell->n_face_bases +
                           i_dof * src_cell->n_face_bases + i_polyface;
          row_nums[i_num] = global_dof_number;
        }
        ++dof_count;
      }
  }

  model->flux_gen1->push_to_global_vec(model->flux_gen1->conserved_vars_flux,
                                       row_nums,
                                       conserved_vars_trace_vec,
                                       ADD_VALUES);
  model->flux_gen1->push_to_global_vec(
    model->flux_gen1->V_dot_n_sum, row_nums, V_dot_n, ADD_VALUES);
  model->flux_gen1->push_to_global_vec(
    model->flux_gen1->face_count, row_nums, face_count, ADD_VALUES);
}

template <int dim>
void explicit_gn_dispersive<dim>::compute_avg_prim_vars_flux(
  const explicit_nswe<dim> *const src_cell,
  double const *const local_conserved_vars_sums,
  double const *const local_face_count,
  double const *const local_V_jump)
{
  // First, we get trace of h from the local_prim_vars_sums
  // Hence, the h dof, can NEVER be blocked.
  avg_prim_vars_flux =
    eigen3mat::Zero((dim + 1) * this->n_faces * this->n_face_bases, 1);
  jump_V_dot_n = eigen3mat::Zero(this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    unsigned dof_count = 0;
    for (unsigned i_dof = 0; i_dof < dim + 1; ++i_dof)
      if (src_cell->dof_names_on_faces[i_face][i_dof])
      {
        for (unsigned i_polyface = 0; i_polyface < src_cell->n_face_bases;
             ++i_polyface)
        {
          unsigned local_dof_number =
            src_cell->dofs_ID_in_this_rank[i_face][dof_count] *
              src_cell->n_face_bases +
            i_polyface;
          unsigned i_num = i_face * (dim + 1) * src_cell->n_face_bases +
                           i_dof * src_cell->n_face_bases + i_polyface;
          connected_face_count[i_num] = local_face_count[local_dof_number];
          assert(connected_face_count[i_num] > 0);
          avg_prim_vars_flux(i_num, 0) =
            local_conserved_vars_sums[local_dof_number] /
            local_face_count[local_dof_number];
          if (i_dof == 0)
          {
            jump_V_dot_n(i_face * this->n_face_bases + i_polyface, 0) =
              local_V_jump[local_dof_number] /
              local_face_count[local_dof_number];
          }
        }
        ++dof_count;
      }
  }

  /*
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    unsigned dof_count = 1;
    for (unsigned i_dof = 1; i_dof < dim + 1; ++i_dof)
    {
      if (src_cell->dof_names_on_faces[i_face][i_dof])
      {
        for (unsigned i_polyface = 0; i_polyface < src_cell->n_face_bases;
             ++i_polyface)
        {
          unsigned local_dof_number =
            src_cell->dofs_ID_in_this_rank[i_face][dof_count] *
              src_cell->n_face_bases +
            i_polyface;
          unsigned i_num = i_face * (dim + 1) * src_cell->n_face_bases +
                           i_dof * src_cell->n_face_bases + i_polyface;
          connected_face_count[i_num] = local_face_count[local_dof_number];
          assert(connected_face_count[i_num] > 0);
          avg_prim_vars_flux(i_num, 0) =
            local_conserved_vars_sums[local_dof_number] /
            local_face_count[local_dof_number];
        }
        ++dof_count;
      }
    }
  }
  */

  /*
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();
  eigen3mat exact_q_hat((dim + 1) * this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim + 1> > qhat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(
      explicit_gn_dispersive_qs,
      *(this->the_face_basis),
      face_quad_weights,
      qhat_mtl,
      time_integrator->get_current_stage_time());
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
        exact_q_hat((i_face * (dim + 1) + i_nswe_dim) * this->n_face_bases +
                    i_poly) = qhat_mtl[i_poly][i_nswe_dim];
  }
  */

  //
  // This is the block that compute fluxes of v1, v2 from
  // fluxes of hv1, hv2, and h
  //
  /*
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  const unsigned n_faces = this->n_faces;
  const unsigned n_face_bases = this->n_face_bases;
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    std::vector<double> face_JxW = this->face_quad_fe_vals->get_JxW_values();
    eigen3mat Mat1_face = eigen3mat::Zero(n_face_bases, n_face_bases);
    eigen3mat RHS_v1 = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat RHS_v2 = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat N_face = eigen3mat::Zero(n_face_bases, 1);
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      const std::vector<double> &face_basis = this->the_face_basis->value(
        face_quad_points[i_face_quad], this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        N_face(i_polyface, 0) = face_basis[i_polyface];
      double avg_h_flux =
        (N_face.transpose() *
         avg_prim_vars_flux.block(
           i_face * (dim + 1) * n_face_bases, 0, n_face_bases, 1))(0, 0);
      double avg_hv1_hat =
        (N_face.transpose() *
         avg_prim_vars_flux.block(
           (i_face * (dim + 1) + 1) * n_face_bases, 0, n_face_bases, 1))(0, 0);
      double avg_hv2_hat =
        (N_face.transpose() *
         avg_prim_vars_flux.block(
           (i_face * (dim + 1) + 2) * n_face_bases, 0, n_face_bases, 1))(0, 0);
      double avg_v1_hat = avg_hv1_hat / avg_h_flux;
      double avg_v2_hat = avg_hv2_hat / avg_h_flux;
      RHS_v1 += face_JxW[i_face_quad] * avg_v1_hat * N_face;
      RHS_v2 += face_JxW[i_face_quad] * avg_v2_hat * N_face;
      Mat1_face += face_JxW[i_face_quad] * N_face * N_face.transpose();
    }
    eigen3ldlt Mat1_face_ldlt = Mat1_face.ldlt();
    eigen3mat v1_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v1));
    eigen3mat v2_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v2));
    avg_prim_vars_flux.block(
      (i_face * (dim + 1) + 1) * n_face_bases, 0, n_face_bases, 1) =
      v1_trace_on_face;
    avg_prim_vars_flux.block(
      (i_face * (dim + 1) + 2) * n_face_bases, 0, n_face_bases, 1) =
      v2_trace_on_face;
  }
  */

  /*
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    unsigned dof_count = 1;
    for (unsigned i_dof = 1; i_dof < dim + 1; ++i_dof)
    {
      if (src_cell->dof_names_on_faces[i_face][i_dof])
      {
        for (unsigned i_polyface = 0; i_polyface < src_cell->n_face_bases;
             ++i_polyface)
        {
          unsigned local_dof_number =
            src_cell->dofs_ID_in_this_rank[i_face][dof_count] *
              src_cell->n_face_bases +
            i_polyface;
          unsigned i_num = i_face * (dim + 1) * src_cell->n_face_bases +
                           i_dof * src_cell->n_face_bases + i_polyface;
          connected_face_count[i_num] = local_face_count[local_dof_number];
          assert(connected_face_count[i_num] > 0);
          avg_prim_vars_flux(i_num, 0) =
            local_conserved_vars_sums[local_dof_number] /
            local_face_count[local_dof_number];
        }
        ++dof_count;
      }
      //      else
      //      {
      //        for (unsigned i_polyface = 0; i_polyface <
      //        src_cell->n_face_bases;
      //             ++i_polyface)
      //        {
      //          unsigned i_num = i_face * (dim + 1) * src_cell->n_face_bases +
      //                           i_dof * src_cell->n_face_bases + i_polyface;
      //          unsigned j_num = i_face * dim * src_cell->n_face_bases +
      //                           (i_dof - 1) * src_cell->n_face_bases +
      //                           i_polyface;
      //          avg_prim_vars_flux(i_num, 0) = semi_exact_V_trace(j_num, 0);
      //        }
      //      }
    }
  }
  */
  //  std::ofstream prim_vars_test;
  //  prim_vars_test.open("prim_vars_test.txt",
  //                      std::ofstream::out | std::ofstream::app);
  //  const Eigen::IOFormat fmt1(2, Eigen::DontAlignCols, " ", " ", "", "", "",
  //  "");
  //  prim_vars_test << "Cell: " << this->id_num << " "
  //                 << avg_prim_vars_flux.format(fmt1) << std::endl;
  //  prim_vars_test.close();
}

template <int dim>
void explicit_gn_dispersive<dim>::compute_prim_vars_derivatives()
{
  this->reinit_cell_fe_vals();
  static_assert(dim == 2, "The problem dimension should be 2.");
  const unsigned n_faces = this->n_faces;
  const unsigned n_cell_bases = this->n_cell_bases;
  const unsigned n_face_bases = this->n_face_bases;
  const unsigned elem_quad_size = this->elem_quad_bundle->size();

  //  std::ofstream prim_vars_test;
  //  prim_vars_test.open("prim_vars_test.txt",
  //                      std::ofstream::out | std::ofstream::app);

  std::vector<dealii::DerivativeForm<1, dim, dim> > d_forms =
    this->cell_quad_fe_vals->get_inverse_jacobians();
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

  eigen3mat Mat1_cell = eigen3mat::Zero(n_cell_bases, n_cell_bases);
  eigen3mat RHS_grad_h = eigen3mat::Zero(dim * n_cell_bases, 1);
  eigen3mat RHS_div_V = eigen3mat::Zero(n_cell_bases, 1);
  eigen3mat RHS_grad_V = eigen3mat::Zero(dim * dim * n_cell_bases, 1);

  {
    eigen3mat grad_N, div_N, div_N2, NT;
    unsigned n_int_pts = this->elem_quad_bundle->size();
    for (unsigned i_quad = 0; i_quad < n_int_pts; ++i_quad)
    {
      grad_N = eigen3mat::Zero(n_cell_bases, dim);
      div_N = eigen3mat::Zero(dim * n_cell_bases, 1);
      div_N2 = eigen3mat::Zero(dim * dim * n_cell_bases, dim);

      NT = this->the_elem_basis->get_func_vals_at_iquad(i_quad);
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        dealii::Tensor<1, dim> grad_N_at_point = grad_N_x[i_poly][i_quad];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          div_N(i_dim * n_cell_bases + i_poly) = grad_N_at_point[i_dim];
          div_N2(i_dim * n_cell_bases + i_poly, 0) =
            div_N2((i_dim + dim) * n_cell_bases + i_poly, 1) =
              grad_N_at_point[i_dim];
          grad_N(i_poly, i_dim) = grad_N_at_point[i_dim];
        }
      }

      double h_h = (NT * last_stage_q.block(0, 0, n_cell_bases, 1))(0, 0);
      double hv1_h =
        (NT * last_stage_q.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double hv2_h =
        (NT * last_stage_q.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v1 = hv1_h / h_h;
      double v2 = hv2_h / h_h;
      Eigen::Matrix<double, 2, 1> V_h;
      V_h << v1, v2;

      Mat1_cell += cell_JxW[i_quad] * NT.transpose() * NT;
      RHS_grad_h -= cell_JxW[i_quad] * h_h * div_N;
      RHS_div_V -= cell_JxW[i_quad] * grad_N * V_h;
      RHS_grad_V -= cell_JxW[i_quad] * div_N2 * V_h;
    }
  }

  Eigen::Matrix<double, 2, 1> normal;
  Eigen::Matrix<double, 4, 2> normal2 = Eigen::Matrix<double, 4, 2>::Zero();
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    /*
     * Here, we project face quadratue points to the element space.
     * So that we can integrate a function which is defined on the element
     * domain, on face.
     */

    //    prim_vars_test << " face:" << i_face << ":";

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
    eigen3mat N_vec, N_ten;
    eigen3mat Nj = eigen3mat::Zero(n_cell_bases, 1);
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      N_vec = eigen3mat::Zero(dim * n_cell_bases, dim);
      N_ten = eigen3mat::Zero(dim * dim * n_cell_bases, dim * dim);
      std::vector<double> N_at_projected_quad_point =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);

      const std::vector<double> &face_basis = this->the_face_basis->value(
        face_quad_points[i_face_quad], this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        NT_face(0, i_polyface) = face_basis[i_polyface];

      std::vector<double> avg_prim_flux_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
        avg_prim_flux_at_iquad[i_nswe_dim] =
          (NT_face *
           avg_prim_vars_flux.block((i_face * (dim + 1) + i_nswe_dim) *
                                      n_face_bases,
                                    0,
                                    n_face_bases,
                                    1))(0, 0);
      double jump_V_dot_n_at_quad = 0;
      if (std::abs(
            connected_face_count[i_face * (dim + 1) * this->n_face_bases] -
            2.) < 1e-6)
      {
        jump_V_dot_n_at_quad =
          (NT_face *
           jump_V_dot_n.block(
             i_face * this->n_face_bases, 0, this->n_face_bases, 1))(0, 0);
      }

      Eigen::Matrix<double, 2, 1> V_flux_at_iquad;
      //      V_flux_at_iquad << avg_prim_flux_at_iquad[1],
      //      avg_prim_flux_at_iquad[2];
      double tau1 = 1.;
      V_flux_at_iquad << avg_prim_flux_at_iquad[1] -
                           tau1 * jump_V_dot_n_at_quad,
        avg_prim_flux_at_iquad[2] - tau1 * jump_V_dot_n_at_quad;

      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        Nj(i_poly, 0) = N_at_projected_quad_point[i_poly];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          N_vec(i_dim * n_cell_bases + i_poly, i_dim) =
            N_at_projected_quad_point[i_poly];
          N_ten(i_dim * n_cell_bases + i_poly, i_dim) =
            N_ten((i_dim + dim) * n_cell_bases + i_poly, i_dim + dim) =
              N_at_projected_quad_point[i_poly];
        }
      }
      // Now, we get the value of h, hv1, hv2 at the current quad point.
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        normal(i_dim, 0) = normals[i_face_quad](i_dim);
        normal2(i_dim, 0) = normal2(i_dim + dim, 1) =
          normals[i_face_quad](i_dim);
      }

      //      if (i_face_quad == 0)
      //        for (auto &&val1 : avg_prim_flux_at_iquad)
      //          prim_vars_test << val1 << " ";

      RHS_grad_h +=
        face_JxW[i_face_quad] * N_vec * normal * avg_prim_flux_at_iquad[0];
      RHS_div_V +=
        face_JxW[i_face_quad] * Nj * normal.transpose() * V_flux_at_iquad;
      RHS_grad_V += face_JxW[i_face_quad] * N_ten * normal2 * V_flux_at_iquad;
    } // End of face_i integration loop.
  }   // End of iteration over faces loop.

  //  prim_vars_test << std::endl;
  //  prim_vars_test.close();

  grad_h = eigen3mat::Zero(dim * n_cell_bases, 1);
  div_V = eigen3mat::Zero(n_cell_bases, 1);
  grad_V = eigen3mat::Zero(dim * dim * n_cell_bases, 1);
  eigen3ldlt Mat1_ldlt = Mat1_cell.ldlt();
  div_V = Mat1_ldlt.solve(RHS_div_V);
  eigen3mat h_x = Mat1_ldlt.solve(RHS_grad_h.block(0, 0, n_cell_bases, 1));
  eigen3mat h_y =
    Mat1_ldlt.solve(RHS_grad_h.block(n_cell_bases, 0, n_cell_bases, 1));
  grad_h << h_x, h_y;
  eigen3mat v1_x = Mat1_ldlt.solve(RHS_grad_V.block(0, 0, n_cell_bases, 1));
  eigen3mat v1_y =
    Mat1_ldlt.solve(RHS_grad_V.block(n_cell_bases, 0, n_cell_bases, 1));
  eigen3mat v2_x =
    Mat1_ldlt.solve(RHS_grad_V.block(2 * n_cell_bases, 0, n_cell_bases, 1));
  eigen3mat v2_y =
    Mat1_ldlt.solve(RHS_grad_V.block(3 * n_cell_bases, 0, n_cell_bases, 1));
  grad_V << v1_x, v1_y, v2_x, v2_y;

  // It looks like that the derivatives are calculated almost correctly!
  /*
  unsigned i_quad1 = 0;
  eigen3mat NT1 = this->the_elem_basis->get_func_vals_at_iquad(i_quad1);
  eigen3mat div_V_val = NT1 * div_V;
  eigen3mat grad_h_val = NT1 * grad_h.block(0, 0, n_cell_bases, 1);
  std::vector<dealii::Point<dim> > quad_pt_locs =
    this->cell_quad_fe_vals->get_quadrature_points();
  double x1 = quad_pt_locs[i_quad1][0];
  std::cout << cos(x1) / (1 + 0.2 * sin(4 * x1)) << div_V_val(0, 0)
            << std::endl;
  std::cout << 0.8 * cos(4 * x1) << grad_h_val << std::endl;
  std::cout << "Let God take you wherever he wants." << std::endl;
  */
}

template <int dim>
void explicit_gn_dispersive<dim>::produce_trace_of_grad_prim_vars(
  const explicit_nswe<dim> *const src_cell)
// We should add grad_h to this function later.
{
  this->reinit_cell_fe_vals();

  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  const unsigned n_faces = this->n_faces;
  const unsigned n_face_bases = this->n_face_bases;
  const unsigned n_cell_bases = this->n_cell_bases;
  eigen3mat V_x_trace = eigen3mat::Zero(dim * n_faces * n_face_bases, 1);
  eigen3mat V_y_trace = eigen3mat::Zero(dim * n_faces * n_face_bases, 1);
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    std::vector<double> face_JxW = this->face_quad_fe_vals->get_JxW_values();
    eigen3mat Mat1_face = eigen3mat::Zero(n_face_bases, n_face_bases);
    eigen3mat RHS_v1_x = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat RHS_v1_y = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat RHS_v2_x = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat RHS_v2_y = eigen3mat::Zero(n_face_bases, 1);
    eigen3mat Nj = eigen3mat::Zero(n_cell_bases, 1);
    eigen3mat N_face = eigen3mat::Zero(n_face_bases, 1);
    std::vector<dealii::Point<dim> > projected_quad_points(
      this->face_quad_bundle->size());
    dealii::QProjector<dim>::project_to_face(
      *(this->face_quad_bundle), i_face, projected_quad_points);
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      std::vector<double> N_at_projected_quad_point =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);
      const std::vector<double> &face_basis = this->the_face_basis->value(
        face_quad_points[i_face_quad], this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        N_face(i_polyface, 0) = face_basis[i_polyface];
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
        Nj(i_poly, 0) = N_at_projected_quad_point[i_poly];
      double v1_x_val =
        (Nj.transpose() * grad_V.block(0, 0, n_cell_bases, 1))(0, 0);
      double v1_y_val =
        (Nj.transpose() * grad_V.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2_x_val =
        (Nj.transpose() *
         grad_V.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2_y_val =
        (Nj.transpose() *
         grad_V.block(3 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      RHS_v1_x += face_JxW[i_face_quad] * v1_x_val * N_face;
      RHS_v1_y += face_JxW[i_face_quad] * v1_y_val * N_face;
      RHS_v2_x += face_JxW[i_face_quad] * v2_x_val * N_face;
      RHS_v2_y += face_JxW[i_face_quad] * v2_y_val * N_face;
      Mat1_face += face_JxW[i_face_quad] * N_face * N_face.transpose();
    }
    eigen3ldlt Mat1_face_ldlt = Mat1_face.ldlt();
    eigen3mat v1_x_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v1_x));
    eigen3mat v1_y_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v1_y));
    eigen3mat v2_x_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v2_x));
    eigen3mat v2_y_trace_on_face = std::move(Mat1_face_ldlt.solve(RHS_v2_y));
    V_x_trace.block(i_face * dim * n_face_bases, 0, n_face_bases, 1) =
      v1_x_trace_on_face;
    V_x_trace.block(
      i_face * dim * n_face_bases + n_face_bases, 0, n_face_bases, 1) =
      v2_x_trace_on_face;
    V_y_trace.block(i_face * dim * n_face_bases, 0, n_face_bases, 1) =
      v1_y_trace_on_face;
    V_y_trace.block(
      i_face * dim * n_face_bases + n_face_bases, 0, n_face_bases, 1) =
      v2_y_trace_on_face;
  }
  std::vector<double> V_x_trace_vec(V_x_trace.data(),
                                    V_x_trace.data() + V_x_trace.rows());
  std::vector<double> V_y_trace_vec(V_y_trace.data(),
                                    V_y_trace.data() + V_y_trace.rows());

  std::vector<int> row_nums(dim * n_faces * n_face_bases, -1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    unsigned dof_count = 1;
    for (unsigned i_dof = 1; i_dof < dim + 1; ++i_dof)
    {
      if (src_cell->dof_names_on_faces[i_face][i_dof])
      {
        for (unsigned i_polyface = 0; i_polyface < src_cell->n_face_bases;
             ++i_polyface)
        {
          unsigned global_dof_number =
            src_cell->dofs_ID_in_all_ranks[i_face][dof_count] *
              src_cell->n_face_bases +
            i_polyface;
          unsigned i_num = i_face * dim * src_cell->n_face_bases +
                           (i_dof - 1) * src_cell->n_face_bases + i_polyface;
          row_nums[i_num] = global_dof_number;
        }
        ++dof_count;
      }
    }
  }

  //  std::ofstream prim_vars_test;
  //  prim_vars_test.open("prim_vars_test.txt",
  //                      std::ofstream::out | std::ofstream::app);
  //  prim_vars_test << "V_x at Cell:" << this->id_num << " ";
  //  for (auto &&e1 : V_x_trace_vec)
  //    prim_vars_test << e1 << " ";
  //  prim_vars_test << std::endl;
  //  prim_vars_test << "V_y at Cell:" << this->id_num << " ";
  //  for (auto &&e1 : V_y_trace_vec)
  //    prim_vars_test << e1 << " ";
  //  prim_vars_test << std::endl;
  //  prim_vars_test.close();

  model->flux_gen1->push_to_global_vec(
    model->flux_gen1->V_x_sum, row_nums, V_x_trace_vec, ADD_VALUES);
  model->flux_gen1->push_to_global_vec(
    model->flux_gen1->V_y_sum, row_nums, V_y_trace_vec, ADD_VALUES);
}

template <int dim>
void explicit_gn_dispersive<dim>::compute_avg_grad_V_flux(
  const explicit_nswe<dim> *const src_cell,
  double const *const local_V_x_sums,
  double const *const local_V_y_sums)
{
  avg_grad_V_flux =
    eigen3mat::Zero(dim * dim * this->n_faces * this->n_face_bases, 1);

  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    unsigned dof_count = 1;
    for (unsigned i_dof = 1; i_dof < dim + 1; ++i_dof)
    {
      if (src_cell->dof_names_on_faces[i_face][i_dof])
      {
        for (unsigned i_polyface = 0; i_polyface < src_cell->n_face_bases;
             ++i_polyface)
        {
          unsigned local_dof_number1 =
            src_cell->dofs_ID_in_this_rank[i_face][dof_count] *
              src_cell->n_face_bases +
            i_polyface;
          unsigned i_num = i_face * (dim + 1) * src_cell->n_face_bases +
                           i_dof * src_cell->n_face_bases + i_polyface;
          unsigned j_num1 = i_face * dim * dim * src_cell->n_face_bases +
                            (dim * (i_dof - 1) + 0) * src_cell->n_face_bases +
                            i_polyface;
          unsigned j_num2 = i_face * dim * dim * src_cell->n_face_bases +
                            (dim * (i_dof - 1) + 1) * src_cell->n_face_bases +
                            i_polyface;
          assert(connected_face_count[i_num] > 0);
          avg_grad_V_flux(j_num1, 0) =
            local_V_x_sums[local_dof_number1] / connected_face_count[i_num];
          avg_grad_V_flux(j_num2, 0) =
            local_V_y_sums[local_dof_number1] / connected_face_count[i_num];
        }
        ++dof_count;
      }
    }
  }
  //  std::ofstream prim_vars_test;
  //  prim_vars_test.open("prim_vars_test.txt",
  //                      std::ofstream::out | std::ofstream::app);
  //  const Eigen::IOFormat fmt1(2, Eigen::DontAlignCols, " ", " ", "", "", "",
  //  "");
  //  prim_vars_test << "grad_V Trace:" << this->id_num << " "
  //                 << avg_grad_V_flux.format(fmt1) << std::endl;
  //  prim_vars_test.close();
}

template <int dim>
void explicit_gn_dispersive<dim>::compute_grad_grad_V()
{
  this->reinit_cell_fe_vals();
  static_assert(dim == 2, "The problem dimension should be 2.");
  const unsigned n_faces = this->n_faces;
  const unsigned n_cell_bases = this->n_cell_bases;
  const unsigned n_face_bases = this->n_face_bases;
  const unsigned elem_quad_size = this->elem_quad_bundle->size();

  std::vector<dealii::DerivativeForm<1, dim, dim> > d_forms =
    this->cell_quad_fe_vals->get_inverse_jacobians();
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

  eigen3mat Mat1_cell = eigen3mat::Zero(n_cell_bases, n_cell_bases);
  eigen3mat RHS_grad_v1_x = eigen3mat::Zero(dim * n_cell_bases, 1);
  eigen3mat RHS_grad_v1_y = eigen3mat::Zero(dim * n_cell_bases, 1);
  eigen3mat RHS_grad_v2_x = eigen3mat::Zero(dim * n_cell_bases, 1);
  eigen3mat RHS_grad_v2_y = eigen3mat::Zero(dim * n_cell_bases, 1);

  {
    eigen3mat div_N, NT;
    unsigned n_int_pts = this->elem_quad_bundle->size();
    for (unsigned i_quad = 0; i_quad < n_int_pts; ++i_quad)
    {
      div_N = eigen3mat::Zero(dim * n_cell_bases, 1);

      NT = this->the_elem_basis->get_func_vals_at_iquad(i_quad);
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        dealii::Tensor<1, dim> grad_N_at_point = grad_N_x[i_poly][i_quad];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
          div_N(i_dim * n_cell_bases + i_poly) = grad_N_at_point[i_dim];
      }

      double v1_x = (NT * grad_V.block(0, 0, n_cell_bases, 1))(0, 0);
      double v1_y = (NT * grad_V.block(n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2_x =
        (NT * grad_V.block(2 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);
      double v2_y =
        (NT * grad_V.block(3 * n_cell_bases, 0, n_cell_bases, 1))(0, 0);

      Mat1_cell += cell_JxW[i_quad] * NT.transpose() * NT;
      RHS_grad_v1_x -= cell_JxW[i_quad] * v1_x * div_N;
      RHS_grad_v1_y -= cell_JxW[i_quad] * v1_y * div_N;
      RHS_grad_v2_x -= cell_JxW[i_quad] * v2_x * div_N;
      RHS_grad_v2_y -= cell_JxW[i_quad] * v2_y * div_N;
    }
  }

  Eigen::Matrix<double, 2, 1> normal;
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    /*
     * Here, we project face quadratue points to the element space.
     * So that we can integrate a function which is defined on the element
     * domain, on face.
     */
    std::vector<dealii::Point<dim> > projected_quad_points(
      this->face_quad_bundle->size());
    dealii::QProjector<dim>::project_to_face(
      *(this->face_quad_bundle), i_face, projected_quad_points);
    std::vector<dealii::Point<dim> > normals =
      this->face_quad_fe_vals->get_normal_vectors();
    std::vector<double> face_JxW = this->face_quad_fe_vals->get_JxW_values();

    eigen3mat NT_face = eigen3mat::Zero(1, n_face_bases);
    eigen3mat N_vec;
    eigen3mat Nj = eigen3mat::Zero(n_cell_bases, 1);
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      N_vec = eigen3mat::Zero(dim * n_cell_bases, dim);
      std::vector<double> N_at_projected_quad_point =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);

      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        Nj(i_poly, 0) = N_at_projected_quad_point[i_poly];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          N_vec(i_dim * n_cell_bases + i_poly, i_dim) =
            N_at_projected_quad_point[i_poly];
        }
      }

      const std::vector<double> &face_basis = this->the_face_basis->value(
        face_quad_points[i_face_quad], this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        NT_face(0, i_polyface) = face_basis[i_polyface];

      unsigned row1 = i_face * dim * dim * n_face_bases; // This is face row.
      unsigned num1 = n_face_bases;
      double avg_v1_x_flux_at_iquad =
        (NT_face * avg_grad_V_flux.block(row1, 0, num1, 1))(0, 0);
      double avg_v1_y_flux_at_iquad =
        (NT_face * avg_grad_V_flux.block(row1 + num1, 0, num1, 1))(0, 0);
      double avg_v2_x_flux_at_iquad =
        (NT_face * avg_grad_V_flux.block(row1 + 2 * num1, 0, num1, 1))(0, 0);
      double avg_v2_y_flux_at_iquad =
        (NT_face * avg_grad_V_flux.block(row1 + 3 * num1, 0, num1, 1))(0, 0);

      // Now, we get the value of h, hv1, hv2 at the current quad point.
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_face_quad](i_dim);

      RHS_grad_v1_x +=
        face_JxW[i_face_quad] * N_vec * normal * avg_v1_x_flux_at_iquad;
      RHS_grad_v1_y +=
        face_JxW[i_face_quad] * N_vec * normal * avg_v1_y_flux_at_iquad;
      RHS_grad_v2_x +=
        face_JxW[i_face_quad] * N_vec * normal * avg_v2_x_flux_at_iquad;
      RHS_grad_v2_y +=
        face_JxW[i_face_quad] * N_vec * normal * avg_v2_y_flux_at_iquad;
    } // End of face_i integration loop.
  }   // End of iteration over faces loop.
  grad_grad_V = eigen3mat::Zero(dim * dim * dim * n_cell_bases, 1);
  eigen3ldlt Mat1_ldlt = Mat1_cell.ldlt();
  eigen3mat v1_xx = Mat1_ldlt.solve(RHS_grad_v1_x.block(0, 0, n_cell_bases, 1));
  eigen3mat v1_xy =
    Mat1_ldlt.solve(RHS_grad_v1_x.block(n_cell_bases, 0, n_cell_bases, 1));
  eigen3mat v1_yx = Mat1_ldlt.solve(RHS_grad_v1_y.block(0, 0, n_cell_bases, 1));
  eigen3mat v1_yy =
    Mat1_ldlt.solve(RHS_grad_v1_y.block(n_cell_bases, 0, n_cell_bases, 1));
  eigen3mat v2_xx = Mat1_ldlt.solve(RHS_grad_v2_x.block(0, 0, n_cell_bases, 1));
  eigen3mat v2_xy =
    Mat1_ldlt.solve(RHS_grad_v2_x.block(n_cell_bases, 0, n_cell_bases, 1));
  eigen3mat v2_yx = Mat1_ldlt.solve(RHS_grad_v2_y.block(0, 0, n_cell_bases, 1));
  eigen3mat v2_yy =
    Mat1_ldlt.solve(RHS_grad_v2_y.block(n_cell_bases, 0, n_cell_bases, 1));
  grad_grad_V << v1_xx, v1_xy, v1_yx, v1_yy, v2_xx, v2_xy, v2_yx, v2_yy;
  // It looks like that the derivatives are calculated almost correctly!
  /*
  unsigned i_quad1 = 0;
  eigen3mat NT1 = this->the_elem_basis->get_func_vals_at_iquad(i_quad1);
  eigen3mat div_V_val = NT1 * div_V;
  eigen3mat grad_h_val = NT1 * grad_h.block(0, 0, n_cell_bases, 1);
  std::vector<dealii::Point<dim> > quad_pt_locs =
    this->cell_quad_fe_vals->get_quadrature_points();
  double x1 = quad_pt_locs[i_quad1][0];
  std::cout << cos(x1) / (1 + 0.2 * sin(4 * x1)) << div_V_val(0, 0)
            << std::endl;
  std::cout << 0.8 * cos(4 * x1) << grad_h_val << std::endl;
  std::cout << "Let God take you wherever he wants." << std::endl;
  */
}
