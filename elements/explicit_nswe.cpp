#include "explicit_nswe.hpp"
#include "support_classes.hpp"

template <int dim>
explicit_nswe_grad_b_func_class<dim, dealii::Tensor<1, dim> >
  explicit_nswe<dim>::explicit_nswe_grad_b_func{};

template <int dim>
explicit_nswe_qis_func_class<dim, typename explicit_nswe<dim>::nswe_vec>
  explicit_nswe<dim>::explicit_nswe_qis_func{};

template <int dim>
explicit_nswe_L_func_class<dim, typename explicit_nswe<dim>::nswe_vec>
  explicit_nswe<dim>::explicit_nswe_L_func{};

template <int dim>
explicit_nswe_zero_func_class<dim, typename explicit_nswe<dim>::nswe_vec>
  explicit_nswe<dim>::explicit_nswe_zero_func{};

template <int dim>
solver_options explicit_nswe<dim>::required_solver_options()
{
  return solver_options::ignore_mat_zero_entries;
}

template <int dim>
solver_type explicit_nswe<dim>::required_solver_type()
{
  return solver_type::implicit_petsc_aij;
}

template <int dim>
unsigned explicit_nswe<dim>::get_num_dofs_per_node()
{
  return dim + 1;
}

/*!
 * The move constrcutor of the derived class should call the move
 * constructor of the base class using std::move. Otherwise the copy
 * constructor will be called.
 */
template <int dim>
explicit_nswe<dim>::explicit_nswe(explicit_nswe &&inp_cell) noexcept
  : GenericCell<dim>(std::move(inp_cell)),
    model(inp_cell.model),
    time_integrator(model->time_integrator),
    last_step_q(std::move(inp_cell.last_step_q)),
    last_iter_qhat(std::move(inp_cell.last_iter_qhat)),
    last_stage_q(std::move(inp_cell.last_stage_q)),
    ki_s(std::move(inp_cell.ki_s)),
    ki_hats(std::move(inp_cell.ki_hats))
{
}

template <int dim>
explicit_nswe<dim>::explicit_nswe(
  typename GenericCell<dim>::dealiiCell &inp_cell,
  const unsigned &id_num_,
  const unsigned &poly_order_,
  explicit_hdg_model<dim, explicit_nswe> *model_)
  : GenericCell<dim>(inp_cell, id_num_, poly_order_),
    model(model_),
    time_integrator(model_->time_integrator),
    last_step_q((dim + 1) * this->n_cell_bases, 1),
    last_iter_qhat((dim + 1) * this->n_faces * this->n_face_bases, 1),
    last_stage_q((dim + 1) * this->n_cell_bases, 1),
    ki_s(time_integrator->order,
         eigen3mat::Zero((dim + 1) * this->n_cell_bases, 1)),
    ki_hats(time_integrator->order,
            eigen3mat::Zero((dim + 1) * this->n_faces * this->n_face_bases, 1))
{
}

template <int dim>
void explicit_nswe<dim>::assign_BCs(const bool &at_boundary,
                                    const unsigned &i_face,
                                    const dealii::Point<dim> &face_center)
{
  // Example 1
  /*
  if (at_boundary && face_center[0] < -0.99)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::in_out_BC;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  */
  /*
  if (at_boundary && face_center[0] > 0.99)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::in_out_BC;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  */
  /* Example 1, 2, 3 of GN
  if (at_boundary && (face_center[0] < -9.99))
  {
    this->BCs[i_face] = GenericCell<dim>::BC::in_out_BC;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  */
  if (at_boundary)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::solid_wall;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  // End of example 1
  // Paper 3 - Example 2
  /*
  if (at_boundary && face_center[0] < -5. + 1.e-6)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::in_out_BC;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  else if (at_boundary)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::solid_wall;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  */
  // End of Paper 3 - example 2
  // Narrowing channel in paper 3
  /*
  if (at_boundary && face_center[0] < -2.0 + 1e-6)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::in_out_BC;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  else if (at_boundary && face_center[0] > 2.0 - 1e-6)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::in_out_BC;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  else if (at_boundary)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::solid_wall;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
  */
  // End of narrowing channel in paper3
  // Green-Naghdi first example: Flat bottom
  /*
  if (at_boundary)
  {
    this->BCs[i_face] = GenericCell<dim>::BC::essential;
    this->dof_names_on_faces[i_face].resize(dim + 1, 0);
  }
  */
  // End of Green-Naghdi first example: Flat bottom
  else
  {
    this->BCs[i_face] = GenericCell<dim>::BC::not_set;
    this->dof_names_on_faces[i_face].resize(dim + 1, 1);
  }
}

template <int dim>
explicit_nswe<dim>::~explicit_nswe()
{
}

template <int dim>
void explicit_nswe<dim>::assign_initial_data()
{
  this->reinit_cell_fe_vals();

  const std::vector<double> &cell_quad_weights =
    this->elem_quad_bundle->get_weights();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();

  mtl::vec::dense_vector<nswe_vec> qi_0_mtl;
  this->project_to_elem_basis(explicit_nswe_qis_func,
                              *(this->the_elem_basis),
                              cell_quad_weights,
                              qi_0_mtl);
  for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
    for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    {
      last_step_q(i_nswe_dim * this->n_cell_bases + i_poly, 0) =
        qi_0_mtl[i_poly][i_nswe_dim];
    }

  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim + 1> > qhat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(explicit_nswe_qis_func,
                                       *(this->the_face_basis),
                                       face_quad_weights,
                                       qhat_mtl,
                                       0.);
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
      {
        unsigned row0 = (i_face * (dim + 1) + i_nswe_dim) * this->n_face_bases;
        last_iter_qhat(row0 + i_poly) = qhat_mtl[i_poly][i_nswe_dim];
      }
  }
}

template <int dim>
void explicit_nswe<dim>::set_previous_step_results(
  const explicit_gn_dispersive<dim> *const src_cell)
{
  //  last_step_q = std::move(*last_step_q_);
  last_step_q = src_cell->last_step_q;
}

template <int dim>
Eigen::Matrix<double, (dim + 1), dim>
explicit_nswe<dim>::get_Fij(const std::vector<double> &qs)
{
  assert(dim == 2);
  double g = gravity;
  double q1 = qs[0];
  double q2 = qs[1];
  double q3 = qs[2];
  Eigen::Matrix<double, (dim + 1), dim> Fij;
  Fij << q2, q3, q2 * q2 / q1 + g * q1 * q1 / 2., q2 * q3 / q1, q2 * q3 / q1,
    q3 * q3 / q1 + g * q1 * q1 / 2.;
  return Fij;
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_d_Fij_dqk_nj(const std::vector<double> &qs,
                                     const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  double g = gravity;
  double q1 = qs[0];
  double q2 = qs[1];
  double q3 = qs[2];
  nswe_jac result1;
  nswe_jac result2;
  result1 << 0, 1, 0, -q2 * q2 / q1 / q1 + g * q1, 2 * q2 / q1, 0,
    -q2 * q3 / q1 / q1, q3 / q1, q2 / q1;
  result2 << 0, 0, 1, -q2 * q3 / q1 / q1, q3 / q1, q2 / q1,
    -q3 * q3 / q1 / q1 + g * q1, 0, 2 * q3 / q1;
  return result1 * normal[0] + result2 * normal[1];
}

template <int dim>
Eigen::Matrix<double, dim *(dim + 1), dim + 1>
explicit_nswe<dim>::get_partial_Fik_qj(const std::vector<double> &qs)
{
  assert(dim == 2);
  double g = gravity;
  double q1 = qs[0];
  double q2 = qs[1];
  double q3 = qs[2];
  Eigen::Matrix<double, (dim + 1) * dim, dim + 1> partial_Fik_qj;
  partial_Fik_qj << 0, 1, 0, -q2 * q2 / q1 / q1 + g * q1, 2 * q2 / q1, 0,
    -q2 * q3 / q1 / q1, q3 / q1, q2 / q1, 0, 0, 1, -q2 * q3 / q1 / q1, q3 / q1,
    q2 / q1, -q3 * q3 / q1 / q1 + g * q1, 0, 2 * q3 / q1;
  return partial_Fik_qj;
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_tauij_LF(const std::vector<double> &qhats,
                                 const dealii::Point<dim> normal)
{
  assert(dim == 2);
  double g = gravity;
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double Vn = q2 / q1 * n1 + q3 / q1 * n2;
  return (fabs(Vn) + sqrt(g * q1)) * nswe_jac::Identity();
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_partial_tauij_qhk_qj_LF(
  const std::vector<double> &qs,
  const std::vector<double> &qhats,
  const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  double q1 = qs[0];
  double q2 = qs[1];
  double q3 = qs[2];
  double qh1 = qhats[0];
  double qh2 = qhats[1];
  double qh3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  double Vn = qh2 / qh1 * n1 + qh3 / qh1 * n2;
  if (qh3 >= 0)
  {
    nswe_jac result;
    /*
    double coeff1 = (sqrt(g * qh1) - 2. * Vn) / 2. / qh1;
    result << coeff1 * q1, n1 * q1 / qh1, n2 * q1 / qh1, coeff1 * q2,
      n1 * q2 / qh1, n2 * q2 / qh1, coeff1 * q3, n1 * q3 / qh1, n2 * q3 / qh1;
    */
    result << (q1 * (qh1 * sqrt(g * qh1) - 2 * (n1 * qh2 + n2 * qh3))) /
                (2. * pow(qh1, 2)),
      (n1 * q1) / qh1, (n2 * q1) / qh1,
      (q2 * (qh1 * sqrt(g * qh1) - 2 * (n1 * qh2 + n2 * qh3))) /
        (2. * pow(qh1, 2)),
      (n1 * q2) / qh1, (n2 * q2) / qh1,
      (q3 * (qh1 * sqrt(g * qh1) - 2 * (n1 * qh2 + n2 * qh3))) /
        (2. * pow(qh1, 2)),
      (n1 * q3) / qh1, (n2 * q3) / qh1;
    return result;
  }
  else
  {
    nswe_jac result;
    /*
    double coeff1 = (sqrt(g * qh1) + 2. * Vn) / 2. / qh1;
    result << coeff1 * q1, -n1 * q1 / qh1, -n2 * q1 / qh1, coeff1 * q2,
      -n1 * q2 / qh1, -n2 * q2 / qh1, coeff1 * q3, -n1 * q3 / qh1,
      -n2 * q3 / qh1;
    */
    result << (q1 * (qh1 * sqrt(g * qh1) + 2 * n1 * qh2 + 2 * n2 * qh3)) /
                (2. * pow(qh1, 2)),
      -((n1 * q1) / qh1), -((n2 * q1) / qh1),
      (q2 * (qh1 * sqrt(g * qh1) + 2 * n1 * qh2 + 2 * n2 * qh3)) /
        (2. * pow(qh1, 2)),
      -((n1 * q2) / qh1), -((n2 * q2) / qh1),
      (q3 * (qh1 * sqrt(g * qh1) + 2 * n1 * qh2 + 2 * n2 * qh3)) /
        (2. * pow(qh1, 2)),
      -((n1 * q3) / qh1), -((n2 * q3) / qh1);
    return result;
  }
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_XR(const std::vector<double> &qhats,
                           const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  nswe_jac XR;
  XR << 0, -q1, q1, -n2, sqrt(g) * n1 * pow(q1, 1.5) - q2,
    sqrt(g) * n1 * pow(q1, 1.5) + q2, n1, sqrt(g) * n2 * pow(q1, 1.5) - q3,
    sqrt(g) * n2 * pow(q1, 1.5) + q3;
  return std::move(XR);
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_XL(const std::vector<double> &qhats,
                           const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  nswe_jac XL;
  XL << (n2 * q2 - n1 * q3) / q1, -n2, n1,
    (-sqrt(g) * pow(q1, 1.5) - n1 * q2 - n2 * q3) /
      (2. * sqrt(g) * pow(q1, 2.5)),
    n1 / (2. * sqrt(g) * pow(q1, 1.5)), n2 / (2. * sqrt(g) * pow(q1, 1.5)),
    (sqrt(g) * pow(q1, 1.5) - n1 * q2 - n2 * q3) /
      (2. * sqrt(g) * pow(q1, 2.5)),
    n1 / (2. * sqrt(g) * pow(q1, 1.5)), n2 / (2. * sqrt(g) * pow(q1, 1.5));
  return std::move(XL);
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_Dn(const std::vector<double> &qhats,
                           const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double Vn = q2 / q1 * n1 + q3 / q1 * n2;
  double g = gravity;
  nswe_jac Dn;
  Dn << Vn, 0., 0., 0., Vn - sqrt(g) * sqrt(q1), 0., 0., 0.,
    Vn + sqrt(g) * sqrt(q1);
  return std::move(Dn);
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_absDn(const std::vector<double> &qhats,
                              const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  double Vn = q2 / q1 * n1 + q3 / q1 * n2;
  nswe_jac absDn;
  if (sqrt(g) * sqrt(q1) <= Vn) // sqrt(h) <= Vn
  {
    absDn << Vn, 0., 0., 0., Vn - sqrt(g) * sqrt(q1), 0., 0., 0.,
      Vn + sqrt(g) * sqrt(q1);
  }
  else if (0 <= Vn && Vn < sqrt(g) * sqrt(q1)) // 0 <= Vn < sqrt(h)
  {
    absDn << Vn, 0., 0., 0., -Vn + sqrt(g) * sqrt(q1), 0., 0., 0.,
      Vn + sqrt(g) * sqrt(q1);
  }
  else if (-sqrt(g) * sqrt(q1) <= Vn && Vn < 0) // -sqrt(h) <= Vn < 0
  {
    absDn << -Vn, 0., 0., 0., -Vn + sqrt(g) * sqrt(q1), 0., 0., 0.,
      Vn + sqrt(g) * sqrt(q1);
  }
  else if (Vn < -sqrt(g) * sqrt(q1)) // Vn < -sqrt(h)
  {
    absDn << -Vn, 0., 0., 0., -Vn + sqrt(g) * sqrt(q1), 0., 0., 0.,
      -Vn - sqrt(g) * sqrt(q1);
  }
  else
  {
    assert(false);
  }
  return std::move(absDn);
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_Aij_plus(const std::vector<double> &qhats,
                                 const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  nswe_jac XR = get_XR(qhats, normal);
  nswe_jac Dn = get_Dn(qhats, normal);
  nswe_jac absDn = get_absDn(qhats, normal);
  nswe_jac XL = get_XL(qhats, normal);
  nswe_jac Aij = XR * ((Dn + absDn) / 2.) * XL;
  return std::move(Aij);
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_Aij_mnus(const std::vector<double> &qhats,
                                 const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  nswe_jac XR = get_XR(qhats, normal);
  nswe_jac Dn = get_Dn(qhats, normal);
  nswe_jac absDn = get_absDn(qhats, normal);
  nswe_jac XL = get_XL(qhats, normal);
  nswe_jac Aij = XR * ((Dn - absDn) / 2.) * XL;
  return std::move(Aij);
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_Aij_absl(const std::vector<double> &qhats,
                                 const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  nswe_jac XR = get_XR(qhats, normal);
  nswe_jac absDn = get_absDn(qhats, normal);
  nswe_jac XL = get_XL(qhats, normal);
  nswe_jac Aij = XR * absDn * XL;
  return std::move(Aij);
}

template <int dim>
template <typename T>
Eigen::Matrix<double, dim + 1, 1> explicit_nswe<dim>::get_solid_wall_BB(
  const T &qs, const T &qhats, const dealii::Point<dim> &normal)
{
  double q1 = qs[0];
  double q2 = qs[1];
  double q3 = qs[2];
  double qh1 = qhats[0];
  double qh2 = qhats[1];
  double qh3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  Eigen::Matrix<double, dim + 1, 1> BB;
  BB << q1 - qh1, q2 - q2 * n1 * n1 - q3 * n1 * n2 - qh2,
    q3 - q2 * n1 * n2 - q3 * n2 * n2 - qh3;
  return BB;
}

template <int dim>
template <typename T>
std::vector<typename explicit_nswe<dim>::nswe_jac>
explicit_nswe<dim>::get_dRik_dqhj(const T &qhats,
                                  const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  std::vector<nswe_jac> d_Rik_qhj(dim + 1);
  d_Rik_qhj[0] << 0., -1., 1., 0., (3. * n1 * sqrt(g * q1)) / 2.,
    (3. * n1 * sqrt(g * q1)) / 2., 0., (3. * n2 * sqrt(g * q1)) / 2.,
    (3 * n2 * sqrt(g * q1)) / 2.;
  d_Rik_qhj[1] << 0., 0., 0., 0., -1., 1., 0., 0., 0.;
  d_Rik_qhj[2] << 0., 0., 0., 0., 0., 0., 0., -1., 1.;
  return d_Rik_qhj;
}

template <int dim>
template <typename T>
std::vector<typename explicit_nswe<dim>::nswe_jac>
explicit_nswe<dim>::get_dLik_dqhj(const T &qhats,
                                  const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  std::vector<nswe_jac> d_Lik_qhj(dim + 1);
  d_Lik_qhj[0] << (-(n2 * q2) + n1 * q3) / pow(q1, 2), 0, 0,
    (2 * sqrt(g) * pow(q1, 1.5) + 5 * n1 * q2 + 5 * n2 * q3) /
      (4. * sqrt(g) * pow(q1, 3.5)),
    (-3 * n1) / (4. * sqrt(g) * pow(q1, 2.5)),
    (-3 * n2) / (4. * sqrt(g) * pow(q1, 2.5)),
    (-2 * sqrt(g) * pow(q1, 1.5) + 5 * n1 * q2 + 5 * n2 * q3) /
      (4. * sqrt(g) * pow(q1, 3.5)),
    (-3 * n1) / (4. * sqrt(g) * pow(q1, 2.5)),
    (-3 * n2) / (4. * sqrt(g) * pow(q1, 2.5));
  d_Lik_qhj[1] << n2 / q1, 0, 0, -n1 / (2. * sqrt(g) * pow(q1, 2.5)), 0, 0,
    -n1 / (2. * sqrt(g) * pow(q1, 2.5)), 0, 0;
  d_Lik_qhj[2] << -(n1 / q1), 0, 0, -n2 / (2. * sqrt(g) * pow(q1, 2.5)), 0, 0,
    -n2 / (2. * sqrt(g) * pow(q1, 2.5)), 0, 0;
  return d_Lik_qhj;
}

template <int dim>
template <typename T>
std::vector<typename explicit_nswe<dim>::nswe_jac>
explicit_nswe<dim>::get_dDik_dqhj(const T &qhats,
                                  const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  std::vector<nswe_jac> dDik_qhj(dim + 1);
  dDik_qhj[0] << -((n1 * q2 + n2 * q3) / pow(q1, 2)), 0, 0, 0,
    -(sqrt(g) * pow(q1, 1.5) + 2 * n1 * q2 + 2 * n2 * q3) / (2. * pow(q1, 2)),
    0, 0, 0,
    (sqrt(g) * pow(q1, 1.5) - 2 * n1 * q2 - 2 * n2 * q3) / (2. * pow(q1, 2));
  dDik_qhj[1] << n1 / q1, 0., 0., 0., n1 / q1, 0., 0., 0., n1 / q1;
  dDik_qhj[2] << n2 / q1, 0., 0., 0., n2 / q1, 0., 0., 0., n2 / q1;
  return dDik_qhj;
}

template <int dim>
template <typename T>
std::vector<typename explicit_nswe<dim>::nswe_jac>
explicit_nswe<dim>::get_dAbsDik_dqhj(const T &qhats,
                                     const dealii::Point<dim> &normal)
{
  double q1 = qhats[0];
  double q2 = qhats[1];
  double q3 = qhats[2];
  double n1 = normal[0];
  double n2 = normal[1];
  double g = gravity;
  double Vn = q2 / q1 * n1 + q3 / q1 * n2;
  std::vector<nswe_jac> dAbsDik_qhj(dim + 1);
  if (sqrt(g) * sqrt(q1) <= Vn) // sqrt(h) <= Vn
  {
    dAbsDik_qhj[0] << -((n1 * q2 + n2 * q3) / pow(q1, 2)), 0, 0, 0,
      -(sqrt(g) * pow(q1, 1.5) + 2 * n1 * q2 + 2 * n2 * q3) / (2. * pow(q1, 2)),
      0, 0, 0,
      (sqrt(g) * pow(q1, 1.5) - 2 * n1 * q2 - 2 * n2 * q3) / (2. * pow(q1, 2));
    dAbsDik_qhj[1] << n1 / q1, 0., 0., 0., n1 / q1, 0., 0., 0., n1 / q1;
    dAbsDik_qhj[2] << n2 / q1, 0., 0., 0., n2 / q1, 0., 0., 0., n2 / q1;
  }
  else if (0 <= Vn && Vn < sqrt(g) * sqrt(q1)) // 0 <= Vn < sqrt(h)
  {
    dAbsDik_qhj[0] << -((n1 * q2 + n2 * q3) / pow(q1, 2)), 0, 0, 0,
      (sqrt(g) * pow(q1, 1.5) + 2 * n1 * q2 + 2 * n2 * q3) / (2. * pow(q1, 2)),
      0, 0, 0,
      (sqrt(g) * pow(q1, 1.5) - 2 * n1 * q2 - 2 * n2 * q3) / (2. * pow(q1, 2));
    dAbsDik_qhj[1] << n1 / q1, 0., 0., 0., -n1 / q1, 0., 0., 0., n1 / q1;
    dAbsDik_qhj[2] << n2 / q1, 0., 0., 0., -n2 / q1, 0., 0., 0., n2 / q1;
  }
  else if (-sqrt(g) * sqrt(q1) <= Vn && Vn < 0) // -sqrt(h) <= Vn < 0
  {
    dAbsDik_qhj[0] << ((n1 * q2 + n2 * q3) / pow(q1, 2)), 0, 0, 0,
      (sqrt(g) * pow(q1, 1.5) + 2 * n1 * q2 + 2 * n2 * q3) / (2. * pow(q1, 2)),
      0, 0, 0,
      (sqrt(g) * pow(q1, 1.5) - 2 * n1 * q2 - 2 * n2 * q3) / (2. * pow(q1, 2));
    dAbsDik_qhj[1] << -n1 / q1, 0., 0., 0., -n1 / q1, 0., 0., 0., n1 / q1;
    dAbsDik_qhj[2] << -n2 / q1, 0., 0., 0., -n2 / q1, 0., 0., 0., n2 / q1;
  }
  else if (Vn < -sqrt(g) * sqrt(q1)) // Vn < -sqrt(h)
  {
    dAbsDik_qhj[0] << ((n1 * q2 + n2 * q3) / pow(q1, 2)), 0, 0, 0,
      (sqrt(g) * pow(q1, 1.5) + 2 * n1 * q2 + 2 * n2 * q3) / (2. * pow(q1, 2)),
      0, 0, 0,
      -(sqrt(g) * pow(q1, 1.5) - 2 * n1 * q2 - 2 * n2 * q3) / (2. * pow(q1, 2));
    dAbsDik_qhj[1] << -n1 / q1, 0., 0., 0., -n1 / q1, 0., 0., 0., -n1 / q1;
    dAbsDik_qhj[2] << -n2 / q1, 0., 0., 0., -n2 / q1, 0., 0., 0., -n2 / q1;
  }
  else
  {
    assert(false);
  }
  return dAbsDik_qhj;
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_dAik_dqhj_qk_plus(const std::vector<double> &qs,
                                          const std::vector<double> &qhats,
                                          const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  nswe_jac result;
  Eigen::Matrix<double, dim + 1, 1> qs_vec(qs.data());
  const std::vector<nswe_jac> &dRik_dqhj = get_dRik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dLik_dqhj = get_dLik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dDik_dqhj = get_dDik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dAbsDik_dqhj = get_dAbsDik_dqhj(qhats, normal);
  nswe_jac XR = get_XR(qhats, normal);
  nswe_jac Dn = get_Dn(qhats, normal);
  nswe_jac absDn = get_absDn(qhats, normal);
  nswe_jac XL = get_XL(qhats, normal);
  for (unsigned j = 0; j < dim + 1; ++j)
  {
    nswe_jac dAik_dqhj = (dRik_dqhj[j] * ((Dn + absDn) / 2.) * XL) +
                         (XR * ((dDik_dqhj[j] + dAbsDik_dqhj[j]) / 2.) * XL) +
                         (XR * ((Dn + absDn) / 2.) * dLik_dqhj[j]);
    result.block(0, j, dim + 1, 1) = dAik_dqhj * qs_vec;
  }
  return result;
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac explicit_nswe<dim>::get_dAik_dqhj_qk_mnus(
  const Eigen::Matrix<double, dim + 1, 1> &qs,
  const std::vector<double> &qhats,
  const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  nswe_jac result;
  const std::vector<nswe_jac> &dRik_dqhj = get_dRik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dLik_dqhj = get_dLik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dDik_dqhj = get_dDik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dAbsDik_dqhj = get_dAbsDik_dqhj(qhats, normal);
  nswe_jac XR = get_XR(qhats, normal);
  nswe_jac Dn = get_Dn(qhats, normal);
  nswe_jac absDn = get_absDn(qhats, normal);
  nswe_jac XL = get_XL(qhats, normal);
  for (unsigned j = 0; j < dim + 1; ++j)
  {
    nswe_jac dAik_dqhj = (dRik_dqhj[j] * ((Dn - absDn) / 2.) * XL) +
                         (XR * ((dDik_dqhj[j] - dAbsDik_dqhj[j]) / 2.) * XL) +
                         (XR * ((Dn - absDn) / 2.) * dLik_dqhj[j]);
    result.block(0, j, dim + 1, 1) = dAik_dqhj * qs;
  }
  return result;
}

template <int dim>
typename explicit_nswe<dim>::nswe_jac
explicit_nswe<dim>::get_dAik_dqhj_qk_absl(const std::vector<double> &qs,
                                          const std::vector<double> &qhats,
                                          const dealii::Point<dim> &normal)
{
  assert(dim == 2);
  nswe_jac result;
  Eigen::Matrix<double, dim + 1, 1> qs_vec(qs.data());
  const std::vector<nswe_jac> &dRik_dqhj = get_dRik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dLik_dqhj = get_dLik_dqhj(qhats, normal);
  const std::vector<nswe_jac> &dAbsDik_dqhj = get_dAbsDik_dqhj(qhats, normal);
  nswe_jac XR = get_XR(qhats, normal);
  nswe_jac absDn = get_absDn(qhats, normal);
  nswe_jac XL = get_XL(qhats, normal);
  for (unsigned j = 0; j < dim + 1; ++j)
  {
    nswe_jac dAik_dqhj = (dRik_dqhj[j] * absDn * XL) +
                         (XR * dAbsDik_dqhj[j] * XL) +
                         (XR * absDn * dLik_dqhj[j]);
    result.block(0, j, dim + 1, 1) = dAik_dqhj * qs_vec;
  }
  return result;
}

template <int dim>
void explicit_nswe<dim>::calculate_matrices()
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

  A00 = eigen3mat::Zero((dim + 1) * n_cell_bases, (dim + 1) * n_cell_bases);
  E00 = eigen3mat::Zero(n_faces * (dim + 1) * n_face_bases,
                        n_faces * (dim + 1) * n_face_bases);
  E01 = eigen3mat::Zero(n_faces * (dim + 1) * n_face_bases,
                        n_faces * (dim + 1) * n_face_bases);
  F01 = eigen3mat::Zero((dim + 1) * n_cell_bases, 1);
  F02 = eigen3mat::Zero((dim + 1) * n_cell_bases, 1);
  F04 = eigen3mat::Zero(n_faces * (dim + 1) * n_face_bases, 1);
  F05 = eigen3mat::Zero(n_faces * (dim + 1) * n_face_bases, 1);
  F06 = eigen3mat::Zero((dim + 1) * n_cell_bases, 1);

  /*
   * Integrating over the cell domain.
   */
  {
    std::vector<eigen3mat> last_stage_qs_at_quads;
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      last_stage_qs_at_quads.push_back(
        this->the_elem_basis->get_dof_vals_at_quads(
          last_stage_q.block(i_nswe_dim * n_cell_bases, 0, n_cell_bases, 1)));
    eigen3mat NT_nswe_vec;
    for (unsigned i_quad = 0; i_quad < this->elem_quad_bundle->size(); ++i_quad)
    {
      NT_nswe_vec = eigen3mat::Zero(dim + 1, (dim + 1) * n_cell_bases);
      eigen3mat grad_N_nswe_vec0 =
        eigen3mat::Zero((dim + 1) * n_cell_bases, dim * (dim + 1));
      eigen3sparse_mat grad_N_nswe_vec((dim + 1) * n_cell_bases,
                                       dim * (dim + 1));
      dealii::Tensor<1, dim> grad_b_at_quad_tensor =
        explicit_nswe_grad_b_func.value(quad_pt_locs[i_quad],
                                        quad_pt_locs[i_quad]);
      Eigen::Matrix<double, dim + 1, 1> grad_b_at_quad;
      grad_b_at_quad(0, 0) = 0.;
      for (unsigned i_dim = 1; i_dim < dim + 1; ++i_dim)
        grad_b_at_quad(i_dim, 0) = grad_b_at_quad_tensor[i_dim - 1];
      std::vector<eigen3triplet> grad_N_nswe_vec_triplets;
      grad_N_nswe_vec_triplets.reserve(n_cell_bases * (dim + 1) * dim);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        NT_nswe_vec.block(
          i_nswe_dim, i_nswe_dim * n_cell_bases, 1, n_cell_bases) =
          this->the_elem_basis->get_func_vals_at_iquad(i_quad);
      }
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          unsigned i_num = i_dim * (dim + 1) + i_nswe_dim;
          for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
          {
            dealii::Tensor<1, dim> grad_N_at_point = grad_N_x[i_poly][i_quad];
            grad_N_nswe_vec0(i_nswe_dim * n_cell_bases + i_poly, i_num) =
              grad_N_at_point[i_dim];
            grad_N_nswe_vec_triplets.push_back(
              eigen3triplet(i_nswe_dim * n_cell_bases + i_poly,
                            i_num,
                            grad_N_at_point[i_dim]));
          }
        }
      }
      grad_N_nswe_vec.setFromTriplets(grad_N_nswe_vec_triplets.begin(),
                                      grad_N_nswe_vec_triplets.end());
      std::vector<double> last_stage_qs_at_i_quad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
        last_stage_qs_at_i_quad[i_nswe_dim] =
          last_stage_qs_at_quads[i_nswe_dim](i_quad, 0);
      Eigen::Matrix<double, (dim + 1), dim> Fij =
        get_Fij(last_stage_qs_at_i_quad);
      Eigen::Matrix<double, (dim + 1) * dim, 1> Fij_reshaped =
        Eigen::Map<Eigen::Matrix<double, (dim + 1) * dim, 1> >(Fij.data());
      /*
       * Now we sum over all quadrature points to obtain the final form of
       * matrices.
       */
      A00 += cell_JxW[i_quad] * NT_nswe_vec.transpose() * NT_nswe_vec;
      F02 += cell_JxW[i_quad] * grad_N_nswe_vec0 * Fij_reshaped;
      F06 += cell_JxW[i_quad] * gravity * last_stage_qs_at_i_quad[0] *
             NT_nswe_vec.transpose() * grad_b_at_quad;
    }
  }
  /*
   * Starting the integration over the faces.
   */
  Eigen::Matrix<double, dim, 1> normal;
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    eigen3mat E00_on_face =
      eigen3mat::Zero((dim + 1) * n_face_bases, (dim + 1) * n_face_bases);
    eigen3mat E01_on_face =
      eigen3mat::Zero((dim + 1) * n_face_bases, (dim + 1) * n_face_bases);
    eigen3mat F04_on_face = eigen3mat::Zero((dim + 1) * n_face_bases, 1);
    eigen3mat F05_on_face = eigen3mat::Zero((dim + 1) * n_face_bases, 1);
    /*
     * Here, we project face quadratue points to the element space.
     * So that a function which is defined on the element domain,
     * can be integrated on face.
     */
    std::vector<dealii::Point<dim> > projected_quad_points(
      this->face_quad_bundle->size());
    dealii::QProjector<dim>::project_to_face(
      *(this->face_quad_bundle), i_face, projected_quad_points);
    std::vector<dealii::Point<dim> > normals =
      this->face_quad_fe_vals->get_normal_vectors();
    std::vector<double> face_JxW = this->face_quad_fe_vals->get_JxW_values();
    std::vector<dealii::Point<dim> > face_quad_points_locs =
      this->face_quad_fe_vals->get_quadrature_points();
    // Declaring new vectors to compute the face operators.
    eigen3mat NT_face = eigen3mat::Zero(1, n_face_bases);
    eigen3mat NT_face_nswe_vec =
      eigen3mat::Zero(dim + 1, (dim + 1) * n_face_bases);
    eigen3mat NT = eigen3mat::Zero(1, n_cell_bases);
    eigen3mat NT_nswe_vec = eigen3mat::Zero(dim + 1, (dim + 1) * n_cell_bases);
    /* We declare the eigen3 matrix for storing the following tensors */
    nswe_jac partial_Fij_qhk_nj, partial_tauij_qhk_qj, partial_tauij_qhk_qhj,
      tauij;
    /* Also, we declare the followings for flux conservation equation
     *
     * d_Apik__d_qhj_qk: partial of A_{ik}^+ w.r.t. qhj multiplied by q_k
     * d_Anik__d_qhj_qinfk: partial of A_{ik}^- w.r.t. qhj multiplied by
     * qinf_k
     * d_Aaij_d_qhj_qhk: partial |A_{ij}| w.r.t. qhj multiplied by qhat_k
     */
    nswe_jac dAik_dqhj_qhk, dAik_dqhj_qk_plus, dAik_dqhj_qinfk_mnus,
      dAik_dqhj_qhk_absl;
    nswe_jac Aij, Aij_plus, Aij_absl, Aij_mnus;
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_face_quad](i_dim);
      /*
       * 1.1. We obtain the values of cell basis functions at face quadrature
       *      points, and store them in an eigen3 matrix.
       */
      const std::vector<double> &cell_basis_at_iquad =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        NT(0, i_poly) = cell_basis_at_iquad[i_poly];
      }
      /* 1.2. We store the values of qs in the last stage in this vector. */
      std::vector<double> qs_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        NT_nswe_vec.block(
          i_nswe_dim, i_nswe_dim * n_cell_bases, 1, n_cell_bases) = NT;
        qs_at_iquad[i_nswe_dim] =
          (NT *
           last_stage_q.block(i_nswe_dim * n_cell_bases, 0, n_cell_bases, 1))(
            0, 0);
      }
      /*
       * 2.1. Then we obtain the values of face basis at face quadrature
       *      points, and store them in an eigen3 matrix.
       */
      const std::vector<double> &face_basis_at_iquad =
        this->the_face_basis->value(face_quad_points[i_face_quad],
                                    this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        NT_face(0, i_polyface) = face_basis_at_iquad[i_polyface];
      /* 2.2. We store the values of qhats in the last iteration in this
       *      vector. */
      std::vector<double> qhats_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        NT_face_nswe_vec.block(
          i_nswe_dim, i_nswe_dim * n_face_bases, 1, n_face_bases) = NT_face;
        unsigned row0 =
          (i_face * (dim + 1) + i_nswe_dim) * (this->n_face_bases);
        qhats_at_iquad[i_nswe_dim] =
          (NT_face * last_iter_qhat.block(row0, 0, this->n_face_bases, 1))(0,
                                                                           0);
      }
      /*
       * 3.1. Finally we compute the value of qinf at face quadrature points,
       *      and store them in the following vector.
       */
      dealii::Tensor<1, dim + 1> qinfs_at_iquad =
        explicit_nswe_qis_func.value(face_quad_points_locs[i_face_quad],
                                     face_quad_points_locs[i_face_quad],
                                     time_integrator->get_current_stage_time());
      Eigen::Matrix<double, dim + 1, 1> qinf_vec;
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
        qinf_vec[i_nswe_dim] = qinfs_at_iquad[i_nswe_dim];
      /*
       * Now, based on the above q and qhat, we obtain the tensors required
       * for
       * calculating the matrices.
       */
      partial_Fij_qhk_nj =
        get_d_Fij_dqk_nj(qhats_at_iquad, normals[i_face_quad]);
      tauij = get_tauij_LF(qhats_at_iquad, normals[i_face_quad]);
      partial_tauij_qhk_qhj = get_partial_tauij_qhk_qj_LF(
        qhats_at_iquad, qhats_at_iquad, normals[i_face_quad]);
      partial_tauij_qhk_qj = get_partial_tauij_qhk_qj_LF(
        qs_at_iquad, qhats_at_iquad, normals[i_face_quad]);
      Aij_plus = get_Aij_plus(qhats_at_iquad, normals[i_face_quad]);
      Aij_mnus = get_Aij_mnus(qhats_at_iquad, normals[i_face_quad]);
      Aij_absl = get_Aij_absl(qhats_at_iquad, normals[i_face_quad]);
      dAik_dqhj_qk_plus = get_dAik_dqhj_qk_plus(
        qs_at_iquad, qhats_at_iquad, normals[i_face_quad]);
      dAik_dqhj_qinfk_mnus =
        get_dAik_dqhj_qk_mnus(qinf_vec, qhats_at_iquad, normals[i_face_quad]);
      dAik_dqhj_qhk_absl = get_dAik_dqhj_qk_absl(
        qhats_at_iquad, qhats_at_iquad, normals[i_face_quad]);

      Eigen::Matrix<double, dim + 1, dim> Fij_hat = get_Fij(qhats_at_iquad);
      Eigen::Map<Eigen::Matrix<double, dim + 1, 1> > q_vec(qs_at_iquad.data());
      Eigen::Map<Eigen::Matrix<double, dim + 1, 1> > qh_vec(
        qhats_at_iquad.data());
      /*
       * Finally we integrate over the faces of the element to calculate the
       * matrices corresponding to the bilinear forms and functionals.
       */
      if (this->BCs[i_face] == GenericCell<dim>::not_set)
      {
        E00_on_face += face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() *
                       (partial_Fij_qhk_nj + partial_tauij_qhk_qj -
                        partial_tauij_qhk_qhj - tauij) *
                       NT_face_nswe_vec;
        F04_on_face += face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() *
                       (Fij_hat * normal + tauij * q_vec - tauij * qh_vec);
      }
      if (this->BCs[i_face] == GenericCell<dim>::in_out_BC)
      {
        E01_on_face += face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() *
                       (dAik_dqhj_qk_plus - dAik_dqhj_qinfk_mnus -
                        dAik_dqhj_qhk_absl - Aij_absl) *
                       NT_face_nswe_vec;
        F05_on_face +=
          face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() *
          (Aij_plus * q_vec - Aij_mnus * qinf_vec - Aij_absl * qh_vec);
      }
      if (this->BCs[i_face] == GenericCell<dim>::solid_wall)
      {
        Eigen::Matrix<double, dim + 1, 1> BB =
          get_solid_wall_BB(qs_at_iquad, qhats_at_iquad, normals[i_face_quad]);
        nswe_jac d_BB_dqh;
        d_BB_dqh << -1., 0., 0., 0., -1., 0., 0., 0., -1.;
        E01_on_face += face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() *
                       d_BB_dqh * NT_face_nswe_vec;
        F05_on_face +=
          face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() * BB;
      }
      if (this->BCs[i_face] == GenericCell<dim>::outflow_BC)
      {
        Eigen::Matrix<double, dim + 1, 1> BB;
        BB << q_vec[0] - qh_vec[0], q_vec[1] - qh_vec[1], q_vec[2] - qh_vec[2];
        nswe_jac d_BB_dqh;
        d_BB_dqh << -1., 0., 0., 0., -1., 0., 0., 0., -1.;
        E01_on_face += face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() *
                       d_BB_dqh * NT_face_nswe_vec;
        F05_on_face +=
          face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() * BB;
      }
      if (this->BCs[i_face] == GenericCell<dim>::inflow_BC)
      {
        Eigen::Matrix<double, dim + 1, 1> BB;
        BB << qinf_vec[0] - qh_vec[0], qinf_vec[1] - qh_vec[1],
          qinf_vec[2] - qh_vec[2];
        nswe_jac d_BB_dqh;
        d_BB_dqh << -1., 0., 0., 0., -1., 0., 0., 0., -1.;
        E01_on_face += face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() *
                       d_BB_dqh * NT_face_nswe_vec;
        F05_on_face +=
          face_JxW[i_face_quad] * NT_face_nswe_vec.transpose() * BB;
      }
      F01 += face_JxW[i_face_quad] * NT_nswe_vec.transpose() *
             (Fij_hat * normal + tauij * q_vec - tauij * qh_vec);
    }
    E00.block(i_face * (dim + 1) * n_face_bases,
              i_face * (dim + 1) * n_face_bases,
              (dim + 1) * n_face_bases,
              (dim + 1) * n_face_bases) = E00_on_face;
    E01.block(i_face * (dim + 1) * n_face_bases,
              i_face * (dim + 1) * n_face_bases,
              (dim + 1) * n_face_bases,
              (dim + 1) * n_face_bases) = E01_on_face;
    F04.block(
      i_face * (dim + 1) * n_face_bases, 0, (dim + 1) * n_face_bases, 1) =
      F04_on_face;
    F05.block(
      i_face * (dim + 1) * n_face_bases, 0, (dim + 1) * n_face_bases, 1) =
      F05_on_face;
  }
}

template <int dim>
void explicit_nswe<dim>::assemble_globals(const solver_update_keys &keys_)
{
  /*
  const std::vector<double> &cell_quad_weights =
    this->elem_quad_bundle->get_weights();
  */
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();
  // Computing the matrices for the current iteratio.
  this->reinit_cell_fe_vals();
  last_stage_q = last_step_q + model->time_integrator->get_sum_h_aij_kj(ki_s);
  calculate_matrices();

  std::vector<int> row_nums((dim + 1) * this->n_faces * this->n_face_bases, -1);
  std::vector<int> col_nums((dim + 1) * this->n_faces * this->n_face_bases, -1);
  /*
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
      {
        unsigned i_num =
          ((dim + 1) * i_face + i_nswe_dim) * this->n_face_bases + i_poly;
        int global_dof_number;
        // BEWARE! This is a bug ! If we just want to remove one of the dofs
        // and keep the others open, this next check fails. To fix this, we
        // have to store -1 for closed dofs.
        //
        if (this->dofs_ID_in_all_ranks[i_face].size() > 0)
        {
          unsigned counter = 0;
          if (this->dof_names_on_faces[i_face][i_nswe_dim] == 1)
          {
            global_dof_number = this->dofs_ID_in_all_ranks[i_face][i_nswe_dim] *
                                  this->n_face_bases +
                                i_poly;
            row_nums[i_num] = global_dof_number;
            col_nums[i_num] = global_dof_number;
          }
        }
      }
  */
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
          unsigned i_num = i_face * (dim + 1) * this->n_face_bases +
                           dof_count * this->n_face_bases + i_polyface;
          col_nums[i_num] = global_dof_number;
          row_nums[i_num] = global_dof_number;
        }
        ++dof_count;
      }
    }
  }

  eigen3mat q_inf((dim + 1) * this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim + 1> > qhat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(
      explicit_nswe_qis_func,
      *(this->the_face_basis),
      face_quad_weights,
      qhat_mtl,
      time_integrator->get_current_stage_time());
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
      {
        unsigned row0 = (i_face * (dim + 1) + i_nswe_dim) * this->n_face_bases;
        q_inf(row0 + i_poly) = qhat_mtl[i_poly][i_nswe_dim];
      }
  }
  // Here, we set dqhat_BC to the exact value of dqhat - last_iter_qhat.
  // Later, we set those values, which no BC should be applied on them
  // equal to zero.
  //
  //     THIS FUNCTIONALITY IS DEPRECATED, AND SHOULD BE REMOVED.
  //

  eigen3mat dqhat_BC = q_inf - last_iter_qhat;

  if (keys_ & update_mat)
  {
    eigen3mat cell_mat = E00 + E01;

    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      unsigned num_mat_elements =
        this->n_face_bases * (dim + 1) * this->n_face_bases * (dim + 1);
      std::vector<int> face_row_nums((dim + 1) * this->n_face_bases, -1);
      std::vector<int> face_col_nums((dim + 1) * this->n_face_bases, -1);
      unsigned dof_count = 0;
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        int global_dof_number;
        /* BEWARE! This is a bug ! If we just want to remove one of the dofs
         * and keep the others open, this next check fails. To fix this, we
         * have to store -1 for closed dofs.
         */
        if (this->dof_names_on_faces[i_face][i_nswe_dim])
        {
          for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
          {
            unsigned i_num = dof_count * this->n_face_bases + i_poly;
            global_dof_number = this->dofs_ID_in_all_ranks[i_face][dof_count] *
                                  this->n_face_bases +
                                i_poly;
            face_row_nums[i_num] = global_dof_number;
            face_col_nums[i_num] = global_dof_number;
          }
          dof_count++;
        }
        /*
        if (this->dofs_ID_in_all_ranks[i_face].size() > 0)
        {
          for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
          {
            unsigned i_num = i_nswe_dim * this->n_face_bases + i_poly;
            global_dof_number = this->dofs_ID_in_all_ranks[i_face][i_nswe_dim] *
                                  this->n_face_bases +
                                i_poly;
            face_row_nums[i_num] = global_dof_number;
            face_col_nums[i_num] = global_dof_number;
          }
        }
        */
      }
      eigen3mat cell_mat_on_face_ =
        cell_mat.block(i_face * (dim + 1) * this->n_face_bases,
                       i_face * (dim + 1) * this->n_face_bases,
                       (dim + 1) * this->n_face_bases,
                       (dim + 1) * this->n_face_bases);
      std::vector<double> cell_mat_on_face(
        cell_mat_on_face_.data(), cell_mat_on_face_.data() + num_mat_elements);
      this->model->solver->push_to_global_mat(
        face_row_nums, face_col_nums, cell_mat_on_face, ADD_VALUES);
    }
  }
  if (keys_ & update_rhs)
  {
    for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
    {
      for (unsigned i_dof = 0;
           i_dof < this->dofs_ID_in_this_rank[i_face].size();
           ++i_dof)
      {
        if (this->dof_names_on_faces[i_face][i_dof] == 1)
          for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
            dqhat_BC((i_face * (dim + 1) + i_dof) * this->n_face_bases + i_poly,
                     0) = 0;
      }
    }
    eigen3mat rhs_vec = -((E00 + E01) * dqhat_BC + F04 + F05);
    std::vector<double> rhs_col(rhs_vec.data(),
                                rhs_vec.data() + rhs_vec.rows());
    model->solver->push_to_rhs_vec(row_nums, rhs_col, ADD_VALUES);
  }
  if (keys_ & update_sol)
  {
  }

  wreck_it_Ralph(A00);
  wreck_it_Ralph(E00);
  wreck_it_Ralph(E01);
  wreck_it_Ralph(F01);
  wreck_it_Ralph(F02);
  wreck_it_Ralph(F04);
  wreck_it_Ralph(F05);
  wreck_it_Ralph(F06);
}

template <int dim>
double
explicit_nswe<dim>::get_trace_increment_norm(const double *const local_uhat)
{
  this->reinit_cell_fe_vals();
  const std::vector<double> &face_quad_weights =
    this->face_quad_bundle->get_weights();

  eigen3mat exact_qhat((dim + 1) * this->n_faces * this->n_face_bases, 1);
  for (unsigned i_face = 0; i_face < this->n_faces; ++i_face)
  {
    mtl::vec::dense_vector<dealii::Tensor<1, dim + 1> > qhat_mtl;
    this->reinit_face_fe_vals(i_face);
    this->project_essential_BC_to_face(
      explicit_nswe_qis_func,
      *(this->the_face_basis),
      face_quad_weights,
      qhat_mtl,
      time_integrator->get_current_stage_time());
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      for (unsigned i_poly = 0; i_poly < this->n_face_bases; ++i_poly)
        exact_qhat((i_face * (dim + 1) + i_nswe_dim) * this->n_face_bases +
                   i_poly) = qhat_mtl[i_poly][i_nswe_dim];
  }

  eigen3mat last_dqhat = exact_qhat - last_iter_qhat;
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
        last_dqhat((i_face * (dim + 1) + i_dof) * this->n_face_bases + i_poly,
                   0) = local_uhat[global_dof_number];
      }
    }
  }
  double delta_qh_integrated =
    this->get_error_on_faces(explicit_nswe_zero_func, last_dqhat);

  last_iter_qhat += last_dqhat;
  return delta_qh_integrated;
}

template <int dim>
template <typename T>
double explicit_nswe<dim>::compute_internal_dofs(
  const double *const,
  eigen3mat &q2,
  eigen3mat &q1,
  const poly_space_basis<T, dim> &output_basis)
{
  last_step_q += model->time_integrator->get_sum_h_bi_ki(ki_s);

  q2 = last_step_q.block(0, 0, this->n_cell_bases, 1);
  q1 = last_step_q.block(this->n_cell_bases, 0, dim * this->n_cell_bases, 1);

  eigen3mat nodal_u = output_basis.get_dof_vals_at_quads(q2);
  eigen3mat nodal_q(dim * this->n_cell_bases, 1);
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    nodal_q.block(i_dim * this->n_cell_bases, 0, this->n_cell_bases, 1) =
      output_basis.get_dof_vals_at_quads(
        q1.block(i_dim * this->n_cell_bases, 0, this->n_cell_bases, 1));
  }

  /* Now we calculate the refinement critera */
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

  wreck_it_Ralph(A00);
  wreck_it_Ralph(E00);
  wreck_it_Ralph(E01);
  wreck_it_Ralph(F01);
  wreck_it_Ralph(F02);
  wreck_it_Ralph(F04);
  wreck_it_Ralph(F05);
  wreck_it_Ralph(F06);
  return 0.0;
}

template <int dim>
void explicit_nswe<dim>::internal_vars_errors(const eigen3mat &u_vec,
                                              const eigen3mat &q_vec,
                                              double &u_error,
                                              double &q_error)
{
  this->reinit_cell_fe_vals();
  eigen3mat total_vec((dim + 1) * this->n_cell_bases, 1);
  total_vec << u_vec, q_vec;
  double error_u2 = this->get_error_in_cell(
    explicit_nswe_qis_func, total_vec, time_integrator->get_current_time());
  u_error += error_u2;
  q_error += 0;
}

template <int dim>
void explicit_nswe<dim>::calculate_stage_matrices()
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

  A00 = eigen3mat::Zero((dim + 1) * n_cell_bases, (dim + 1) * n_cell_bases);
  F01 = eigen3mat::Zero((dim + 1) * n_cell_bases, 1);
  F02 = eigen3mat::Zero((dim + 1) * n_cell_bases, 1);
  F06 = eigen3mat::Zero((dim + 1) * n_cell_bases, 1);

  /*
   * Integrating over the cell domain.
   */
  {
    std::vector<eigen3mat> last_stage_qs_at_quads;
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      last_stage_qs_at_quads.push_back(
        this->the_elem_basis->get_dof_vals_at_quads(
          last_stage_q.block(i_nswe_dim * n_cell_bases, 0, n_cell_bases, 1)));
    eigen3mat NT_nswe_vec;
    for (unsigned i_quad = 0; i_quad < this->elem_quad_bundle->size(); ++i_quad)
    {
      NT_nswe_vec = eigen3mat::Zero(dim + 1, (dim + 1) * n_cell_bases);
      eigen3mat grad_N_nswe_vec0 =
        eigen3mat::Zero((dim + 1) * n_cell_bases, dim * (dim + 1));
      eigen3sparse_mat grad_N_nswe_vec((dim + 1) * n_cell_bases,
                                       dim * (dim + 1));
      dealii::Tensor<1, dim> grad_b_at_quad_tensor =
        explicit_nswe_grad_b_func.value(quad_pt_locs[i_quad],
                                        quad_pt_locs[i_quad]);
      Eigen::Matrix<double, dim + 1, 1> grad_b_at_quad;
      grad_b_at_quad[0] = 0.;
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        grad_b_at_quad[i_dim + 1] = grad_b_at_quad_tensor[i_dim];

      std::vector<eigen3triplet> grad_N_nswe_vec_triplets;
      grad_N_nswe_vec_triplets.reserve(n_cell_bases * (dim + 1) * dim);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        NT_nswe_vec.block(
          i_nswe_dim, i_nswe_dim * n_cell_bases, 1, n_cell_bases) =
          this->the_elem_basis->get_func_vals_at_iquad(i_quad);
      }
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          unsigned i_num = i_dim * (dim + 1) + i_nswe_dim;
          for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
          {
            dealii::Tensor<1, dim> grad_N_at_point = grad_N_x[i_poly][i_quad];
            grad_N_nswe_vec0(i_nswe_dim * n_cell_bases + i_poly, i_num) =
              grad_N_at_point[i_dim];
            grad_N_nswe_vec_triplets.push_back(
              eigen3triplet(i_nswe_dim * n_cell_bases + i_poly,
                            i_num,
                            grad_N_at_point[i_dim]));
          }
        }
      }
      grad_N_nswe_vec.setFromTriplets(grad_N_nswe_vec_triplets.begin(),
                                      grad_N_nswe_vec_triplets.end());
      std::vector<double> last_stage_qs_at_i_quad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
        last_stage_qs_at_i_quad[i_nswe_dim] =
          last_stage_qs_at_quads[i_nswe_dim](i_quad, 0);
      Eigen::Matrix<double, (dim + 1), dim> Fij =
        get_Fij(last_stage_qs_at_i_quad);
      Eigen::Matrix<double, (dim + 1) * dim, 1> Fij_reshaped =
        Eigen::Map<Eigen::Matrix<double, (dim + 1) * dim, 1> >(Fij.data());
      /*
       * Now we sum over all quadrature points to obtain the final form of
       * matrices.
       */
      A00 += cell_JxW[i_quad] * NT_nswe_vec.transpose() * NT_nswe_vec;
      F02 += cell_JxW[i_quad] * grad_N_nswe_vec0 * Fij_reshaped;
      F06 += cell_JxW[i_quad] * gravity * last_stage_qs_at_i_quad[0] *
             NT_nswe_vec.transpose() * grad_b_at_quad;
    }
  }
  /*
   * Starting the integration over the faces.
   */
  Eigen::Matrix<double, dim, 1> normal;
  std::vector<dealii::Point<dim - 1> > face_quad_points =
    this->face_quad_bundle->get_points();
  for (unsigned i_face = 0; i_face < n_faces; ++i_face)
  {
    this->reinit_face_fe_vals(i_face);
    /*
     * Here, we project face quadratue points to the element space.
     * So that a function which is defined on the element domain,
     * can be integrated on face.
     */
    std::vector<dealii::Point<dim> > projected_quad_points(
      this->face_quad_bundle->size());
    dealii::QProjector<dim>::project_to_face(
      *(this->face_quad_bundle), i_face, projected_quad_points);
    std::vector<dealii::Point<dim> > normals =
      this->face_quad_fe_vals->get_normal_vectors();
    std::vector<double> face_JxW = this->face_quad_fe_vals->get_JxW_values();
    std::vector<dealii::Point<dim> > face_quad_points_locs =
      this->face_quad_fe_vals->get_quadrature_points();
    // Declaring new vectors to compute the face operators.
    eigen3mat NT_face = eigen3mat::Zero(1, n_face_bases);
    eigen3mat NT_face_nswe_vec =
      eigen3mat::Zero(dim + 1, (dim + 1) * n_face_bases);
    eigen3mat NT = eigen3mat::Zero(1, n_cell_bases);
    eigen3mat NT_nswe_vec = eigen3mat::Zero(dim + 1, (dim + 1) * n_cell_bases);
    /* We declare the eigen3 matrix for storing the following tensors */
    nswe_jac tauij;
    /* Also, we declare the followings for flux conservation equation
     *
     * d_Apik__d_qhj_qk: partial of A_{ik}^+ w.r.t. qhj multiplied by q_k
     * d_Anik__d_qhj_qinfk: partial of A_{ik}^- w.r.t. qhj multiplied by
     * qinf_k
     * d_Aaij_d_qhj_qhk: partial |A_{ij}| w.r.t. qhj multiplied by qhat_k
     */
    for (unsigned i_face_quad = 0; i_face_quad < this->face_quad_bundle->size();
         ++i_face_quad)
    {
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = normals[i_face_quad](i_dim);
      /*
       * 1.1. We obtain the values of cell basis functions at face quadrature
       *      points, and store them in an eigen3 matrix.
       */
      const std::vector<double> &cell_basis_at_iquad =
        this->the_elem_basis->value(projected_quad_points[i_face_quad]);
      for (unsigned i_poly = 0; i_poly < n_cell_bases; ++i_poly)
      {
        NT(0, i_poly) = cell_basis_at_iquad[i_poly];
      }
      /* 1.2. We store the values of qs in the last stage in this vector. */
      std::vector<double> qs_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        NT_nswe_vec.block(
          i_nswe_dim, i_nswe_dim * n_cell_bases, 1, n_cell_bases) = NT;
        qs_at_iquad[i_nswe_dim] =
          (NT *
           last_stage_q.block(i_nswe_dim * n_cell_bases, 0, n_cell_bases, 1))(
            0, 0);
      }
      /*
       * 2.1. Then we obtain the values of face basis at face quadrature
       *      points, and store them in an eigen3 matrix.
       */
      const std::vector<double> &face_basis_at_iquad =
        this->the_face_basis->value(face_quad_points[i_face_quad],
                                    this->half_range_flag[i_face]);
      for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
        NT_face(0, i_polyface) = face_basis_at_iquad[i_polyface];
      /* 2.2. We store the values of qhats in the last iteration in this
       *      vector. */
      std::vector<double> qhats_at_iquad(dim + 1);
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      {
        NT_face_nswe_vec.block(
          i_nswe_dim, i_nswe_dim * n_face_bases, 1, n_face_bases) = NT_face;
        unsigned row0 =
          (i_face * (dim + 1) + i_nswe_dim) * (this->n_face_bases);
        qhats_at_iquad[i_nswe_dim] =
          (NT_face * last_iter_qhat.block(row0, 0, this->n_face_bases, 1))(0,
                                                                           0);
      }
      /*
       * 3.1. Finally we compute the value of qinf at face quadrature points,
       *      and store them in the following vector.
       */
      dealii::Tensor<1, dim + 1> qinfs_at_iquad =
        explicit_nswe_qis_func.value(face_quad_points_locs[i_face_quad],
                                     face_quad_points_locs[i_face_quad],
                                     time_integrator->get_current_stage_time());
      Eigen::Matrix<double, dim + 1, 1> qinf_vec;
      for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
        qinf_vec[i_nswe_dim] = qinfs_at_iquad[i_nswe_dim];
      /*
       * Now, based on the above q and qhat, we obtain the tensors required
       * for
       * calculating the matrices.
       */
      tauij = get_tauij_LF(qhats_at_iquad, normals[i_face_quad]);
      Eigen::Matrix<double, dim + 1, dim> Fij_hat = get_Fij(qhats_at_iquad);
      Eigen::Map<Eigen::Matrix<double, dim + 1, 1> > q_vec(qs_at_iquad.data());
      Eigen::Map<Eigen::Matrix<double, dim + 1, 1> > qh_vec(
        qhats_at_iquad.data());
      /*
       * Finally we integrate over the faces of the element to calculate the
       * matrices corresponding to the bilinear forms and functionals.
       */
      F01 += face_JxW[i_face_quad] * NT_nswe_vec.transpose() *
             (Fij_hat * normal + tauij * q_vec - tauij * qh_vec);
    }
  }
}

template <int dim>
void explicit_nswe<dim>::ready_for_next_stage()
{
  const std::vector<double> &cell_quad_weights =
    this->elem_quad_bundle->get_weights();

  this->reinit_cell_fe_vals();
  calculate_stage_matrices();

  eigen3mat f03((dim + 1) * this->n_cell_bases, 1);
  mtl::vec::dense_vector<dealii::Tensor<1, dim + 1> > f03_mtl;
  this->project_to_elem_basis(explicit_nswe_L_func,
                              *(this->the_elem_basis),
                              cell_quad_weights,
                              f03_mtl,
                              time_integrator->get_current_stage_time());
  for (unsigned i_poly = 0; i_poly < this->n_cell_bases; ++i_poly)
    for (unsigned i_nswe_dim = 0; i_nswe_dim < dim + 1; ++i_nswe_dim)
      f03(i_nswe_dim * this->n_cell_bases + i_poly, 0) =
        f03_mtl[i_poly][i_nswe_dim];

  Eigen::FullPivLU<eigen3mat> A00_lu(A00);
  eigen3mat ki = A00_lu.solve(-(F01 - F02 + F06 - A00 * f03));
  //  eigen3mat ki = A00_lu.solve(-(F01 - F02 - A00 * f03));
  ki_s[model->time_integrator->get_current_stage() - 1] = ki;

  wreck_it_Ralph(A00);
  wreck_it_Ralph(F01);
  wreck_it_Ralph(F02);
  wreck_it_Ralph(F06);
}

template <int dim>
void explicit_nswe<dim>::ready_for_next_time_step()
{
}
