#include "cell_class.hpp"

#ifndef FLAT_LIN_WDISP_HPP
#define FLAT_LIN_WDISP_HPP

static double mu = 1.0;

/*!
 * \brief
 * This class gives the values of the auxiliary unknown \f$\mathbf q_1\f$ at
 * any given point at a given time.
 * \details
 * Refer to \ref equation_sec "formulation" and
 * \ref GN_0_0_stage2_page "numerical examples" to know more about \f$\mathbf
 * q_1\f$,
 * and how it is related to \f$q_2\f$ and \f$\nabla \zeta\f$
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct q1_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    dealii::Tensor<1, in_point_dim> q1;
    /* Example 1: */
    q1[0] = sin(M_PI * (x[0] - t));
    q1[1] = 0;
    /* End of Example 1 */
    return q1;
  }
};

/*!
 * \brief
 * This class gives the values of the auxiliary unknown \f$q_2\f$ at
 * any given point at a given time.
 * \details
 * Refer to \ref equation_sec "formulation" and
 * \ref GN_0_0_stage2_page "numerical examples" to know more about \f$q_2\f$,
 * and how it is related to \f$\mathbf q_1\f$ and \f$\nabla \zeta\f$.
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct q2_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0.0) const final
  {
    double q2;
    /* Example 1: */
    q2 = M_PI * cos(M_PI * (x[0] - t));
    /* End of Example 1 */
    return q2;
  }
};

/*!
 * \brief
 * This class gives the values of the unknown \f$\zeta\f$ at
 * any given point at a given time.
 * \details
 * Refer to \ref equation_sec "formulation" and
 * \ref GN_0_0_stage2_page "numerical examples" to know more about the
 * relation of \f$\zeta\f$ and \f$\mathbf q_1\f$ and \f$q_2\f$.
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct zeta_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    double zeta;
    /* Example 1: */
    zeta = -(3.0 + mu * M_PI * M_PI) / 3.0 / M_PI * cos(M_PI * (x[0] - t));
    /* End of Example 1 */
    return zeta;
  }
};

template <int in_point_dim, typename output_type>
struct grad_zeta_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double &t = 0) const final
  {
    dealii::Tensor<1, in_point_dim> grad_zeta;
    /* Example 1: */
    grad_zeta[0] = (1.0 + mu * M_PI * M_PI / 3.0) * sin(M_PI * (x[0] - t));
    grad_zeta[1] = 0.0;
    /* End of Example 1 */
    return grad_zeta;
  }
};

/*!
 * \page GN_0_0_stage2_page Numerical Examples
 *
 * <b>Example 1: </b>As a first step we solve the following system
 * of equations:
 * \f[
 * \left\{ \begin{aligned} & \partial_t \zeta+\nabla \cdot \mathbf V = 0 \\ &
 * \partial_t \mathbf V - \frac{\mu}{3} \nabla (\nabla \cdot (\partial_t \mathbf
 * V)) + \nabla \zeta = 0 \end{aligned} \right.
 * \f]
 * The reader can refer to the \ref equation_sec "equation section" of
 * the documentation to know more about this equation. The solution
 * procedure is also explained there. Generally, we use a method based on
 * splitting the operator to the following two parts:
 * \f[
 * \text{Eq a: }
 * \left\{ \begin{aligned} & \partial_t \zeta+\nabla \cdot \mathbf V =
 * 0 \\ & \partial_t \mathbf V = 0 \end{aligned} \right. ,\quad
 * \text{Eq b: } \left\{ \begin{aligned} & \partial_t \zeta = 0 \\ &
 * \partial_t \mathbf V + \left(\mathcal I + \mu \mathcal T\right)^{-
 * 1} \nabla \zeta = 0 \end{aligned} \right.
 * \f]
 * We write Eq. b, in the following form of first order system:
 * \f[
 * \begin{aligned}
 * \partial_t \zeta &= 0 \\
 * \partial_t \mathbf V + \mathbf q_1 &= 0 \\
 * \mathbf q_1 - \frac \mu 3 \nabla \cdot (q_2\mathbf I) &= \nabla \zeta \\
 * q_2 - \nabla \cdot \mathbf q_1 &= 0
 * \end{aligned}
 * \f]
 * We consider the domain to be the 2D strip \f$(0,20)\times(0,1)\f$, with
 * periodic boundary conditions at the two ends, and assume the following
 * forms for the unknowns:
 * \f[
 * \begin{gathered}
 * \mathbf q_1 =
 *   \begin{Bmatrix}
 *     \sin \pi(x-t) \\ 0
 *   \end{Bmatrix}; \quad
 * q_2 = \nabla \cdot \mathbf q_1 = \pi \cos\pi(x-t) \\
 * \nabla \zeta = \mathbf q_1 - \frac \mu 3 \nabla q_2 =
 *   \begin{Bmatrix}
 *     (1+\mu \pi^2/3) \sin\pi\left(x-t\right) \\ 0
 *   \end{Bmatrix} \Longrightarrow
 * \zeta = -\frac {3+\mu\pi^2}{3\pi} \cos\pi(x-t)\\
 * \partial_t \mathbf V + \mathbf q_1 = 0 \Longrightarrow
 * \mathbf V =
 *   \begin{Bmatrix}
 *     -\frac 1 \pi \cos\pi(x-t) \\ 0
 *   \end{Bmatrix};
 * \end{gathered}
 * \f]
 * We also calculate the residual of the first equation, based on the
 * above \f$\mathbf V, \zeta\f$:
 * \f[
 * \partial_t \zeta + \nabla \cdot \mathbf V = \frac{\mu\pi^2}{3}
 * \sin\pi(x-t).
 * \f]
 * All of these definitions are included in the \ref input_data_group
 * "input data group". The model dimenstion and initial discretization
 * is contained in SolutionManager::SolutionManager. The boundary conditions are
 * assigned in SolutionManager::set_boundary_indicator.
 */

/*!
 * \ingroup cells
 */
template <int dim>
struct GN_eps_0_beta_0 : public GenericCell<dim>
{
  const double mu = 1.0;
  using elem_basis_type = typename GenericCell<dim>::elem_basis_type;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;

  static solver_options required_solver_options();
  static solver_type required_solver_type();
  static unsigned get_num_dofs_per_node();

  /*!
   * \brief Constructor for the GN_eps_0_beta_0.
   * \details Since we are using factory pattern to create cells,
   * we call this constructor, from the GenericCell::make_cell. In this case,
   * due to IS-A rule, the constructor of the GenericCell will be called as
   * well.
   * This is not good, because we have already deleted the default constructor
   * of the GenericCell. So, we call our desired constructor by putting it in
   * the
   * initializer list.
   * \param inp_cell
   * \param id_num_
   */
  GN_eps_0_beta_0() = delete;
  GN_eps_0_beta_0(const GN_eps_0_beta_0 &inp_cell) = delete;
  GN_eps_0_beta_0(GN_eps_0_beta_0 &&inp_cell) noexcept;
  GN_eps_0_beta_0(typename GenericCell<dim>::dealiiCell &inp_cell,
                  const unsigned &id_num_,
                  const unsigned &poly_order_,
                  hdg_model<dim, GN_eps_0_beta_0> *model_);
  ~GN_eps_0_beta_0() final;
  eigen3mat A1, A2, B1, B2, C2, C1, D1, D2, E1;
  void assign_BCs(const bool &at_boundary,
                  const unsigned &i_face,
                  const dealii::Point<dim> &face_center);

  /*!
   * \brief Caluclates the matrix form of the bilinear forms of the
   * equations.
   *
   * We want to construct the bilinear forms for the following equation:
   * \f[
   *   \left\{
   *     \begin{aligned}
   *       & (\mathbf q_1, \mathbf U) -
   *         \frac \mu 3 \langle \widehat {\mathsf {q_2}} \cdot
   *                             \mathbf n,\mathbf U
   *                     \rangle +
   *         \frac \mu 3 (q_2 \mathbf I , \nabla \mathbf U) =
   *         (\nabla \zeta , \mathbf U)
   *      \\
   *       & (q_2 , p_2) -
   *         \langle \hat{\mathbf q}_1 \cdot \mathbf n, p_2 \rangle +
   *         (\mathbf q_1 , \nabla p_2) = 0
   *     \end{aligned}
   *   \right.
   * \f]
   * Which finds the following form, after substituting:
   * \f$
   *   \widehat{\mathsf {q_2}} \cdot \mathbf n =
   *     (q_2 \mathbf I) \cdot \mathbf n +
   *     \boldsymbol \tau (\mathbf q_1 - \hat {\mathbf q_1})
   * \f$:
   * \f[
   *   \left\{
   *     \begin{aligned}
   *       & (\mathbf q_1, \mathbf U)
   *         -\frac \mu 3 (\nabla q_2, \mathbf U)
   *         -\frac \mu 3 \langle
   *                       {\boldsymbol \tau} \mathbf q_1,\mathbf U
   *                     \rangle
   *         +\frac \mu 3 \langle \boldsymbol{\tau}
   *                       \hat{\mathbf q}_1,\mathbf U
   *                     \rangle =
   *         (\nabla \zeta , \mathbf U) \\
   *       & (q_2 , p_2) -
   *         \langle \hat{\mathbf q}_1 \cdot \mathbf n, p_2 \rangle +
   *         (\mathbf q_1 , \nabla p_2) = 0
   *     \end{aligned}
   *   \right.
   * \f]
   * Next, we define the following bilinear forms:
   * \f[
   *   a_1(\mathbf q_1,\mathbf U) = (\mathbf q_1, \mathbf U);
   *   \quad
   *   b_1(q_2, \mathbf U) = (\nabla q_2 , \mathbf U);
   *   \quad
   *   d_1(\mathbf q_1, \mathbf U) =
   *     \langle\boldsymbol \tau \mathbf q_1, \mathbf U \rangle;
   *   \quad
   *   c_2(\hat{\mathbf q}_1, \mathbf U) =
   *     \langle \tau \hat{\mathbf q}_1 ,\mathbf U \rangle;
   * \f]
   * \f[
   *   a_2(q_2,p_2) = (q_2, p_2);
   *   \quad
   *   c_1(\hat {\mathbf q}_1 , p_2) =
   *     \langle \hat {\mathbf q}_1 \cdot \mathbf n, p_2 \rangle;
   *   \quad
   *   z_1(\mathbf U) = (\nabla \zeta , \mathbf U).
   * \f]
   * We also define the following two bilinear operators for testing purposes:
   * \f[
   * b_2(q_2, \mathbf U) = (q_2 , \nabla \cdot \mathbf U); \quad
   * d_2(q_2, \mathbf U) = \langle q_2, \mathbf U \cdot \mathbf n \rangle.
   * \f]
   * So, we have:
   * \f[
   *   \left\{
   *     \begin{aligned}
   *       & A_1 Q_1 - \frac \mu 3 B_1 Q_2 -\frac \mu 3 D_1 Q_1
   *         + \frac \mu 3 C_2 \hat{Q}_1 = Z_1 \\
   *       & A_2 Q_2 + B_1^T Q_1 - C_1 \hat{Q}_1 = 0
   *     \end{aligned}
   *   \right.
   * \f]
   * Instead of computing \f$Z_1\f$ directly, we use
   * \f$Z_1 = B_1 \zeta\f$. Meanwhile, we want to conserve
   * the numerical flux across element edges, i.e. \f$
   * \widehat {\mathsf{q_2}} \cdot \mathbf n =
   * (q_2 \mathbf I) \cdot \mathbf n +
   * \boldsymbol \tau (\mathbf q_1 - \hat {\mathbf q}_1)
   * \f$ should be continuous across faces
   * (refer to \ref subsec_02_02 "theory section" for more info).
   * Or, \f$ \langle[\![ \widehat{\mathsf q_2}
   *  \cdot \mathbf n ]\!] ,\mu \rangle = 0\f$:
   * \f[
   *   \langle q_2 \mathbf n , \boldsymbol \mu \rangle
   *   + \langle \tau {\mathbf q}_1 , \boldsymbol \mu \rangle
   *   - \langle \tau \hat {\mathbf q}_1, \boldsymbol \mu \rangle
   *   = 0
   * \f]
   * As a result, we construct the following matrices:
   * \f[
   *   \begin{gathered}
   *     d_1^T(q_2 , \boldsymbol \mu)
   *     = \langle q_2 \mathbf n , \boldsymbol \mu \rangle,
   *     \quad
   *     c_2^T(\mathbf q_1,\boldsymbol \mu)
   *     = \langle \tau \mathbf q_1 , \boldsymbol \mu \rangle,
   *     \quad
   *     e_1(\hat {\mathbf q}_1 , \boldsymbol \mu)
   *     = \langle \tau \hat{\mathbf q}_1, \boldsymbol \mu \rangle
   *   \end{gathered}
   * \f]
   */
  void calculate_matrices();

  void assign_initial_data();

  void assemble_globals(const solver_update_keys &keys_);

  template <typename T>
  double compute_internal_dofs(const double *const local_uhat_vec,
                               eigen3mat &solved_u_vec,
                               eigen3mat &solved_q_vec,
                               const poly_space_basis<T, dim> &output_basis);

  void internal_vars_errors(const eigen3mat &u_vec,
                            const eigen3mat &q_vec,
                            double &u_error,
                            double &q_error);

  void ready_for_next_iteration();

  void ready_for_next_time_step();

  std::vector<double> taus;
  /*! Shallowness parameter \f$\mu\f$ (cf \ref GN_0_0_stage2_page "examples").
   */

  static q1_func_class<dim, dealii::Tensor<1, dim> > q1_func;
  static q2_func_class<dim, double> q2_func;
  static zeta_func_class<dim, double> zeta_func;
  static grad_zeta_func_class<dim, dealii::Tensor<1, dim> > grad_zeta_func;

  hdg_model<dim, GN_eps_0_beta_0> *model;
};

#include "gn_eps_0_beta_0.cpp"

#endif
