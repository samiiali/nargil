#include "cell_class.hpp"
#include "elements/explicit_nswe.hpp"

#ifndef EXPLICIT_GN_DISPERSIVE
#define EXPLICIT_GN_DISPERSIVE

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_L_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    dealii::Tensor<1, in_point_dim> L;
    L[0] = 4. / 3. * sin(x[0]);
    L[1] = 4. / 3. * cos(x[1]);
    /*
    L[0] = (1.0 + M_PI * M_PI / 3.0) * sin(M_PI * x[0]);
    L[1] = 0.0;
    */
    return L;
  }
};

/**
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct explicit_gn_dispersive_W1_class
  : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0) const final
  {
    dealii::Tensor<1, in_point_dim, double> w1;
    w1[0] = sin(x[0]);
    w1[1] = cos(x[1]);
    /* Example 1: */
    /*
    w1[0] = sin(M_PI * x[0]);
    w1[1] = 0;
    */
    /* End of Example 1 */
    return w1;
  }
};

/**
 * ## Governing Equation
 * @ingroup cells
 * This element forms the equations for the nonlinear disperesive part of the
 * Green-Naghdi equation. We first explore the case of flat bottom and later
 * explain the formulation for the arbitrary bottom topography.
 * The equation that we want to solve can be written in the following form with
 * dimension:
 * \f[
 * \begin{cases}
 *   \partial_t h = 0 \\
 *   \partial_t(hV) - \frac 1 \alpha gh \nabla \zeta +
 *     (1+ \alpha h \mathcal T \frac 1 h)^{-1}
 *     \left[\frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)\right] = 0
 * \end{cases}
 * \f]
 * With,
 * \f[
 *   \begin{gathered}
 *     \mathcal T[]W= \mathcal R_1[](\nabla \cdot W)+
 *       \beta \mathcal R_2[](\nabla b \cdot W)
 *       \xrightarrow{b=\text{const.}}
 *       \mathcal T[]W= \mathcal R_1[](\nabla \cdot W),\\
 *     \mathcal Q_1(V) = -2 \mathcal R_1\left(\partial_1 V \cdot
 *       \partial_2 V^{\perp}+ (\nabla \cdot V)^2\right)+
 *       \beta \mathcal R_2\left(V \cdot (V \cdot \nabla)\nabla b\right)
 *       \xrightarrow{b=\text{const.}}
 *       \mathcal Q_1(V) = -2 \mathcal R_1\left(\partial_1 V \cdot
 *       \partial_2 V^{\perp}+ (\nabla \cdot V)^2\right),\\
 *     \mathcal R_1(W) = -\frac 1 {3h} \nabla (h^3 W)
 *                       - \beta \frac h 2 W \nabla b
 *     \xrightarrow{b=\text{const.}}
 *     \mathcal R_1(W)=-\frac 1 {3h} \nabla (h^3 W),\\
 *     \mathcal R_2(W) = \frac 1 {2h} \nabla (h^2 W) + \beta W \nabla b
 *     \xrightarrow{b=\text{const.}}
 *     \mathcal R_2(W) = \frac 1 {2h} \nabla (h^2 W).
 *   \end{gathered}
 * \f]
 * This equaion should be solved in two steps:
 * 1. Solve \f$(1+\alpha h \mathcal T \frac 1 h) W_1 =
 *   \frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)\f$
 *   based on the values of \f$\zeta\f$ and \f$V\f$ of the last step
 *   and taking \f$h\f$ constant.
 * 2. Solve \f$\partial_t (hV) - \frac 1 \alpha gh \nabla \zeta + W_1 = 0\f$,
 *    with \f$W_1\f$ taken from the last step.
 *
 * ### 1. Solving the static differential operator
 * The static part of the problem, i.e. \f$(1+\alpha h \mathcal T \frac 1 h) W_1
 * =
 * \frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)\f$, results in the
 * following
 * extended form:
 * \f[
 *   \begin{aligned}
 *   W_1 + \alpha h \mathcal T (\frac 1 h W)
 *     &= W_1 + \alpha h \mathcal R_1(\nabla \cdot \frac 1 h W_1) \\
 *     &= W_1 - \frac \alpha 3 \nabla (h^3 \nabla \cdot (\frac 1 h W_1))
 *   \end{aligned}
 * \f]
 * \f[
 *   \begin{aligned}
 *   \frac 1 \alpha gh \nabla \zeta + h \mathcal Q_1(V)
 *     &= \frac 1 \alpha gh \nabla \zeta - 2h \mathcal R_1
 *        (\partial_1 V\cdot \partial_2 V^\perp + (\nabla \cdot V)^2) \\
 *     &= \frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2)
 *   \end{aligned}
 * \f]
 * We introduce a new unknown to write the above equation in terms of
 * two first order equations:
 * \f[
 *   \begin{cases}
 *     h^{-3} W_2 - \nabla \cdot (\frac 1 h W_1) = 0 \\
 *     W_1 - \frac \alpha 3 \nabla (W_2) = \frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2)
 *   \end{cases}
 * \f]
 * We want to satisfy this equation in weak sense; hence, we require for all
 * \f$p_1, p_2\f$ in the space of the appropriate test functions:
 * \f[
 *   \begin{aligned}
 *   (h^{-3} W_2, p_2) - \langle \frac 1 h {\hat W}_1 \cdot n , p_2 \rangle
 *              + (\frac 1 h W_1, \nabla p_2) &= 0 \\
 *   (W_1, p_1) - \frac \alpha 3 \langle W_2^* \cdot n , p_1 \rangle
 *              + \frac \alpha 3 (W_2, \nabla \cdot p_1)
 *     &= \left(\frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2), p_1 \right)
 *   \end{aligned}
 * \f]
 * With the following definition for the numerical flux \f$W_2^*\f$:
 * \f[
 *   W_2^* \cdot n= W_2 I \cdot n + \tau (W_1 - \hat W_1)
 * \f]
 * We will get:
 * \f[
 *   \begin{cases}
 *   (h^{-3} W_2, p_2) - \langle \frac 1 h {\hat W}_1 \cdot n , p_2 \rangle
 *              + (\frac 1 h W_1, \nabla p_2) = 0 \\
 *   (W_1, p_1) - \frac \alpha 3 ( \nabla W_2 , p_1 )
 *              - \frac \alpha 3 \langle \tau W_1, p_1 \rangle
 *              + \frac \alpha 3 \langle \tau \hat W_1, p_1 \rangle
 *     = \left(\frac 1 \alpha gh \nabla \zeta
 *      + \frac 2 3 \nabla(\partial_1 V\cdot \partial_2 V^\perp
 *                         + (\nabla \cdot V)^2), p_1 \right)
 *   \end{cases}
 * \f]
 * \f$W_2\f$ is a scalar valued function and \f$W_1\f$ is a vector valued
 * function. Thus, \f$W_2^*\f$ and \f$\tau\f$ are 2-tensors. Defining the
 * following bilinear operators:
 * \f[
 *   \begin{gathered}
 *   a_{02} (W_2,p_2) = (h^{-3} W_2,p_2); \quad
 *   b_{01}^T (W_1, p_2) = (\frac 1 h W_1, \nabla p_2); \quad
 *   c_{01} (\hat W_1, p_2) = \langle \frac 1 h {\hat W}_1 \cdot n, p_2 \rangle;
 *   \\
 *   a_{01} (W_1, p_1) = (W_1, p_1); \quad
 *   b_{02} (W_2, p_1) = (\nabla W_2, p_1); \quad
 *   d_{01} (W_1, p_1) = \langle \tau W_1, p_1 \rangle; \quad
 *   c_{02} (\hat W_1,p_1) = \langle \tau \hat W_1, p_1 \rangle.
 *   \end{gathered}
 * \f]
 * So, we obtain:
 * \f[
 * \begin{cases}
 *   A_{02} W_2 + B_{01}^T W_1 - C_{01} \hat W_1 = 0 \\
 *   A_{01} W_1 - \frac \alpha 3 B_{02} W_2 - \frac \alpha 3 D_{01} W_1
 *              + \frac \alpha 3 C_{02} \hat W_1 = L_{01}
 * \end{cases}
 * \f]
 *
 * Solving this system of equations, we will obtain \f$W_1\f$. Next, we
 * substitute \f$W_1\f$ in the second equation and solve \f$hV\f$ in the next
 * time step.
 *
 * We also want to satisfy the flux conservation condition, which means
 * \f$\langle [\![ W_2^* \cdot n ]\!], \mu\rangle = 0\f$, which means:
 * \f[
 *   \langle W_2 , \mu \cdot n \rangle
 *   + \langle \tau W_1 , \mu \rangle
 *   - \langle \tau \hat W_1 , \mu \rangle = 0
 * \f]
 * Which requires the following bilinear forms:
 * \f[
 *   C_{03}^T (W_2, \mu) = \langle W_2, \mu \cdot n \rangle ; \quad
 *   E_{01} (\hat W_1 , \mu) = \langle \tau \hat W_1 , \mu \rangle.
 * \f]
 *
 * to get:
 *
 * \f[
 *   C_{03}^T W_2 + C_{02}^T W_1 - E_{01} \hat W_1 = 0.
 * \f]
 *
 *
 * ### 2. Solving the transient part
 * After we substitute \f$W_1\f$ from the above solution, we get:
 * \f[
 *   \partial_t (h V) = \frac 1 \alpha g h \nabla \zeta - W_1
 * \f]
 * Hence, we can go to the next stage or time step.
 */
template <int dim>
struct explicit_gn_dispersive : public GenericCell<dim>
{
  const double alpha = 1.0;
  using elem_basis_type = typename GenericCell<dim>::elem_basis_type;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;
  typedef dealii::Tensor<1, dim + 1, double> nswe_vec;

  static solver_options required_solver_options();
  static solver_type required_solver_type();
  static unsigned get_num_dofs_per_node();

  /**
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
  explicit_gn_dispersive() = delete;
  explicit_gn_dispersive(const explicit_gn_dispersive &inp_cell) = delete;
  explicit_gn_dispersive(explicit_gn_dispersive &&inp_cell) noexcept;
  explicit_gn_dispersive(
    typename GenericCell<dim>::dealiiCell &inp_cell,
    const unsigned &id_num_,
    const unsigned &poly_order_,
    hdg_model_with_explicit_rk<dim, explicit_gn_dispersive> *model_);
  ~explicit_gn_dispersive() final;
  eigen3mat A02, B01T, C01, A01, B02, D01, C02, L01, C03T, E01;
  void assign_BCs(const bool &at_boundary,
                  const unsigned &i_face,
                  const dealii::Point<dim> &face_center);

  void calculate_matrices();

  void calculate_stage_matrices();

  void assign_initial_data();

  void set_previous_step_results(eigen3mat *last_step_q);

  eigen3mat *get_previous_step_results();

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

  double get_iteration_increment_norm(const double *const local_uhat);

  void ready_for_next_time_step();

  void ready_for_next_stage();

  void ready_for_next_iteration();

  std::vector<double> taus;
  hdg_model_with_explicit_rk<dim, explicit_gn_dispersive> *model;
  explicit_RKn<4, original_RK> *time_integrator;

  static explicit_nswe_qis_func_class<dim, nswe_vec> explicit_gn_dispersive_qs;
  static explicit_gn_dispersive_L_class<dim, dealii::Tensor<1, dim> >
    explicit_gn_dispersive_L;
  static explicit_gn_dispersive_W1_class<dim, dealii::Tensor<1, dim> >
    explicit_gn_dispersive_W1;

  eigen3mat last_step_q;
  eigen3mat last_stage_q;

  std::vector<eigen3mat> ki_s;
};

#include "explicit_gn_dispersive.cpp"

#endif
