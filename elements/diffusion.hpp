#include "cell_class.hpp"

#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

static double eps_inv = 1.E9;

template <int dim>
Eigen::Matrix<double, 2, 1> b_magnetic_field(const dealii::Point<dim> &x)
{
  Eigen::Matrix<double, 2, 1> b;
  if (false) // Francois's first example
  {
    Eigen::Matrix<double, 2, 1> B;
    B << M_PI * cos(M_PI * x[0]) * sin(M_PI * x[1]),
      -M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]);
    b = B / sqrt(B.squaredNorm());
  } // End of Francois's first example

  if (true) // Francois's second example
  {
    double x0 = x[0];
    double y0 = x[1];
    Eigen::Matrix<double, 2, 1> B;
    B << y0 *
           (1.0454094304027911248139427451111 - 6.344929669275584 * pow(x0, 4) -
            3.045900221537398 * pow(y0, 2) + 0.7139527551314367 * pow(y0, 4) +
            pow(x0, 2) *
              (4.9974868672199735 + 3.1052605622150047 * pow(y0, 2)) +
            pow(x0, 2) * (9.137700664612193 + 5.354645663485775 * pow(x0, 2) -
                          7.139527551314367 * pow(y0, 2)) *
              log(x0)),
      x0 *
        (1.0924999150635792 - 1.9208977712867574 * pow(x0, 4) -
         9.566337199526071 * pow(y0, 2) + 0.23225160672109002 * pow(y0, 4) +
         pow(x0, 2) * (0.7918182674569383 + 10.012536506808281 * pow(y0, 2)) +
         (0.8904094304027912 + 1.3386614158714438 * pow(x0, 4) -
          9.137700664612193 * pow(y0, 2) + 3.5697637756571834 * pow(y0, 4) +
          pow(x0, 2) * (4.568850332306096492013787610749 -
                        10.70929132697155 * pow(y0, 2))) *
           log(x0));
    b = B / sqrt(B.squaredNorm());
  } // End of Francois's second example
  return b;
}

/*!
 * \brief
 * This class gives the values of the diffusivity tensor \f$\kappa_{ij}\f$ at
 * any given point.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 * \ingroup input_data_group
 */
template <int in_point_dim, typename output_type>
struct kappa_inv_class : public Function<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &) const final
  {
    Eigen::Matrix2d kappa_inv_;
    if (false) // Example 1
    {
      kappa_inv_ << 1.0 / exp(x[0] + x[1]), 0.0, 0.0, 1.0 / exp(x[0] - x[1]);
    } // End of example 1

    if (true) // Francois's Example 1 and 2 //
    {
      Eigen::Matrix<double, 2, 1> b = b_magnetic_field(x);
      Eigen::Matrix2d kappa =
        Eigen::Matrix2d::Identity() + (eps_inv - 1) * b * b.transpose();
      kappa_inv_ = kappa.inverse();
    } // End of example 1 //

    return kappa_inv_;
  }
};

template <int in_point_dim>
struct tau_func_class : public TimeFunction<in_point_dim, double>
{
  kappa_inv_class<in_point_dim, eigen3mat> kappa_inv_;
  virtual double value(const dealii::Point<in_point_dim> &x,
                       const dealii::Point<in_point_dim> &n,
                       const double & = 0.) const final
  {
    double tau = 5E4;
    if (true) // Francios's example //
    {
      Eigen::Matrix<double, 2, 1> b = b_magnetic_field(x);
      Eigen::Matrix<double, 2, 1> vec1, vec2;
      vec1 << -b[1], b[0]; // This direction corresponds to eigenvalue = 1
      vec2 << b[0], b[1];  // This direction corresponds to eigenvalue = eps_inv
      double vec1_dot_n = fabs(vec1[0] * fabs(n[0]) + vec1[1] * fabs(n[1]));
      double vec2_dot_n = fabs(vec1[0] * fabs(n[0]) + vec1[1] * fabs(n[1]));
      tau = vec1_dot_n * 1 + vec2_dot_n * tau;
    } // End of Francois's example

    return tau;
  }
};

/*!
 * \ingroup input_data_group
 * \brief This class gives the values of the analytical \f$u\f$ at a given
 * point.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 */
template <int in_point_dim>
struct u_func_class : public TimeFunction<in_point_dim, double>
{
  virtual double value(const dealii::Point<in_point_dim> &x,
                       const dealii::Point<in_point_dim> &,
                       const double & = 0.) const final
  {
    double u_func = 0;

    if (in_point_dim == 2)
    {
      if (false) // Example 1
        u_func = sin(M_PI * x[0]) * cos(M_PI * x[1]);

      if (false) // Francois's Example 1
        u_func = cos(M_PI * x[0]) * cos(M_PI * x[1]);
    }
    if (in_point_dim == 3)
    {
      if (false)
        u_func = sin(M_PI * x[0]) * cos(M_PI * x[1]) * sin(M_PI * x[2]);
    }

    return u_func;
  }
};

/*!
 * \ingroup input_data_group
 * \brief This class gives the values of the analytical \f$q_i\f$ at a given
 * point.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 */
template <int in_point_dim, typename output_type>
struct q_func_class : public TimeFunction<in_point_dim, output_type>
{
  virtual output_type value(const dealii::Point<in_point_dim> &x,
                            const dealii::Point<in_point_dim> &,
                            const double & = 0.0) const final
  {
    dealii::Tensor<1, in_point_dim> q_func;
    if (in_point_dim == 2)
    {
      q_func[0] =
        -exp(x[0] + x[1]) * M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]);
      q_func[1] = exp(x[0] - x[1]) * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
    }
    if (in_point_dim == 3)
    {
      q_func[0] =
        -M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]) * sin(M_PI * x[2]);
      q_func[1] = M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]) * sin(M_PI * x[2]);
      q_func[2] =
        -M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) * cos(M_PI * x[2]);
    }

    return q_func;
  }
};

/*!
 * \ingroup input_data_group
 * \brief This class gives the values of the analytical \f$(q_i)_{,i}\f$ at
 * a given point.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 */
template <int in_point_dim>
struct divq_func_class : public Function<in_point_dim, double>
{
  virtual double value(const dealii::Point<in_point_dim> &x,
                       const dealii::Point<in_point_dim> &) const final
  {
    if (in_point_dim == 3)
      return 3 * M_PI * M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) *
             sin(M_PI * x[2]);
    return 2 * M_PI * M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
  }
};

/*!
 * \ingroup input_data_group
 * \brief This class gives the values of the analytical \f$f\f$ at a given
 * point.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 */
template <int in_point_dim>
struct f_func_class : public TimeFunction<in_point_dim, double>
{
  virtual double value(const dealii::Point<in_point_dim> &x,
                       const dealii::Point<in_point_dim> &,
                       const double & = 0.) const final
  {
    double f_func = 0;
    if (in_point_dim == 2)
    {
      if (false)
      {
        f_func = M_PI * M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) *
                   (exp(x[0] + x[1]) + exp(x[0] - x[1])) -
                 M_PI * exp(x[0] + x[1]) * cos(M_PI * x[0]) * cos(M_PI * x[1]) -
                 M_PI * exp(x[0] - x[1]) * sin(M_PI * x[0]) * sin(M_PI * x[1]);
      }

      if (false) // Francois's first example
      {
        f_func = -2 * M_PI * M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]);
        f_func = -f_func;
      }

      if (true) // Francois's second example
      {
        double x0 = x[0];
        double y0 = x[1];
        double psi0 =
          0.0864912785478786 - 0.3573346678775552 * pow(x0, 6) -
          0.5227047152013956 * pow(y0, 2) + 0.7614750553843495 * pow(y0, 4) -
          0.11899212585523944 * pow(y0, 6) +
          pow(x0, 4) * (-0.08759857890489645 + 3.172464834637792 * pow(y0, 2)) +
          pow(x0, 2) * (0.32364759993109177 - 2.4987434336099867 * pow(y0, 2) -
                        0.7763151405537512 * pow(y0, 4)) +
          pow(x0, 2) *
            (0.4452047152013956 + 0.22311023597857396 * pow(x0, 4) -
             4.568850332306097 * pow(y0, 2) + 1.7848818878285917 * pow(y0, 4) +
             pow(x0, 2) *
               (1.1422125830765242 - 2.6773228317428877 * pow(y0, 2))) *
            log(x0);
        f_func = 0.0;
        if (psi0 > 0.)
          f_func = 0.9 * pow(psi0, 1. / 3.);
        if (psi0 < 0.)
          f_func = -0.9 * pow(-psi0, 1. / 3.);
        f_func = -f_func;
      }
    }
    if (in_point_dim == 3)
      if (false) // A three dim example
        f_func = 3 * M_PI * M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) *
                 sin(M_PI * x[2]);

    return f_func;
  }
};

/*!
 * \ingroup input_data_group
 * \brief This class gives the values of the analytical \f$g_D\f$ at a given
 * point.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 */
template <int in_point_dim>
struct dirichlet_BC_func_class : public TimeFunction<in_point_dim, double>
{
  u_func_class<in_point_dim> u_func;
  virtual double value(const dealii::Point<in_point_dim> &x,
                       const dealii::Point<in_point_dim> &,
                       const double & = 0.0) const final
  {
    double gD;

    if (false) // Define gD based on u
      gD = u_func.value(x, x);

    if (false) // One of examples
      gD = 0;

    if (true) // Francois's second example
      gD = 10.0;

    return gD;
  }
};

/*!
 * \ingroup input_data_group
 * \brief This class gives the values of the analytical \f$g_N\f$ at a given
 * point.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 */
template <int in_point_dim>
struct neumann_BC_func_class : public Function<in_point_dim, double>
{
  virtual double value(const dealii::Point<in_point_dim> &x,
                       const dealii::Point<in_point_dim> &n) const final
  {
    q_func_class<in_point_dim, dealii::Tensor<1, in_point_dim> > q_func;
    double gN;
    if (false) // One of the examples
      gN = 0;

    gN = q_func.value(x, x) * n;

    return gN;
  }
};

/*!
 * \ingroup cells
 */
template <int dim>
struct Diffusion : public GenericCell<dim>
{
  using elem_basis_type = typename GenericCell<dim>::elem_basis_type;
  typedef std::unique_ptr<dealii::FEValues<dim> > FE_val_ptr;
  typedef std::unique_ptr<dealii::FEFaceValues<dim> > FEFace_val_ptr;

  static solver_options required_solver_options();
  static solver_type required_solver_type();
  static unsigned get_num_dofs_per_node();

  /*!
   * \brief Constructor for the DiffusionCell.
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
  Diffusion() = delete;
  Diffusion(const Diffusion &inp_cell) = delete;
  Diffusion(Diffusion &&inp_cell) noexcept;
  Diffusion(typename GenericCell<dim>::dealiiCell &inp_cell,
            const unsigned &id_num_,
            const unsigned &poly_order_,
            hdg_model<dim, Diffusion> *model_);
  ~Diffusion() final;
  eigen3mat A, B, C, D, E, H, H2, M, DM_star, DB2;
  void assign_BCs(const bool &at_boundary,
                  const unsigned &i_face,
                  const dealii::Point<dim> &face_center);

  void assign_initial_data();

  void calculate_matrices();

  void calculate_postprocess_matrices();

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

  double get_postprocessed_error_in_cell(const TimeFunction<dim, double> &func,
                                         const Eigen::MatrixXd &input_vector,
                                         const double &time = 0);

  eigen3mat ustar;
  static kappa_inv_class<dim, Eigen::MatrixXd> kappa_inv;
  static tau_func_class<dim> tau_func;
  static u_func_class<dim> u_func;
  static q_func_class<dim, dealii::Tensor<1, dim> > q_func;
  static divq_func_class<dim> divq_func;
  static f_func_class<dim> f_func;
  static dirichlet_BC_func_class<dim> dirichlet_bc_func;
  static neumann_BC_func_class<dim> Neumann_BC_func;

  hdg_model<dim, Diffusion> *model;
};

#include "diffusion.cpp"

#endif
