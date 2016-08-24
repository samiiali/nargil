#include "solution_manager.hpp"
#include "support_classes.hpp"

#ifndef GENERIC_MODEL_HPP
#define GENERIC_MODEL_HPP

template <int dim>
struct SolutionManager;

template <int dim, template <int> class CellType>
struct generic_model
{
  generic_model(SolutionManager<dim> *const sol_);
  virtual ~generic_model();

  SolutionManager<dim> *const manager;
  unsigned comm_rank;

  dealii::DoFHandler<dim> DoF_H_Refine;
  dealii::DoFHandler<dim> DoF_H_System;

  unsigned n_global_DOFs_rank_owns;
  unsigned n_global_DOFs_on_all_ranks;
  unsigned n_local_DOFs_on_this_rank;
  std::vector<int> n_local_DOFs_connected_to_DOF;
  std::vector<int> n_nonlocal_DOFs_connected_to_DOF;
  std::vector<int> scatter_from, scatter_to;

  unsigned get_global_mat_block_size();
};

#include "generic_model.cpp"

#include "hdg_model_with_explicit_rk.hpp"
#include "explicit_hdg_model.hpp"
#include "hdg_model.hpp"

#endif
