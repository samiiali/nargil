#include "hdg_model_with_explicit_rk.hpp"

template <int dim, template <int> class CellType>
hdg_model_with_explicit_rk<dim, CellType>::hdg_model_with_explicit_rk(
  SolutionManager<dim> *const sol_,
  explicit_RKn<4, original_RK> *time_integrator_)
  : generic_model<dim, CellType>(sol_),
    poly_order(this->manager->poly_order),
    n_faces_per_cell(this->manager->n_faces_per_cell),
    DG_Elem(poly_order),
    DG_System(DG_Elem, 1 + dim),
    time_integrator(time_integrator_)
{
}

/*!
 * \brief We construct an \c std::vector, which contains object of type
 * GenericCell. These represent active cells in the mesh.
 * \details Each object of GenericCell type contains an rvalue reference
 * (of type iterator)
 * to one of the dealii cells in the mesh (GenericCell::dealii_Cell_Type).
 * The vector SolutionManager::all_owned_cells contains all of the active
 * cells in the current processor. We also fill SolutionManager::cell_ID_to_num
 * with a counter of locally owned cells.
 */
template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::init_mesh_containers()
{
  all_owned_cells.reserve(
    this->manager->the_grid.n_locally_owned_active_cells());
  unsigned n_cell = 0;
  this->manager->n_ghost_cell = 0;
  this->manager->n_owned_cell = 0;
  for (dealiiCell &&cell : this->DoF_H_System.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      all_owned_cells.push_back(GenericCell<dim>::template make_cell<CellType>(
        cell, this->manager->n_owned_cell, poly_order, this));
      this->manager->cell_ID_to_num[all_owned_cells.back()->cell_id] =
        this->manager->n_owned_cell;
      ++this->manager->n_owned_cell;
    }
    if (cell->is_ghost())
      ++this->manager->n_ghost_cell;
    ++n_cell;
  }
}

/*!
 * \brief We apply the boundary conditions on all of the
 * boundary faces of the model here.
 * \details The boundary conditions for
 * different solved examples are explained in \ref GN_0_0_stage2_page
 * "numerical examples page". The boundary conditions should be
 * applied on all of the boudnary faces, either ghost or locally
 * owned.
 */
template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::set_boundary_indicator()
{
  for (auto &&cell : all_owned_cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      auto &&face = cell->dealii_cell->face(i_face);
      static_cast<CellType<dim> *>(cell.get())
        ->assign_BCs(face->at_boundary(), i_face, face->center());
    }
  }
}

/*!
 * \brief Counts the number of unknowns in the innerCPU and interCPU
 * spaces, for each rank.
 * \details By global we mean those unknowns that we form a global
 * system for them. This does not mean the interCPU unknowns. Also,
 * we form two vectors which contain the unknown ID of the each
 * active DOF, in the innerCPU and interCPU spaces.
 */
template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::count_globals()
{
  unsigned n_face_bases = pow(poly_order + 1, dim - 1);
  std::vector<std::unique_ptr<GenericCell<dim> > > all_ghost_cells;
  all_ghost_cells.reserve(this->manager->n_ghost_cell);
  std::map<std::string, int> ghost_ID_to_num;
  unsigned ghost_cell_counter = 0;
  for (dealiiCell &&cell : this->DoF_H_System.active_cell_iterators())
  {
    if (cell->is_ghost())
    {
      all_ghost_cells.push_back(
        std::move(GenericCell<dim>::template make_cell<CellType>(
          cell, ghost_cell_counter, poly_order, this)));
      std::stringstream ss_id;
      ss_id << cell->id();
      std::string str_id = ss_id.str();
      ghost_ID_to_num[str_id] = ghost_cell_counter;
      ++ghost_cell_counter;
    }
  }
  for (auto &&cell : all_ghost_cells)
  {
    std::vector<dealii::Point<dim> > face_centers(n_faces_per_cell);
    std::vector<bool> faces_at_boundary(n_faces_per_cell, false);
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      auto &&face = cell->dealii_cell->face(i_face);
      static_cast<CellType<dim> *>(cell.get())
        ->assign_BCs(face->at_boundary(), i_face, face->center());
    }
  }

  std::map<unsigned, std::vector<std::string> > face_to_rank_sender;
  std::map<unsigned, unsigned> face_to_rank_recver;
  unsigned local_dof_num_on_this_rank = 0;
  unsigned global_dof_num_on_this_rank = 0;
  unsigned mpi_request_counter = 0;
  unsigned mpi_status_counter = 0;
  std::map<unsigned, bool> is_there_a_msg_from_rank;

  /*!
   * <b>Notes for developer 1:</b>
   * Here, we also count the faces of the model in innerCPU and interCPU
   * spaces, which are connected to the cells owned by this rank. By
   * innerCPU, we mean those faces counted in the subdomain of current rank.
   * By interCPU, we mean those faces which are counted as parts of other
   * subdomains. We have two rules for this:
   *   - rule 1: If a face is common between two subdomains, and one side is
   *         coarser than the other side. This face belongs to the coarser
   *         side; no matter which subdomain has smaller rank.
   *   - rule 2: If a face is connected to two elements of the same refinement
   *         level, and the elements are in two different subdomains, then
   *         the face belongs to the subdomain with smaller rank.
   */
  for (std::unique_ptr<GenericCell<dim> > &cell : all_owned_cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      if (cell->dofs_ID_in_this_rank[i_face].size() == 0)
      {
        const auto &face_i1 = cell->dealii_cell->face(i_face);
        /* The basic case corresponds to face_i1 being on the boundary.
         * In this case we only need to set the number of current face,
         * and we do not bother to know what is going on, on the other
         * side of this face.  You might wonder why I am not thinking
         * about the case that GenericCell::BCs are set equal to
         * GenericCell::essential here. The reason is that inside the
         * assignment function here, we assign dof numbers to those
         * dofs that have some dof_names for themselves.
         */
        if (face_i1->at_boundary())
        {
          if (cell->BCs[i_face] != GenericCell<dim>::periodic)
          {
            cell->assign_local_global_cell_data(i_face,
                                                local_dof_num_on_this_rank,
                                                global_dof_num_on_this_rank,
                                                this->comm_rank,
                                                0);
            local_dof_num_on_this_rank +=
              cell->dof_names_on_faces[i_face].count();
            global_dof_num_on_this_rank +=
              cell->dof_names_on_faces[i_face].count();
          }
          if (cell->BCs[i_face] == GenericCell<dim>::periodic)
          {
          }
        }
        else
        {
          /* At this point, we are sure that the cell has a neighbor. We will
           * have three cases:
           *
           * 1- The neighbor is coarser than the cell. This can only happen if
           *    the neighbor is a ghost cell, otherwise there is something
           *    wrong. So, when the neighbor is ghost, this subdomain does not
           *    own the face. Hence, we have to take the face number from the
           *    corresponding neighboer.
           *
           * 2- The neighbor is finer. In this case the face is owned by this
           *    subdomain, but we will have two subcases:
           *   2a- If the neighbor is in this subdomain, we act as if the domain
           *       was not decomposed.
           *   2b- If the neighbor is in some other subdomain, we have to also
           *       send the face number to all those finer neighbors, along with
           *       the corresponding subface id.
           *
           * 3- The face has neighbors of same refinement. This case is somehow
           *    trichier than what is looks. Because, you have to decide where
           *    face belongs to. As we said before, the face belongs to the
           *    domain which has smaller rank. So, we have to send the face
           *    number from the smaller rank to the higher rank.
           */
          if (cell->dealii_cell->neighbor_is_coarser(i_face))
          {
            /*
             * The neighbor should be a ghost, because in each subdomain, the
             * elements are ordered from coarse to fine.
             */
            dealiiCell &&nb_i1 = cell->dealii_cell->neighbor(i_face);
            assert(nb_i1->is_ghost());
            unsigned face_nb_num = cell->dealii_cell->neighbor_face_no(i_face);
            const auto &face_nb = nb_i1->face(face_nb_num);
            /*!
             * \bug I believe, nb_face_of_nb_num = i_face. Otherwise, something
             * is wrong. I do not change it now, but I will do later.
             */
            unsigned nb_face_of_nb_num = nb_i1->neighbor_face_no(face_nb_num);
            for (unsigned i_nb_subface = 0;
                 i_nb_subface < face_nb->n_children();
                 ++i_nb_subface)
            {
              const dealiiCell &nb_of_nb_i1 =
                nb_i1->neighbor_child_on_subface(face_nb_num, i_nb_subface);
              if (nb_of_nb_i1->subdomain_id() == this->comm_rank)
              {
                unsigned nb_of_nb_num = this->manager->cell_id_to_num_finder(
                  nb_of_nb_i1, this->manager->cell_ID_to_num);
                all_owned_cells[nb_of_nb_num]->assign_local_cell_data(
                  nb_face_of_nb_num,
                  local_dof_num_on_this_rank,
                  nb_i1->subdomain_id(),
                  i_nb_subface + 1);
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
            local_dof_num_on_this_rank +=
              cell->dof_names_on_faces[i_face].count();
          }
          else if (face_i1->has_children())
          {
            cell->assign_local_global_cell_data(i_face,
                                                local_dof_num_on_this_rank,
                                                global_dof_num_on_this_rank,
                                                this->comm_rank,
                                                0);
            for (unsigned i_subface = 0;
                 i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              dealiiCell &&nb_i1 =
                cell->dealii_cell->neighbor_child_on_subface(i_face, i_subface);
              int face_nb_i1 = cell->dealii_cell->neighbor_face_no(i_face);
              std::stringstream nb_ss_id;
              nb_ss_id << nb_i1->id();
              std::string nb_str_id = nb_ss_id.str();
              if (nb_i1->subdomain_id() == this->comm_rank)
              {
                assert(this->manager->cell_ID_to_num.find(nb_str_id) !=
                       this->manager->cell_ID_to_num.end());
                int nb_i1_num = this->manager->cell_ID_to_num[nb_str_id];
                all_owned_cells[nb_i1_num]->assign_local_global_cell_data(
                  face_nb_i1,
                  local_dof_num_on_this_rank,
                  global_dof_num_on_this_rank,
                  this->comm_rank,
                  i_subface + 1);
              }
              else
              {
                /* Here, we are sure that the face is not owned by this rank.
                 * Also, we know our cell is coarser than nb_i1.
                 * Hence, we do not bother to know if the rank of neighbor
                 * subdomain is greater or smaller than the current rank.
                 */
                assert(nb_i1->is_ghost());
                assert(ghost_ID_to_num.find(nb_str_id) !=
                       ghost_ID_to_num.end());
                unsigned nb_i1_num = ghost_ID_to_num[nb_str_id];
                all_ghost_cells[nb_i1_num]->assign_local_global_cell_data(
                  face_nb_i1,
                  local_dof_num_on_this_rank,
                  global_dof_num_on_this_rank,
                  this->comm_rank,
                  i_subface + 1);
                /* Now we send id, face id, subface id, and neighbor face number
                 * to the corresponding rank. */
                char buffer[300];
                std::snprintf(buffer,
                              300,
                              "%s#%d#%d#%d",
                              nb_str_id.c_str(),
                              face_nb_i1,
                              i_subface + 1,
                              global_dof_num_on_this_rank);
                face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
                ++mpi_request_counter;
              }
            }
            local_dof_num_on_this_rank +=
              cell->dof_names_on_faces[i_face].count();
            global_dof_num_on_this_rank +=
              cell->dof_names_on_faces[i_face].count();
          }
          else
          {
            dealiiCell &&nb_i1 = cell->dealii_cell->neighbor(i_face);
            int face_nb_i1 = cell->dealii_cell->neighbor_face_no(i_face);
            std::stringstream nb_ss_id;
            nb_ss_id << nb_i1->id();
            std::string nb_str_id = nb_ss_id.str();
            if (nb_i1->subdomain_id() == this->comm_rank)
            {
              assert(this->manager->cell_ID_to_num.find(nb_str_id) !=
                     this->manager->cell_ID_to_num.end());
              int nb_i1_num = this->manager->cell_ID_to_num[nb_str_id];
              cell->assign_local_global_cell_data(i_face,
                                                  local_dof_num_on_this_rank,
                                                  global_dof_num_on_this_rank,
                                                  this->comm_rank,
                                                  0);
              all_owned_cells[nb_i1_num]->assign_local_global_cell_data(
                face_nb_i1,
                local_dof_num_on_this_rank,
                global_dof_num_on_this_rank,
                this->comm_rank,
                0);
              global_dof_num_on_this_rank +=
                cell->dof_names_on_faces[i_face].count();
            }
            else
            {
              assert(nb_i1->is_ghost());
              if (nb_i1->subdomain_id() > this->comm_rank)
              {
                cell->assign_local_global_cell_data(i_face,
                                                    local_dof_num_on_this_rank,
                                                    global_dof_num_on_this_rank,
                                                    this->comm_rank,
                                                    0);
                assert(ghost_ID_to_num.find(nb_str_id) !=
                       ghost_ID_to_num.end());
                unsigned nb_i1_num = ghost_ID_to_num[nb_str_id];
                all_ghost_cells[nb_i1_num]->assign_local_global_cell_data(
                  face_nb_i1,
                  local_dof_num_on_this_rank,
                  global_dof_num_on_this_rank,
                  this->comm_rank,
                  0);
                /* Now we send id, face id, subface(=0), and neighbor face
                 * number to the corresponding rank. */
                char buffer[300];
                std::snprintf(buffer,
                              300,
                              "%s#%d#%d#%d",
                              nb_str_id.c_str(),
                              face_nb_i1,
                              0,
                              global_dof_num_on_this_rank);
                face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
                global_dof_num_on_this_rank +=
                  cell->dof_names_on_faces[i_face].count();
                ++mpi_request_counter;
              }
              else
              {
                cell->assign_local_cell_data(
                  i_face, local_dof_num_on_this_rank, nb_i1->subdomain_id(), 0);
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
            local_dof_num_on_this_rank +=
              cell->dof_names_on_faces[i_face].count();
          }
        }
      }
    }
  }
  /*
   * The next two variables contain num faces from rank zero to the
   * current rank, including and excluding current rank
   */
  std::vector<unsigned> dofs_count_be4_rank(this->manager->comm_size, 0);
  std::vector<unsigned> dofs_count_up2_rank(this->manager->comm_size, 0);
  unsigned n_dofs_this_rank_owns = global_dof_num_on_this_rank;
  MPI_Allgather(&n_dofs_this_rank_owns,
                1,
                MPI_UNSIGNED,
                dofs_count_up2_rank.data(),
                1,
                MPI_UNSIGNED,
                this->manager->comm);

  for (unsigned i_num = 0; i_num < this->manager->comm_size; ++i_num)
    for (unsigned j_num = 0; j_num < i_num; ++j_num)
      dofs_count_be4_rank[i_num] += dofs_count_up2_rank[j_num];

  for (std::unique_ptr<GenericCell<dim> > &cell : all_owned_cells)
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
      for (unsigned i_dof = 0;
           i_dof < cell->dofs_ID_in_all_ranks[i_face].size();
           ++i_dof)
        cell->dofs_ID_in_all_ranks[i_face][i_dof] +=
          dofs_count_be4_rank[this->comm_rank];

  for (std::unique_ptr<GenericCell<dim> > &ghost_cell : all_ghost_cells)
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
      for (unsigned i_dof = 0;
           i_dof < ghost_cell->dofs_ID_in_all_ranks[i_face].size();
           ++i_dof)
        ghost_cell->dofs_ID_in_all_ranks[i_face][i_dof] +=
          dofs_count_be4_rank[this->comm_rank];

  for (auto &&cell : all_owned_cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      if (cell->BCs[i_face] == GenericCell<dim>::periodic)
      {
      }
    }
  }

  /*
   * Now, we want to also assign a unique number to those faces which
   * do not belong to the current rank. This includes those faces which
   * are connected to a locally owned cell or not. We start ghost face
   * numbers from -10, and go down.
   */
  int ghost_dofs_counter = -10;
  for (std::unique_ptr<GenericCell<dim> > &ghost_cell : all_ghost_cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      if (ghost_cell->dofs_ID_in_this_rank[i_face].size() == 0)
      {
        const auto &face_i1 = ghost_cell->dealii_cell->face(i_face);
        /* The basic case corresponds to face_i1 being on the boundary.
         * In this case we only need to set the number of current face,
         * and we do not bother to know what is going on, on the other
         * side of this face. Because (in the current version), that is
         * the way things are working ! */
        if (face_i1->at_boundary())
        {
          {
            ghost_cell->assign_ghost_cell_data(
              i_face,
              ghost_dofs_counter,
              ghost_dofs_counter,
              ghost_cell->dealii_cell->subdomain_id(),
              0);
            ghost_dofs_counter -=
              ghost_cell->dof_names_on_faces[i_face].count();
          }
        }
        else
        {
          /*
           * We are sure that the face that we are on, is either on the coarser
           * side of an owned cell, or belongs to a lower rank than thr current
           * rank.
           */
          ghost_cell->assign_ghost_cell_data(
            i_face,
            ghost_dofs_counter,
            ghost_dofs_counter,
            ghost_cell->dealii_cell->subdomain_id(),
            0);
          if (face_i1->has_children())
          {
            int face_nb_subface =
              ghost_cell->dealii_cell->neighbor_face_no(i_face);
            for (unsigned i_subface = 0;
                 i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              dealiiCell &&nb_subface =
                ghost_cell->dealii_cell->neighbor_child_on_subface(i_face,
                                                                   i_subface);
              if (nb_subface->is_ghost())
              {
                unsigned nb_subface_num = this->manager->cell_id_to_num_finder(
                  nb_subface, ghost_ID_to_num);
                all_ghost_cells[nb_subface_num]->assign_ghost_cell_data(
                  face_nb_subface,
                  ghost_dofs_counter,
                  ghost_dofs_counter,
                  nb_subface->subdomain_id(),
                  i_subface + 1);
              }
            }
          }
          else if (ghost_cell->dealii_cell->neighbor(i_face)->is_ghost())
          {
            dealiiCell &&nb_i1 = ghost_cell->dealii_cell->neighbor(i_face);
            int face_nb_i1 = ghost_cell->dealii_cell->neighbor_face_no(i_face);
            unsigned nb_i1_num =
              this->manager->cell_id_to_num_finder(nb_i1, ghost_ID_to_num);
            assert(all_ghost_cells[nb_i1_num]
                     ->dofs_ID_in_this_rank[face_nb_i1]
                     .size() == 0);
            assert(all_ghost_cells[nb_i1_num]
                     ->dofs_ID_in_all_ranks[face_nb_i1]
                     .size() == 0);
            all_ghost_cells[nb_i1_num]->assign_ghost_cell_data(
              face_nb_i1,
              ghost_dofs_counter,
              ghost_dofs_counter,
              nb_i1->subdomain_id(),
              0);
          }
          ghost_dofs_counter -= ghost_cell->dof_names_on_faces[i_face].count();
        }
      }
    }
  }

  /*
   * Here we start the interCPU communications. According to almost
   * every specification, to avoid high communication overhead, we
   * should perform all send/recv's in one go. To gain more control, we
   * do this process in two phases:
   *
   *   - Phase 1: We are sending from lower ranks to higher ranks.
   *              Hence, higher ranks skip the sending loop and lower
   *              ranks skip the recv loop.
   *   - Phase 2: We send from higher ranks to lower ranks. Hence,
   *              the lower ranks will stop the sending loop.
   *
   * This way, we make sure that no deadlock will happen.
   */

  /*
   * Phase 1 : Send Loop
   */
  for (auto &&i_send = face_to_rank_sender.rbegin();
       i_send != face_to_rank_sender.rend();
       ++i_send)
  {
    assert(this->comm_rank != i_send->first);
    if (this->comm_rank < i_send->first)
    {
      unsigned num_sends = face_to_rank_sender[i_send->first].size();
      unsigned jth_rank_on_i_send = 0;
      std::vector<MPI_Request> all_mpi_reqs_of_rank(num_sends);
      for (auto &&msg_it : face_to_rank_sender[i_send->first])
      {
        MPI_Isend((char *)msg_it.c_str(),
                  msg_it.size() + 1,
                  MPI_CHAR,
                  i_send->first,
                  this->manager->refn_cycle,
                  this->manager->comm,
                  &all_mpi_reqs_of_rank[jth_rank_on_i_send]);
        ++jth_rank_on_i_send;
      }
      MPI_Waitall(num_sends, all_mpi_reqs_of_rank.data(), MPI_STATUSES_IGNORE);
    }
  }

  /*
   * Phase 1 : Recv Loop
   */
  std::vector<MPI_Status> all_mpi_stats_of_rank(mpi_status_counter);
  unsigned recv_counter = 0;
  bool no_msg_left = (is_there_a_msg_from_rank.size() == 0);
  while (!no_msg_left)
  {
    auto i_recv = is_there_a_msg_from_rank.begin();
    no_msg_left = true;
    for (; i_recv != is_there_a_msg_from_rank.end(); ++i_recv)
    {
      if (i_recv->second && this->comm_rank > i_recv->first)
        no_msg_left = false;
      int flag = 0;
      if (this->comm_rank > i_recv->first)
        MPI_Iprobe(i_recv->first,
                   this->manager->refn_cycle,
                   this->manager->comm,
                   &flag,
                   MPI_STATUS_IGNORE);
      if (flag)
      {
        assert(i_recv->second);
        break;
      }
    }
    if (i_recv != is_there_a_msg_from_rank.end())
    {
      for (unsigned jth_rank_on_i_recv = 0;
           jth_rank_on_i_recv < face_to_rank_recver[i_recv->first];
           ++jth_rank_on_i_recv)
      {
        char buffer[300];
        MPI_Recv(&buffer[0],
                 300,
                 MPI_CHAR,
                 i_recv->first,
                 this->manager->refn_cycle,
                 this->manager->comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        assert(this->manager->cell_ID_to_num.find(cell_unique_id) !=
               this->manager->cell_ID_to_num.end());
        int cell_number = this->manager->cell_ID_to_num[cell_unique_id];
        unsigned face_num = std::stoi(tokens[1]);
        assert(
          all_owned_cells[cell_number]->dofs_ID_in_all_ranks[face_num].size() ==
          0);
        assert(
          all_owned_cells[cell_number]->dof_names_on_faces[face_num].count() !=
          0);
        /*
         * The DOF data received from other CPU is the ID of the first
         * DOF of this face.
         */
        for (unsigned i_dof = 0;
             i_dof <
             all_owned_cells[cell_number]->dof_names_on_faces[face_num].count();
             ++i_dof)
          all_owned_cells[cell_number]
            ->dofs_ID_in_all_ranks[face_num]
            .push_back(std::stoi(tokens[3]) +
                       dofs_count_be4_rank[i_recv->first] + i_dof);
        ++recv_counter;
      }
      i_recv->second = false;
    }
  }

  /*
   * Phase 2 : Send Loop
   */
  for (auto &&i_send = face_to_rank_sender.rbegin();
       i_send != face_to_rank_sender.rend();
       ++i_send)
  {
    assert(this->comm_rank != i_send->first);
    if (this->comm_rank > i_send->first)
    {
      unsigned num_sends = face_to_rank_sender[i_send->first].size();
      unsigned jth_rank_on_i_send = 0;
      std::vector<MPI_Request> all_mpi_reqs_of_rank(num_sends);
      for (auto &&msg_it : face_to_rank_sender[i_send->first])
      {
        MPI_Isend((char *)msg_it.c_str(),
                  msg_it.size() + 1,
                  MPI_CHAR,
                  i_send->first,
                  this->manager->refn_cycle,
                  this->manager->comm,
                  &all_mpi_reqs_of_rank[jth_rank_on_i_send]);
        ++jth_rank_on_i_send;
      }
      MPI_Waitall(num_sends, all_mpi_reqs_of_rank.data(), MPI_STATUSES_IGNORE);
    }
  }

  /*
   * Phase 2 : Recv Loop
   */
  no_msg_left = (is_there_a_msg_from_rank.size() == 0);
  while (!no_msg_left)
  {
    auto i_recv = is_there_a_msg_from_rank.begin();
    no_msg_left = true;
    for (; i_recv != is_there_a_msg_from_rank.end(); ++i_recv)
    {
      if (i_recv->second && this->comm_rank < i_recv->first)
        no_msg_left = false;
      int flag = 0;
      if (this->comm_rank < i_recv->first)
        MPI_Iprobe(i_recv->first,
                   this->manager->refn_cycle,
                   this->manager->comm,
                   &flag,
                   MPI_STATUS_IGNORE);
      if (flag)
      {
        assert(i_recv->second);
        break;
      }
    }
    if (i_recv != is_there_a_msg_from_rank.end())
    {
      for (unsigned jth_rank_on_i_recv = 0;
           jth_rank_on_i_recv < face_to_rank_recver[i_recv->first];
           ++jth_rank_on_i_recv)
      {
        char buffer[300];
        MPI_Recv(&buffer[0],
                 300,
                 MPI_CHAR,
                 i_recv->first,
                 this->manager->refn_cycle,
                 this->manager->comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        assert(this->manager->cell_ID_to_num.find(cell_unique_id) !=
               this->manager->cell_ID_to_num.end());
        int cell_number = this->manager->cell_ID_to_num[cell_unique_id];
        unsigned face_num = std::stoi(tokens[1]);
        assert(
          all_owned_cells[cell_number]->dof_names_on_faces[face_num].count() !=
          0);
        assert(
          all_owned_cells[cell_number]->dofs_ID_in_all_ranks[face_num].size() ==
          0);
        /*
         * The DOF data received from other CPU is the ID of the first
         * DOF of this face.
         */
        for (unsigned i_dof = 0;
             i_dof <
             all_owned_cells[cell_number]->dof_names_on_faces[face_num].count();
             ++i_dof)
          all_owned_cells[cell_number]
            ->dofs_ID_in_all_ranks[face_num]
            .push_back(std::stoi(tokens[3]) +
                       dofs_count_be4_rank[i_recv->first] + i_dof);
        ++recv_counter;
      }
      i_recv->second = false;
    }
  }

  /*           THESE NEXT LOOPS ARE JUST FOR PETSc !!
   *
   * When you want to preallocate stiffness matrix in PETSc, it
   * accpet an argument which contains the number of DOFs connected to
   * the DOF in each row. According to PETSc, if you let PETSc know
   * about this preallocation, you will get a noticeable performance
   * boost.
   */

  /*
   * Now, we want to know, each face belonging to the current
   * rank is connected to how many faces from the current rank and
   * how many faces from other ranks. So, if for example, we are on
   * rank 1, we want to be able to count the innerFaces and
   * interFaces shown below (This is especilly a challange,
   * because the faces of elements on the right are connected to
   * faces of elements in the left via two middle ghost elements.):
   *
   *               ---------------------------
   *                rank 1 | rank 2 | rank 1
   *               --------|--------|---------
   *                rank 1 | rank 2 | rank 1
   *               ----------------------------
   *
   * To this end, Let us build a vector containing each unique
   * face which belongs to this rank (So those faces which do
   * not belong to this rank are not present in this vector !).
   * Then, fill the GenericFace::Parent_Cells vector with
   * those parent cells which also belongs to the current rank.
   * Also, we fill GenericFace::Parent_Ghosts with ghost cells
   * connected to the current face.
   */
  std::vector<GenericDOF<dim> > all_owned_dofs(global_dof_num_on_this_rank);
  for (typename GenericCell<dim>::vec_iter_ptr_type cell_it =
         all_owned_cells.begin();
       cell_it != all_owned_cells.end();
       ++cell_it)
  {
    for (unsigned i_face = 0; i_face < (*cell_it)->n_faces; ++i_face)
    {
      if ((*cell_it)->face_owner_rank[i_face] == this->comm_rank)
      {
        for (unsigned i_dof = 0;
             i_dof < (*cell_it)->dofs_ID_in_all_ranks[i_face].size();
             ++i_dof)
        {
          int dof_i1 = (*cell_it)->dofs_ID_in_all_ranks[i_face][i_dof] -
                       dofs_count_be4_rank[this->comm_rank];
          all_owned_dofs[dof_i1].parent_cells.push_back(cell_it);
          all_owned_dofs[dof_i1].connected_face_of_parent_cell.push_back(
            i_face);
          if (all_owned_dofs[dof_i1].n_local_connected_DOFs == 0)
            all_owned_dofs[dof_i1].n_local_connected_DOFs =
              (*cell_it)->dof_names_on_faces[i_face].count();
        }
      }
    }
  }

  for (typename GenericCell<dim>::vec_iter_ptr_type ghost_cell_it =
         all_ghost_cells.begin();
       ghost_cell_it != all_ghost_cells.end();
       ++ghost_cell_it)
  {
    for (unsigned i_face = 0; i_face < (*ghost_cell_it)->n_faces; ++i_face)
    {
      if ((*ghost_cell_it)->face_owner_rank[i_face] == this->comm_rank)
      {
        for (unsigned i_dof = 0;
             i_dof < (*ghost_cell_it)->dofs_ID_in_all_ranks[i_face].size();
             ++i_dof)
        {
          int dof_i1 = (*ghost_cell_it)->dofs_ID_in_all_ranks[i_face][i_dof] -
                       dofs_count_be4_rank[this->comm_rank];
          all_owned_dofs[dof_i1].parent_ghosts.push_back(ghost_cell_it);
          all_owned_dofs[dof_i1].connected_face_of_parent_ghost.push_back(
            i_face);
          if (all_owned_dofs[dof_i1].n_local_connected_DOFs == 0)
          {
            std::cout << "This is a curious case which should not happen. "
                         "How is that a ghost cell can give a face ownership?"
                      << std::endl;
            assert(all_owned_dofs[dof_i1].n_local_connected_DOFs != 0);
          }
        }
      }
    }
  }

  this->n_global_DOFs_rank_owns = n_dofs_this_rank_owns * n_face_bases;

  /*
   *
   */
  for (GenericDOF<dim> &dof : all_owned_dofs)
  {
    std::map<int, unsigned> local_dofs_num_map;
    std::map<int, unsigned> nonlocal_dofs_num_map;
    for (unsigned i_parent_cell = 0; i_parent_cell < dof.parent_cells.size();
         ++i_parent_cell)
    {
      auto parent_cell = dof.parent_cells[i_parent_cell];
      for (unsigned j_face = 0; j_face < n_faces_per_cell; ++j_face)
      {
        unsigned face_ij = dof.connected_face_of_parent_cell[i_parent_cell];
        if (j_face != face_ij)
          for (unsigned i_dof = 0;
               i_dof < (*parent_cell)->dofs_ID_in_all_ranks[j_face].size();
               ++i_dof)
          {
            if ((*parent_cell)->face_owner_rank[j_face] == this->comm_rank)
              local_dofs_num_map[(*parent_cell)
                                   ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
            else
              nonlocal_dofs_num_map[(*parent_cell)
                                      ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
          }
      }
    }

    for (unsigned i_parent_ghost = 0; i_parent_ghost < dof.parent_ghosts.size();
         ++i_parent_ghost)
    {
      auto parent_ghost = dof.parent_ghosts[i_parent_ghost];
      for (unsigned j_face = 0; j_face < n_faces_per_cell; ++j_face)
      {
        unsigned face_ij = dof.connected_face_of_parent_ghost[i_parent_ghost];
        if (j_face != face_ij)
          for (unsigned i_dof = 0;
               i_dof < (*parent_ghost)->dof_names_on_faces[j_face].count();
               ++i_dof)
          {
            if ((*parent_ghost)->face_owner_rank[j_face] == this->comm_rank)
              local_dofs_num_map[(*parent_ghost)
                                   ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
            else
              nonlocal_dofs_num_map[(*parent_ghost)
                                      ->dofs_ID_in_all_ranks[j_face][i_dof]]++;
          }
      }
    }
    dof.n_local_connected_DOFs += local_dofs_num_map.size();
    dof.n_nonlocal_connected_DOFs = nonlocal_dofs_num_map.size();
  }

  MPI_Allreduce(&this->n_global_DOFs_rank_owns,
                &this->n_global_DOFs_on_all_ranks,
                1,
                MPI_UNSIGNED,
                MPI_SUM,
                this->manager->comm);

  int dof_counter = 0;
  this->n_local_DOFs_connected_to_DOF.resize(this->n_global_DOFs_rank_owns);
  this->n_nonlocal_DOFs_connected_to_DOF.resize(this->n_global_DOFs_rank_owns);
  for (GenericDOF<dim> &dof : all_owned_dofs)
  {
    for (unsigned unknown = 0; unknown < n_face_bases; ++unknown)
    {
      this->n_local_DOFs_connected_to_DOF[dof_counter + unknown] +=
        dof.n_local_connected_DOFs * n_face_bases;
      this->n_nonlocal_DOFs_connected_to_DOF[dof_counter + unknown] +=
        dof.n_nonlocal_connected_DOFs * n_face_bases;
    }
    dof_counter += n_face_bases;
  }

  std::map<unsigned, unsigned> map_from_local_to_global;
  for (std::unique_ptr<GenericCell<dim> > &cell : all_owned_cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      for (unsigned i_dof = 0;
           i_dof < cell->dofs_ID_in_this_rank[i_face].size();
           ++i_dof)
      {
        int index1 = cell->dofs_ID_in_this_rank[i_face][i_dof];
        int index2 = cell->dofs_ID_in_all_ranks[i_face][i_dof];
        assert(index1 >= 0 && index2 >= 0);
        map_from_local_to_global[index1] = index2;
      }
    }
  }
  assert(map_from_local_to_global.size() == local_dof_num_on_this_rank);

  this->n_local_DOFs_on_this_rank = local_dof_num_on_this_rank * n_face_bases;
  this->scatter_from.reserve(this->n_local_DOFs_on_this_rank);
  this->scatter_to.reserve(this->n_local_DOFs_on_this_rank);
  for (const auto &map_it : map_from_local_to_global)
  {
    for (unsigned i_polyface = 0; i_polyface < n_face_bases; ++i_polyface)
    {
      this->scatter_to.push_back(map_it.first * n_face_bases + i_polyface);
      this->scatter_from.push_back(map_it.second * n_face_bases + i_polyface);
    }
  }

  /*
  for (const Face_Class<dim> &face : All_Faces)
  {
    dealii::Point<dim> p_center =
     face.Parent_Cells[0]
      ->dealii_Cell->face(face.connected_face_of_parent_cell[0])
      ->center();
    printf(" rank ID : %3d, el num. %3d, face center %10.3e , %10.3e :
  local "
           "%3d ; nonlocal %3d \n",
           comm_rank,
           face.Parent_Cells[0]->dealii_Cell->index(),
           p_center[0],
           p_center[1],
           face.n_local_connected_faces,
           face.n_nonlocal_connected_faces);
  }
  */

  char buffer[100];
  std::snprintf(buffer,
                100,
                "Number of DOFs in this rank is: %d and number of dofs "
                "in all "
                "ranks is : %d",
                this->n_global_DOFs_rank_owns,
                this->n_global_DOFs_on_all_ranks);
  this->manager->out_logger(this->manager->execution_time, buffer, true);
  //  std::cout << buffer << std::endl;
}

template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::assign_initial_data(
  const explicit_RKn<4, original_RK> &)
{
  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 2);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 2);
#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    const dealii::UpdateFlags &p1_flags =
      dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_inverse_jacobians | dealii::update_jacobians;
    const dealii::UpdateFlags &p2_flags = dealii::update_quadrature_points;
    const dealii::UpdateFlags &p3_flags = dealii::update_quadrature_points;
    const dealii::UpdateFlags &p4_flags = dealii::update_quadrature_points;
    FE_val_ptr p1(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                            DG_Elem,
                                            this->manager->elem_quad_bundle,
                                            p1_flags));
    FEFace_val_ptr p2(
      new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                    DG_Elem,
                                    this->manager->face_quad_bundle,
                                    p2_flags));
    FE_val_ptr p3(new dealii::FEValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_elem_support_points, p3_flags));
    FEFace_val_ptr p4(new dealii::FEFaceValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_face_support_points, p4_flags));
    for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
         i_cell = i_cell + this->manager->n_threads)
    {
      std::unique_ptr<GenericCell<dim> > cell(
        std::move(all_owned_cells[i_cell]));
      cell->attach_FEValues(p1, p2, p3, p4);
      cell->the_elem_basis = &(this->manager->the_elem_basis);
      cell->the_face_basis = &(this->manager->the_face_basis);
      cell->elem_quad_bundle = &(this->manager->elem_quad_bundle);
      cell->face_quad_bundle = &(this->manager->face_quad_bundle);
      static_cast<CellType<dim> *>(cell.get())->assign_initial_data();
      cell->detach_FEValues(p1, p2, p3, p4);
      all_owned_cells[i_cell] = std::move(cell);
    }
  }
}

template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::free_containers()
{
  wreck_it_Ralph(this->manager->cell_ID_to_num);

  wreck_it_Ralph(all_owned_cells);
  wreck_it_Ralph(this->n_local_DOFs_connected_to_DOF);
  wreck_it_Ralph(this->n_nonlocal_DOFs_connected_to_DOF);
  wreck_it_Ralph(this->scatter_from);
  wreck_it_Ralph(this->scatter_to);
}

template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::
  assemble_trace_of_conserved_vars(
    const explicit_hdg_model<dim, explicit_nswe> *const src_model)
{
  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 2);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 2);
#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    const dealii::UpdateFlags &p1_flags =
      dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_inverse_jacobians | dealii::update_jacobians;
    const dealii::UpdateFlags &p2_flags =
      dealii::update_values | dealii::update_JxW_values |
      dealii::update_quadrature_points | dealii::update_face_normal_vectors |
      dealii::update_inverse_jacobians;
    const dealii::UpdateFlags &p3_flags = dealii::update_quadrature_points;
    const dealii::UpdateFlags &p4_flags =
      dealii::update_quadrature_points | dealii::update_face_normal_vectors;

    FE_val_ptr p1(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                            DG_Elem,
                                            this->manager->elem_quad_bundle,
                                            p1_flags));
    FEFace_val_ptr p2(
      new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                    DG_Elem,
                                    this->manager->face_quad_bundle,
                                    p2_flags));
    FE_val_ptr p3(new dealii::FEValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_elem_support_points, p3_flags));
    FEFace_val_ptr p4(new dealii::FEFaceValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_face_support_points, p4_flags));
    for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
         i_cell = i_cell + this->manager->n_threads)
    {
      std::unique_ptr<GenericCell<dim> > cell(
        std::move(all_owned_cells[i_cell]));
      cell->attach_FEValues(p1, p2, p3, p4);
      cell->the_elem_basis = &(this->manager->the_elem_basis);
      cell->the_face_basis = &(this->manager->the_face_basis);
      cell->elem_quad_bundle = &(this->manager->elem_quad_bundle);
      cell->face_quad_bundle = &(this->manager->face_quad_bundle);
      static_cast<CellType<dim> *>(cell.get())
        ->produce_trace_of_conserved_vars(static_cast<explicit_nswe<dim> *>(
          src_model->all_owned_cells[i_cell].get()));
      cell->detach_FEValues(p1, p2, p3, p4);
      all_owned_cells[i_cell] = std::move(cell);
    }
  }
}

template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::compute_and_sum_grad_prim_vars(
  const explicit_hdg_model<dim, explicit_nswe> *const src_model,
  const double *const local_conserved_vars_sums,
  const double *const local_face_count)
{
  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 2);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 2);
#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    const dealii::UpdateFlags &p1_flags =
      dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_inverse_jacobians | dealii::update_jacobians;
    const dealii::UpdateFlags &p2_flags =
      dealii::update_values | dealii::update_JxW_values |
      dealii::update_quadrature_points | dealii::update_face_normal_vectors |
      dealii::update_inverse_jacobians;
    const dealii::UpdateFlags &p3_flags = dealii::update_quadrature_points;
    const dealii::UpdateFlags &p4_flags =
      dealii::update_quadrature_points | dealii::update_face_normal_vectors;

    FE_val_ptr p1(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                            DG_Elem,
                                            this->manager->elem_quad_bundle,
                                            p1_flags));
    FEFace_val_ptr p2(
      new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                    DG_Elem,
                                    this->manager->face_quad_bundle,
                                    p2_flags));
    FE_val_ptr p3(new dealii::FEValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_elem_support_points, p3_flags));
    FEFace_val_ptr p4(new dealii::FEFaceValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_face_support_points, p4_flags));
    for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
         i_cell = i_cell + this->manager->n_threads)
    {
      std::unique_ptr<GenericCell<dim> > cell(
        std::move(all_owned_cells[i_cell]));
      cell->attach_FEValues(p1, p2, p3, p4);
      cell->the_elem_basis = &(this->manager->the_elem_basis);
      cell->the_face_basis = &(this->manager->the_face_basis);
      cell->elem_quad_bundle = &(this->manager->elem_quad_bundle);
      cell->face_quad_bundle = &(this->manager->face_quad_bundle);
      static_cast<CellType<dim> *>(cell.get())
        ->compute_avg_prim_vars_flux(
          static_cast<explicit_nswe<dim> *>(
            src_model->all_owned_cells[i_cell].get()),
          local_conserved_vars_sums,
          local_face_count);
      static_cast<CellType<dim> *>(cell.get())->compute_prim_vars_derivatives();
      static_cast<CellType<dim> *>(cell.get())
        ->produce_trace_of_grad_prim_vars(static_cast<explicit_nswe<dim> *>(
          src_model->all_owned_cells[i_cell].get()));
      cell->detach_FEValues(p1, p2, p3, p4);
      all_owned_cells[i_cell] = std::move(cell);
    }
  }
}

template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::assemble_globals(
  const explicit_hdg_model<dim, explicit_nswe> *const src_model,
  const double *const local_V_x_sums,
  const double *const local_V_y_sums,
  const solver_update_keys &keys)
{
  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 2);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 2);
#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    const dealii::UpdateFlags &p1_flags =
      dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_inverse_jacobians | dealii::update_jacobians;
    const dealii::UpdateFlags &p2_flags =
      dealii::update_values | dealii::update_JxW_values |
      dealii::update_quadrature_points | dealii::update_face_normal_vectors |
      dealii::update_inverse_jacobians;
    const dealii::UpdateFlags &p3_flags = dealii::update_quadrature_points;
    const dealii::UpdateFlags &p4_flags =
      dealii::update_quadrature_points | dealii::update_face_normal_vectors;

    FE_val_ptr p1(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                            DG_Elem,
                                            this->manager->elem_quad_bundle,
                                            p1_flags));
    FEFace_val_ptr p2(
      new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                    DG_Elem,
                                    this->manager->face_quad_bundle,
                                    p2_flags));
    FE_val_ptr p3(new dealii::FEValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_elem_support_points, p3_flags));
    FEFace_val_ptr p4(new dealii::FEFaceValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_face_support_points, p4_flags));
    for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
         i_cell = i_cell + this->manager->n_threads)
    {
      std::unique_ptr<GenericCell<dim> > cell(
        std::move(all_owned_cells[i_cell]));
      cell->attach_FEValues(p1, p2, p3, p4);
      cell->the_elem_basis = &(this->manager->the_elem_basis);
      cell->the_face_basis = &(this->manager->the_face_basis);
      cell->elem_quad_bundle = &(this->manager->elem_quad_bundle);
      cell->face_quad_bundle = &(this->manager->face_quad_bundle);
      static_cast<CellType<dim> *>(cell.get())
        ->compute_avg_grad_V_flux(static_cast<explicit_nswe<dim> *>(
                                    src_model->all_owned_cells[i_cell].get()),
                                  local_V_x_sums,
                                  local_V_y_sums);
      static_cast<CellType<dim> *>(cell.get())->assemble_globals(keys);
      cell->detach_FEValues(p1, p2, p3, p4);
      all_owned_cells[i_cell] = std::move(cell);
    }
  }
}

template <int dim, template <int> class CellType>
bool hdg_model_with_explicit_rk<dim, CellType>::check_for_next_iter(
  double *const local_uhat)
{
  bool explicit_iteration_required = true;
  double this_iter_increment = 0.;
  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 2);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 2);

#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    const dealii::UpdateFlags &p1_flags = dealii::update_default;
    const dealii::UpdateFlags &p2_flags = dealii::update_JxW_values |
                                          dealii::update_quadrature_points |
                                          dealii::update_face_normal_vectors;
    const dealii::UpdateFlags &p3_flags = dealii::update_default;
    const dealii::UpdateFlags &p4_flags = dealii::update_quadrature_points;

    FE_val_ptr p1(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                            DG_Elem,
                                            this->manager->elem_quad_bundle,
                                            p1_flags));
    FEFace_val_ptr p2(
      new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                    DG_Elem,
                                    this->manager->face_quad_bundle,
                                    p2_flags));
    FE_val_ptr p3(new dealii::FEValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_elem_support_points, p3_flags));
    FEFace_val_ptr p4(new dealii::FEFaceValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_face_support_points, p4_flags));

    for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
         i_cell = i_cell + this->manager->n_threads)
    {
      std::unique_ptr<GenericCell<dim> > cell(
        std::move(all_owned_cells[i_cell]));
      cell->attach_FEValues(p1, p2, p3, p4);
      /* After these pointer assignments, we call the followings */
      this_iter_increment += static_cast<CellType<dim> *>(cell.get())
                               ->get_iteration_increment_norm(local_uhat);
      /* Finally, we detach fevalues and give back cell contents back */
      cell->detach_FEValues(p1, p2, p3, p4);
      all_owned_cells[i_cell] = std::move(cell);
    }
  }

  double global_iter_increment;
  MPI_Allreduce(&this_iter_increment,
                &global_iter_increment,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                this->manager->comm);

  if (sqrt(global_iter_increment) < 1E-12)
  {
    if (this->comm_rank == 0)
      std::cout << sqrt(global_iter_increment) << std::endl;

    explicit_iteration_required = false;
#ifdef _OPENMP
#pragma omp parallel
    {
      unsigned thread_id = omp_get_thread_num();
#else
    unsigned thread_id = 0;
    {
#endif
      const dealii::UpdateFlags &p1_flags =
        dealii::update_JxW_values | dealii::update_quadrature_points |
        dealii::update_inverse_jacobians | dealii::update_jacobians;
      const dealii::UpdateFlags &p2_flags =
        dealii::update_JxW_values | dealii::update_quadrature_points |
        dealii::update_face_normal_vectors | dealii::update_inverse_jacobians;
      const dealii::UpdateFlags &p3_flags = dealii::update_quadrature_points;
      const dealii::UpdateFlags &p4_flags = dealii::update_quadrature_points;

      FE_val_ptr p1(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                              DG_Elem,
                                              this->manager->elem_quad_bundle,
                                              p1_flags));
      FEFace_val_ptr p2(
        new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                      DG_Elem,
                                      this->manager->face_quad_bundle,
                                      p2_flags));
      FE_val_ptr p3(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                              DG_Elem,
                                              LGL_elem_support_points,
                                              p3_flags));
      FEFace_val_ptr p4(
        new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                      DG_Elem,
                                      LGL_face_support_points,
                                      p4_flags));

      for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
           i_cell = i_cell + this->manager->n_threads)
      {
        std::unique_ptr<GenericCell<dim> > cell(
          std::move(all_owned_cells[i_cell]));
        cell->attach_FEValues(p1, p2, p3, p4);
        static_cast<CellType<dim> *>(cell.get())
          ->ready_for_next_stage(local_uhat);
        cell->detach_FEValues(p1, p2, p3, p4);
        all_owned_cells[i_cell] = std::move(cell);
      }
    }
  }
  return explicit_iteration_required;
}

template <int dim, template <int> class CellType>
bool hdg_model_with_explicit_rk<dim, CellType>::compute_internal_dofs(
  double *const local_uhat)
{
  bool next_iter_required = false;
  double this_iter_increment = 0.;
  typedef typename GenericCell<dim>::elem_basis_type elem_basis_type;
  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 2);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 2);

  /* active indices contain both ghost and locally owned indices. */
  local_nodal_sol<dim> refn_local_nodal(&this->DoF_H_Refine,
                                        &(this->manager->comm));
  local_nodal_sol<dim> cell_local_nodal(&this->DoF_H_System,
                                        &(this->manager->comm));

  double u_error = 0;
  double q_error = 0;
  double div_q_error = 0;

  std::vector<dealii::Point<1, double> > output_points;
  if (poly_order == 0)
    output_points.push_back(dealii::Point<1, double>(0.5));
  else
    output_points = this->manager->LGL_quad_1D.get_points();
  poly_space_basis<elem_basis_type, dim> the_elem_equal_dist_basis(
    DG_Elem.get_unit_support_points(), output_points, Domain::From_0_to_1);

#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    const dealii::UpdateFlags &p1_flags =
      dealii::update_values | dealii::update_gradients |
      dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_inverse_jacobians | dealii::update_jacobians;
    const dealii::UpdateFlags &p2_flags =
      dealii::update_values | dealii::update_gradients |
      dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_face_normal_vectors | dealii::update_inverse_jacobians;
    const dealii::UpdateFlags &p3_flags = dealii::update_quadrature_points;
    const dealii::UpdateFlags &p4_flags = dealii::update_quadrature_points;

    FE_val_ptr p1(new dealii::FEValues<dim>(this->manager->elem_mapping,
                                            DG_Elem,
                                            this->manager->elem_quad_bundle,
                                            p1_flags));
    FEFace_val_ptr p2(
      new dealii::FEFaceValues<dim>(this->manager->elem_mapping,
                                    DG_Elem,
                                    this->manager->face_quad_bundle,
                                    p2_flags));
    FE_val_ptr p3(new dealii::FEValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_elem_support_points, p3_flags));
    FEFace_val_ptr p4(new dealii::FEFaceValues<dim>(
      this->manager->elem_mapping, DG_Elem, LGL_face_support_points, p4_flags));

    for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
         i_cell = i_cell + this->manager->n_threads)
    {
      std::unique_ptr<GenericCell<dim> > cell(
        std::move(all_owned_cells[i_cell]));
      cell->attach_FEValues(p1, p2, p3, p4);
      /* These two lines are important. I know I am idiot! */
      cell->refn_local_nodal = &refn_local_nodal;
      cell->cell_local_nodal = &cell_local_nodal;
      /* After these pointer assignments, we call the followings */
      eigen3mat u_vec, q_vec;
      this_iter_increment +=
        static_cast<CellType<dim> *>(cell.get())
          ->compute_internal_dofs(
            local_uhat, u_vec, q_vec, std::move(the_elem_equal_dist_basis));
      cell->internal_vars_errors(u_vec, q_vec, u_error, q_error);
      /* Finally, we detach fevalues and give back cell contents back */
      cell->detach_FEValues(p1, p2, p3, p4);
      all_owned_cells[i_cell] = std::move(cell);
    }
  }
  refn_local_nodal.copy_to_global_vec(this->manager->refine_solu);
  cell_local_nodal.copy_to_global_vec(this->manager->visual_solu);

  double global_u_error, global_q_error, global_iter_increment;
  MPI_Reduce(
    &u_error, &global_u_error, 1, MPI_DOUBLE, MPI_SUM, 0, this->manager->comm);
  MPI_Reduce(
    &q_error, &global_q_error, 1, MPI_DOUBLE, MPI_SUM, 0, this->manager->comm);
  MPI_Allreduce(&this_iter_increment,
                &global_iter_increment,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                this->manager->comm);

  if (this->comm_rank == 0)
    std::cout << sqrt(global_iter_increment) << std::endl;
  if (sqrt(global_iter_increment) > 1E-11)
  {
    next_iter_required = true;
#ifdef _OPENMP
#pragma omp parallel
    {
      unsigned thread_id = omp_get_thread_num();
#else
    unsigned thread_id = 0;
    {
#endif

      for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
           i_cell = i_cell + this->manager->n_threads)
      {
        std::unique_ptr<GenericCell<dim> > cell(
          std::move(all_owned_cells[i_cell]));
        static_cast<CellType<dim> *>(cell.get())->ready_for_next_iteration();
        all_owned_cells[i_cell] = std::move(cell);
      }
    }
  }
  else
  {
    if (this->comm_rank == 0)
    {
      char buffer[200];
      std::snprintf(buffer,
                    200,
                    " NEl : %10d, || uh - u ||_L2 : %12.4e; || q - qh ||_L2 "
                    "is: "
                    "%12.4e; || "
                    "div(q - qh) ||_L2 : %12.4e; || ",
                    this->manager->the_grid.n_global_active_cells(),
                    sqrt(global_u_error),
                    sqrt(global_q_error),
                    sqrt(div_q_error));
      this->manager->convergence_result << buffer << std::endl;
    }
#ifdef _OPENMP
#pragma omp parallel
    {
      unsigned thread_id = omp_get_thread_num();
#else
    unsigned thread_id = 0;
    {
#endif

      for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
           i_cell = i_cell + this->manager->n_threads)
      {
        std::unique_ptr<GenericCell<dim> > cell(
          std::move(all_owned_cells[i_cell]));
        static_cast<CellType<dim> *>(cell.get())->ready_for_next_time_step();
        all_owned_cells[i_cell] = std::move(cell);
      }
    }
  }
  return next_iter_required;
}

template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::init_solver(
  const explicit_hdg_model<dim, explicit_nswe> *const src_model)
{
  solver_options options_ = CellType<dim>::required_solver_options();
  solver_type type_ = CellType<dim>::required_solver_type();

  solver = std::move(generic_solver<dim, CellType>::make_solver(
    type_, &(this->manager->comm), this, options_));

  flux_gen = std::move(GN_dispersive_flux_generator<dim>::make_flux_generator(
    &(this->manager->comm), src_model));
}

template <int dim, template <int> class CellType>
void hdg_model_with_explicit_rk<dim, CellType>::reinit_solver(
  const solver_update_keys &update_keys_)
{
  solver_options options_ = CellType<dim>::required_solver_options();
  solver->reinit_components(this, options_, update_keys_);
  flux_gen->free_components();
  flux_gen->init_components();
}

template <int dim, template <int> class CellType>
template <template <int> class srcCellType,
          template <int, template <int> class> class srcModelType>
void hdg_model_with_explicit_rk<dim, CellType>::get_results_from_another_model(
  srcModelType<dim, srcCellType> &src_model)
{
#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    for (unsigned i_cell = thread_id; i_cell < all_owned_cells.size();
         i_cell = i_cell + this->manager->n_threads)
    {
      std::unique_ptr<GenericCell<dim> > src_cell(
        std::move(src_model.all_owned_cells[i_cell]));
      std::unique_ptr<GenericCell<dim> > cell(
        std::move(all_owned_cells[i_cell]));
      static_cast<CellType<dim> *>(cell.get())
        ->set_previous_step_results(
          static_cast<srcCellType<dim> *>(src_cell.get()));
      all_owned_cells[i_cell] = std::move(cell);
      src_model.all_owned_cells[i_cell] = std::move(src_cell);
    }
  }
}
