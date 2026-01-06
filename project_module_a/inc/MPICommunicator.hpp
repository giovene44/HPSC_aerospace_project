#ifndef MPI_COMMUNICATOR_HPP
#define MPI_COMMUNICATOR_HPP

#include "Variables.hpp"
#include <vector>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @brief Wrapper for MPI communication operations needed by Schur complement solver.
 *
 * Provides an abstraction layer that can fall back to serial execution when
 * MPI is not available (USE_MPI not defined).
 *
 * Key operations:
 * - Point-to-point communication for interface value exchange
 * - Collective operations for Schur system assembly and solution broadcast
 * - Cartesian topology for 3D domain decomposition
 */
class MPICommunicator {
public:
    MPICommunicator();
    ~MPICommunicator();

    // Prevent copying (MPI handles are not copyable)
    MPICommunicator(const MPICommunicator&) = delete;
    MPICommunicator& operator=(const MPICommunicator&) = delete;

    /**
     * @brief Initialize MPI (call at program start).
     * @param argc Pointer to argc from main
     * @param argv Pointer to argv from main
     */
    void init(int* argc, char*** argv);

    /**
     * @brief Finalize MPI (call at program end).
     */
    void finalize();

    /**
     * @brief Check if MPI is initialized.
     */
    bool is_initialized() const { return initialized_; }

    // ==================== Topology ====================

    /**
     * @brief Get this process's rank in the communicator.
     */
    int get_rank() const { return rank_; }

    /**
     * @brief Get total number of processes.
     */
    int get_size() const { return size_; }

    /**
     * @brief Create a 3D Cartesian topology.
     * @param px Number of processes in X direction
     * @param py Number of processes in Y direction
     * @param pz Number of processes in Z direction
     */
    void create_cart_topology(int px, int py, int pz);

    /**
     * @brief Get Cartesian coordinates of a rank.
     * @param rank Process rank
     * @param cx Output: X coordinate
     * @param cy Output: Y coordinate
     * @param cz Output: Z coordinate
     */
    void get_cart_coords(int rank, int& cx, int& cy, int& cz) const;

    /**
     * @brief Get rank from Cartesian coordinates.
     * @param cx X coordinate
     * @param cy Y coordinate
     * @param cz Z coordinate
     * @return Process rank at (cx, cy, cz)
     */
    int get_cart_rank(int cx, int cy, int cz) const;

    /**
     * @brief Get left neighbor in a direction.
     * @param direction 0=X, 1=Y, 2=Z
     * @return Rank of left neighbor, or -1 if at boundary
     */
    int get_left_neighbor(int direction) const;

    /**
     * @brief Get right neighbor in a direction.
     * @param direction 0=X, 1=Y, 2=Z
     * @return Rank of right neighbor, or -1 if at boundary
     */
    int get_right_neighbor(int direction) const;

    // ==================== Point-to-Point ====================

    /**
     * @brief Send a single Real value to another process (blocking).
     * @param value Value to send
     * @param dest_rank Destination process rank
     * @param tag Message tag
     */
    void send_value(Real value, int dest_rank, int tag);

    /**
     * @brief Receive a single Real value from another process (blocking).
     * @param src_rank Source process rank
     * @param tag Message tag
     * @return Received value
     */
    Real recv_value(int src_rank, int tag);

    /**
     * @brief Send a vector of Real values (blocking).
     * @param values Vector to send
     * @param dest_rank Destination process rank
     * @param tag Message tag
     */
    void send_vector(const std::vector<Real>& values, int dest_rank, int tag);

    /**
     * @brief Receive a vector of Real values (blocking).
     * @param values Output vector (must be pre-sized)
     * @param src_rank Source process rank
     * @param tag Message tag
     */
    void recv_vector(std::vector<Real>& values, int src_rank, int tag);

    // ==================== Non-blocking ====================

    /**
     * @brief Non-blocking send of a Real value.
     * @param value Value to send
     * @param dest_rank Destination process rank
     * @param tag Message tag
     * @return Request ID for waiting
     */
    int isend_value(Real value, int dest_rank, int tag);

    /**
     * @brief Non-blocking receive of a Real value.
     * @param value Reference to store received value
     * @param src_rank Source process rank
     * @param tag Message tag
     * @return Request ID for waiting
     */
    int irecv_value(Real& value, int src_rank, int tag);

    /**
     * @brief Wait for a single non-blocking operation to complete.
     * @param request_id Request ID from isend/irecv
     */
    void wait(int request_id);

    /**
     * @brief Wait for all pending non-blocking operations.
     */
    void waitall();

    // ==================== Collective Operations ====================

    /**
     * @brief Gather Schur complement contributions from all processes.
     *
     * Each process contributes 2 values (left and right diagonal entries).
     * Result is gathered on all processes (Allgather).
     *
     * @param local_contrib Local contributions (size 2)
     * @param global_contrib Output: all contributions (size 2*num_procs)
     */
    void allgather_schur_diag(const std::vector<Real>& local_contrib,
                              std::vector<Real>& global_contrib);

    /**
     * @brief Reduce interface RHS contributions (sum).
     *
     * Each process contributes to the interface RHS. Since interfaces are
     * shared between neighbors, we need to sum contributions.
     *
     * @param local_rhs Local RHS contribution (size = num_interfaces)
     * @param global_rhs Output: summed RHS (size = num_interfaces)
     * @param root Root process (default 0)
     */
    void reduce_interface_rhs(const std::vector<Real>& local_rhs,
                              std::vector<Real>& global_rhs,
                              int root = 0);

    /**
     * @brief Broadcast interface solution from root to all processes.
     * @param interface_values Interface solution (size = num_interfaces)
     * @param root Root process (default 0)
     */
    void broadcast_interface_solution(std::vector<Real>& interface_values,
                                      int root = 0);

    /**
     * @brief Global barrier - synchronize all processes.
     */
    void barrier();

    /**
     * @brief Exchange interface values with neighbors in a direction.
     *
     * Sends right_value to right neighbor, receives from left neighbor.
     * Sends left_value to left neighbor, receives from right neighbor.
     *
     * @param direction 0=X, 1=Y, 2=Z
     * @param left_value Value at left interface (to send to left neighbor)
     * @param right_value Value at right interface (to send to right neighbor)
     * @param recv_from_left Output: value received from left neighbor
     * @param recv_from_right Output: value received from right neighbor
     */
    void exchange_interface_values(int direction,
                                   Real left_value, Real right_value,
                                   Real& recv_from_left, Real& recv_from_right);

private:
    int rank_;
    int size_;
    bool initialized_;

    // Cartesian topology
    int cart_dims_[3];
    int cart_coords_[3];
    bool has_cart_topology_;

#ifdef USE_MPI
    MPI_Comm world_comm_;
    MPI_Comm cart_comm_;
    std::vector<MPI_Request> pending_requests_;
    std::vector<Real> send_buffers_;  // Buffers for non-blocking sends
#endif
};

// ==================== Implementation ====================

inline MPICommunicator::MPICommunicator()
    : rank_(0), size_(1), initialized_(false), has_cart_topology_(false) {
    cart_dims_[0] = cart_dims_[1] = cart_dims_[2] = 1;
    cart_coords_[0] = cart_coords_[1] = cart_coords_[2] = 0;
#ifdef USE_MPI
    world_comm_ = MPI_COMM_NULL;
    cart_comm_ = MPI_COMM_NULL;
#endif
}

inline MPICommunicator::~MPICommunicator() {
    // Don't finalize here - let user call finalize() explicitly
}

inline void MPICommunicator::init(int* argc, char*** argv) {
#ifdef USE_MPI
    int already_initialized;
    MPI_Initialized(&already_initialized);
    if (!already_initialized) {
        MPI_Init(argc, argv);
    }
    world_comm_ = MPI_COMM_WORLD;
    MPI_Comm_rank(world_comm_, &rank_);
    MPI_Comm_size(world_comm_, &size_);
    initialized_ = true;
#else
    (void)argc;
    (void)argv;
    rank_ = 0;
    size_ = 1;
    initialized_ = true;
#endif
}

inline void MPICommunicator::finalize() {
#ifdef USE_MPI
    if (initialized_) {
        if (cart_comm_ != MPI_COMM_NULL && cart_comm_ != MPI_COMM_WORLD) {
            MPI_Comm_free(&cart_comm_);
        }
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
#endif
    initialized_ = false;
}

inline void MPICommunicator::create_cart_topology(int px, int py, int pz) {
#ifdef USE_MPI
    if (!initialized_) {
        throw std::runtime_error("MPI not initialized");
    }
    if (px * py * pz != size_) {
        throw std::runtime_error("Cart topology size mismatch: px*py*pz != size");
    }

    int dims[3] = {px, py, pz};
    int periods[3] = {0, 0, 0};  // Non-periodic boundaries
    int reorder = 1;

    if (cart_comm_ != MPI_COMM_NULL && cart_comm_ != MPI_COMM_WORLD) {
        MPI_Comm_free(&cart_comm_);
    }

    MPI_Cart_create(world_comm_, 3, dims, periods, reorder, &cart_comm_);
    MPI_Comm_rank(cart_comm_, &rank_);
    MPI_Cart_coords(cart_comm_, rank_, 3, cart_coords_);

    cart_dims_[0] = px;
    cart_dims_[1] = py;
    cart_dims_[2] = pz;
    has_cart_topology_ = true;
#else
    if (px * py * pz != 1) {
        throw std::runtime_error("Serial mode: cart topology must be 1x1x1");
    }
    cart_dims_[0] = cart_dims_[1] = cart_dims_[2] = 1;
    cart_coords_[0] = cart_coords_[1] = cart_coords_[2] = 0;
    has_cart_topology_ = true;
#endif
}

inline void MPICommunicator::get_cart_coords(int rank, int& cx, int& cy, int& cz) const {
#ifdef USE_MPI
    if (has_cart_topology_) {
        int coords[3];
        MPI_Cart_coords(cart_comm_, rank, 3, coords);
        cx = coords[0];
        cy = coords[1];
        cz = coords[2];
    } else {
        cx = cy = cz = 0;
    }
#else
    (void)rank;
    cx = cy = cz = 0;
#endif
}

inline int MPICommunicator::get_cart_rank(int cx, int cy, int cz) const {
#ifdef USE_MPI
    if (has_cart_topology_) {
        int coords[3] = {cx, cy, cz};
        int rank;
        MPI_Cart_rank(cart_comm_, coords, &rank);
        return rank;
    }
    return 0;
#else
    (void)cx; (void)cy; (void)cz;
    return 0;
#endif
}

inline int MPICommunicator::get_left_neighbor(int direction) const {
#ifdef USE_MPI
    if (has_cart_topology_) {
        int left, right;
        MPI_Cart_shift(cart_comm_, direction, 1, &left, &right);
        return (left == MPI_PROC_NULL) ? -1 : left;
    }
    return -1;
#else
    (void)direction;
    return -1;
#endif
}

inline int MPICommunicator::get_right_neighbor(int direction) const {
#ifdef USE_MPI
    if (has_cart_topology_) {
        int left, right;
        MPI_Cart_shift(cart_comm_, direction, 1, &left, &right);
        return (right == MPI_PROC_NULL) ? -1 : right;
    }
    return -1;
#else
    (void)direction;
    return -1;
#endif
}

inline void MPICommunicator::send_value(Real value, int dest_rank, int tag) {
#ifdef USE_MPI
    MPI_Send(&value, 1, MPI_FLOAT, dest_rank, tag, cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_);
#else
    (void)value; (void)dest_rank; (void)tag;
#endif
}

inline Real MPICommunicator::recv_value(int src_rank, int tag) {
#ifdef USE_MPI
    Real value;
    MPI_Recv(&value, 1, MPI_FLOAT, src_rank, tag, cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, MPI_STATUS_IGNORE);
    return value;
#else
    (void)src_rank; (void)tag;
    return 0.0;
#endif
}

inline void MPICommunicator::send_vector(const std::vector<Real>& values, int dest_rank, int tag) {
#ifdef USE_MPI
    MPI_Send(values.data(), static_cast<int>(values.size()), MPI_FLOAT, dest_rank, tag,
             cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_);
#else
    (void)values; (void)dest_rank; (void)tag;
#endif
}

inline void MPICommunicator::recv_vector(std::vector<Real>& values, int src_rank, int tag) {
#ifdef USE_MPI
    MPI_Recv(values.data(), static_cast<int>(values.size()), MPI_FLOAT, src_rank, tag,
             cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, MPI_STATUS_IGNORE);
#else
    (void)values; (void)src_rank; (void)tag;
#endif
}

inline int MPICommunicator::isend_value(Real value, int dest_rank, int tag) {
#ifdef USE_MPI
    send_buffers_.push_back(value);
    MPI_Request request;
    MPI_Isend(&send_buffers_.back(), 1, MPI_FLOAT, dest_rank, tag,
              cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, &request);
    pending_requests_.push_back(request);
    return static_cast<int>(pending_requests_.size()) - 1;
#else
    (void)value; (void)dest_rank; (void)tag;
    return 0;
#endif
}

inline int MPICommunicator::irecv_value(Real& value, int src_rank, int tag) {
#ifdef USE_MPI
    MPI_Request request;
    MPI_Irecv(&value, 1, MPI_FLOAT, src_rank, tag,
              cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, &request);
    pending_requests_.push_back(request);
    return static_cast<int>(pending_requests_.size()) - 1;
#else
    (void)value; (void)src_rank; (void)tag;
    return 0;
#endif
}

inline void MPICommunicator::wait(int request_id) {
#ifdef USE_MPI
    if (request_id >= 0 && request_id < static_cast<int>(pending_requests_.size())) {
        MPI_Wait(&pending_requests_[request_id], MPI_STATUS_IGNORE);
    }
#else
    (void)request_id;
#endif
}

inline void MPICommunicator::waitall() {
#ifdef USE_MPI
    if (!pending_requests_.empty()) {
        MPI_Waitall(static_cast<int>(pending_requests_.size()), pending_requests_.data(), MPI_STATUSES_IGNORE);
        pending_requests_.clear();
        send_buffers_.clear();
    }
#endif
}

inline void MPICommunicator::allgather_schur_diag(const std::vector<Real>& local_contrib,
                                                   std::vector<Real>& global_contrib) {
#ifdef USE_MPI
    global_contrib.resize(size_ * local_contrib.size());
    MPI_Allgather(local_contrib.data(), static_cast<int>(local_contrib.size()), MPI_FLOAT,
                  global_contrib.data(), static_cast<int>(local_contrib.size()), MPI_FLOAT,
                  cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_);
#else
    global_contrib = local_contrib;
#endif
}

inline void MPICommunicator::reduce_interface_rhs(const std::vector<Real>& local_rhs,
                                                   std::vector<Real>& global_rhs,
                                                   int root) {
#ifdef USE_MPI
    global_rhs.resize(local_rhs.size());
    MPI_Reduce(local_rhs.data(), global_rhs.data(), static_cast<int>(local_rhs.size()),
               MPI_FLOAT, MPI_SUM, root, cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_);
#else
    (void)root;
    global_rhs = local_rhs;
#endif
}

inline void MPICommunicator::broadcast_interface_solution(std::vector<Real>& interface_values,
                                                           int root) {
#ifdef USE_MPI
    MPI_Bcast(interface_values.data(), static_cast<int>(interface_values.size()), MPI_FLOAT,
              root, cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_);
#else
    (void)interface_values; (void)root;
#endif
}

inline void MPICommunicator::barrier() {
#ifdef USE_MPI
    MPI_Barrier(cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_);
#endif
}

inline void MPICommunicator::exchange_interface_values(int direction,
                                                        Real left_value, Real right_value,
                                                        Real& recv_from_left, Real& recv_from_right) {
#ifdef USE_MPI
    int left_neighbor = get_left_neighbor(direction);
    int right_neighbor = get_right_neighbor(direction);

    MPI_Request requests[4];
    int num_requests = 0;

    // Send to left, receive from right
    if (left_neighbor >= 0) {
        MPI_Isend(&left_value, 1, MPI_FLOAT, left_neighbor, 0,
                  cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, &requests[num_requests++]);
    }
    if (right_neighbor >= 0) {
        MPI_Irecv(&recv_from_right, 1, MPI_FLOAT, right_neighbor, 0,
                  cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, &requests[num_requests++]);
    }

    // Send to right, receive from left
    if (right_neighbor >= 0) {
        MPI_Isend(&right_value, 1, MPI_FLOAT, right_neighbor, 1,
                  cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, &requests[num_requests++]);
    }
    if (left_neighbor >= 0) {
        MPI_Irecv(&recv_from_left, 1, MPI_FLOAT, left_neighbor, 1,
                  cart_comm_ != MPI_COMM_NULL ? cart_comm_ : world_comm_, &requests[num_requests++]);
    }

    if (num_requests > 0) {
        MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
    }

    // Set boundary values to 0 if no neighbor
    if (left_neighbor < 0) recv_from_left = 0.0;
    if (right_neighbor < 0) recv_from_right = 0.0;
#else
    (void)direction; (void)left_value; (void)right_value;
    recv_from_left = 0.0;
    recv_from_right = 0.0;
#endif
}

#endif // MPI_COMMUNICATOR_HPP
