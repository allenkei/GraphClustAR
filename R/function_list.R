## usethis namespace: start
#' @useDynLib GraphClustAR, .registration = TRUE
## usethis namespace: end
NULL

#' @importFrom Rcpp sourceCpp
NULL



#' Clustering nodes.
#' @description This function clusters nodes.
#' @param TS_by_node Numeric matrix of size N x n. Each row is the time series for a node.
#' @param adj_w Adjacency matrix with non-negative weights
#' @param ADMM_iter ADMM iteration
#' @param lambda GFL penalty
#' @param gamma penalty for the augmentation term
#' @param update_gamma If TRUE, gamma is updated with a schedule
#' @param verbose If TRUE, print info during parameter learning
#'
#' @return A list
#' @export
#'
#' @examples
#' ts_data <- gen_ar1_by_cluster(cluster_sizes = c(50, 50, 50), n = 100)
#' graph_data <- gen_weighted_block_graph(cluster_sizes = c(50, 50, 50))
#' TS_by_node <- ts_data$TS_by_node
#' adj_w <- graph_data$weights
#' labels <- ts_data$labels
#' result <- ClustARp(TS_by_node, adj_w, lag_p=1, ADMM_iter=200, lambda=1, gamma=1,
#'                    update_gamma=FALSE, verbose=TRUE)
#' plot(result$phi, col=labels)
#'
ClustARp <- function(TS_by_node, adj_w, lag_p, ADMM_iter, lambda, gamma, update_gamma, verbose){

  ts_length <- ncol(TS_by_node)
  ar_X_Y <- get_ar_X_Y(TS_by_node, lag_p)
  graph_info <- get_graph_info(adj_w)

  output <- GraphClustARp_cpp(ar_X_Y$X_list, ar_X_Y$Y_list, graph_info$edge_list, graph_info$node_degree,
                             lag_p, ts_length, ADMM_iter, lambda, gamma, update_gamma, verbose)

  return(output)
}



#' Generate nodal AR(1) time series
#' @export
#'
#' @example
#' ts_data <- gen_ar1_by_cluster(cluster_sizes = c(50, 50, 50), n = 100)
gen_ar1_by_cluster <- function(
    cluster_sizes = c(50, 50, 50),
    n = 100,
    mu_vec = c(0, 2, -2),
    phi_vec = c(0.2, 0.6, -0.4),
    sigma_vec = c(0.5, 0.5, 0.5),
    stationary_init = TRUE,
    seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)

  K <- length(cluster_sizes)
  N <- sum(cluster_sizes)

  labels <- rep(seq_len(K), times = cluster_sizes)

  TS_by_node <- matrix(0, nrow = N, ncol = n)

  for (i in seq_len(N)) {
    k <- labels[i]

    mu <- mu_vec[k]
    phi <- phi_vec[k]
    sigma <- sigma_vec[k]

    if (stationary_init && abs(phi) < 1) {
      TS_by_node[i, 1] <- rnorm(1, mean = mu, sd = sigma / sqrt(1 - phi^2))
    } else {
      TS_by_node[i, 1] <- rnorm(1, mean = mu, sd = sigma)
    }

    for (t in 2:n) {
      eps <- rnorm(1, mean = 0, sd = sigma)
      TS_by_node[i, t] <- mu + phi * (TS_by_node[i, t - 1] - mu) + eps
    }
  }

  list(
    TS_by_node = TS_by_node,
    labels = labels,
    mu = mu_vec[labels],
    phi = phi_vec[labels],
    sigma = sigma_vec[labels]
  )
}



#' Generate weighted block graph with 3 or more clusters
#' @export
#'
#' @example
#' graph_data <- gen_weighted_block_graph(cluster_sizes = c(50, 50, 50))
gen_weighted_block_graph <- function(
    cluster_sizes = c(50, 50, 50),
    p_in = 0.30,
    p_out = 0.05,
    weight_in_range = c(1.0, 1.5),
    weight_out_range = c(0.1, 0.5),
    seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)

  K <- length(cluster_sizes)
  N <- sum(cluster_sizes)
  labels <- rep(seq_len(K), times = cluster_sizes)

  A <- matrix(0, nrow = N, ncol = N)
  W <- matrix(0, nrow = N, ncol = N)

  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {

      same_cluster <- labels[i] == labels[j]

      edge_prob <- if (same_cluster) p_in else p_out

      if (runif(1) < edge_prob) {
        A[i, j] <- 1
        A[j, i] <- 1

        if (same_cluster) {
          wij <- runif(1, weight_in_range[1], weight_in_range[2])
        } else {
          wij <- runif(1, weight_out_range[1], weight_out_range[2])
        }

        W[i, j] <- wij
        W[j, i] <- wij
      }
    }
  }

  edge_list <- which(A[upper.tri(A)] == 1, arr.ind = TRUE)
  upper_idx <- which(upper.tri(A), arr.ind = TRUE)
  edges <- upper_idx[A[upper.tri(A)] == 1, , drop = FALSE]

  list(
    adjacency = A,
    weights = W,
    edge_list = edges,
    labels = labels
  )
}
