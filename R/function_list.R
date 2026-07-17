## usethis namespace: start
#' @useDynLib GraphClustAR, .registration = TRUE
## usethis namespace: end
NULL

#' @importFrom Rcpp sourceCpp
#' @importFrom cluster silhouette
NULL



#' Clustering nodes.
#' @description This function clusters nodes.
#' @param TS_by_node Numeric matrix of size N x n. Each row is the time series for a node.
#' @param adj_w Adjacency matrix with non-negative weights
#' @param num_clust_list vector containing list of cluster number
#' @param lambda_list vector containing list of GFL penalty
#' @param ADMM_iter ADMM iteration
#' @param phi_tol ADMM stopping criteria
#' @param verbose If TRUE, print info during parameter learning
#' @param update_gamma If TRUE, gamma is updated with a schedule
#'
#' @return A list
#'
#' @examples
#' ts_data <- gen_ar1_by_cluster(cluster_sizes = c(50, 50, 50), n = 100, seed=123)
#' graph_data <- gen_weighted_block_graph(cluster_sizes = c(50, 50, 50), seed=123)
#' TS_by_node <- ts_data$TS_by_node
#' adj_w <- graph_data$adj_w
#' result <- GraphClustARp(TS_by_node, adj_w, lag_p=1, num_clust_list=2:7,
#'                    lambda_list=c(0.1,0.25,0.5,0.75,1), ADMM_iter=100)
#' result$lambda
#' result$K
#' result$cluster
#' plot(result$ADMM_output$phi, col=result$cluster)
#'
#' @export
GraphClustARp <- function(TS_by_node, adj_w, lag_p, num_clust_list, lambda_list, ADMM_iter,
                     phi_tol=0.001, verbose=F, update_gamma=F){

  ts_length <- ncol(TS_by_node)
  ar_X_Y <- get_ar_X_Y(TS_by_node, lag_p)
  graph_info <- get_graph_info(adj_w)

  n_lambda <- length(lambda_list)

  BIC <- numeric(n_lambda)
  K_selected <- integer(n_lambda)
  Silhouette <- numeric(n_lambda)

  phi_list <- vector("list", n_lambda)
  cluster_list <- vector("list", n_lambda)
  ADMM_output_list <- vector("list", n_lambda)

  for(i in 1:length(lambda_list)){

    lambda <- lambda_list[i]
    gamma <- lambda_list[i]

    ADMM_output <- GraphClustARp_cpp(ar_X_Y$X_list, ar_X_Y$Y_list, graph_info$edge_list, graph_info$node_degree,
                                  lag_p, ts_length, ADMM_iter, lambda, gamma, phi_tol, update_gamma, verbose)

    phi_hat <- ADMM_output$phi
    D_phi <- dist(phi_hat)
    sil_score <- numeric(length(num_clust_list))

    # START OF SELECTING NUM OF CLUSTERs
    for (k in seq_along(num_clust_list)) {

      K_candidate <- num_clust_list[k]
      cls_k <- kmeans(phi_hat, centers = K_candidate, iter.max = 100, nstart = 20)
      sil_k <- cluster::silhouette(cls_k$cluster, D_phi)
      sil_score[k] <- mean(sil_k[, "sil_width"])

    }
    # END OF SELECT NUM OF CLUSTERS

    K_hat <- num_clust_list[which.max(sil_score)]
    cls <- kmeans(phi_hat, centers = K_hat, iter.max = 100, nstart = 20)
    K_selected[i] <- K_hat

    BIC[i] <- cal_ar_BIC(ar_X_Y$X_list, ar_X_Y$Y_list, phi_hat, K_hat, ts_length)

    phi_list[[i]] <- phi_hat
    cluster_list[[i]] <- cls$cluster
    ADMM_output_list[[i]] <- ADMM_output

  }

  #select lambda by minimum BIC
  best_lambda_idx <- which.min(BIC)

  output <- list(
    lambda = lambda_list[best_lambda_idx],
    K = K_selected[best_lambda_idx],
    cluster = cluster_list[[best_lambda_idx]],
    BIC = BIC[best_lambda_idx],
    ADMM_output = ADMM_output_list[[best_lambda_idx]]
  )

  return(output)

}



#' Generate clustered AR(1) time series
#'
#' Generate nodal AR(1) time series with cluster-specific mean,
#' autoregressive coefficient, and innovation variance.
#'
#' @param cluster_sizes Integer vector of cluster sizes.
#' @param n Length of each time series.
#' @param mu_vec Numeric vector of cluster means.
#' @param phi_vec Numeric vector of cluster AR(1) coefficients.
#' @param sigma_vec Numeric vector of cluster innovation standard deviations.
#' @param stationary_init Logical; initialize from the stationary distribution when possible.
#' @param seed Optional random seed.
#'
#' @return A list containing:
#' \describe{
#'   \item{TS_by_node}{Numeric matrix of size \code{N x n}; each row is a node's time series.}
#'   \item{labels}{Cluster label for each node.}
#' }
#'
#' @examples
#' ts_data <- gen_ar1_by_cluster(cluster_sizes = c(50, 50, 50), n = 100)
#'
#' @export
gen_ar1_by_cluster <- function(
    cluster_sizes = c(50, 50, 50),
    n = 100,
    mu_vec = c(0, 2, -2),
    phi_vec = c(0.1, 0.7, -0.5),
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
    labels = labels
  )
}



#' Generate a weighted block graph
#'
#' Generate a weighted undirected graph with cluster-specific edge probabilities and edge weights.
#'
#' @param cluster_sizes Integer vector of cluster sizes.
#' @param p_in Probability of an edge within a cluster.
#' @param p_out Probability of an edge between clusters.
#' @param weight_in_range Length-2 numeric vector giving the range of within-cluster edge weights.
#' @param weight_out_range Length-2 numeric vector giving the range of between-cluster edge weights.
#' @param seed Optional random seed.
#'
#' @return A list containing:
#' \describe{
#'   \item{adj_w}{Weighted adjacency matrix.}
#'   \item{labels}{Cluster label for each node.}
#' }
#'
#' @examples
#' graph_data <- gen_weighted_block_graph(cluster_sizes = c(50, 50, 50))
#'
#' @export
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

  W <- matrix(0, nrow = N, ncol = N)

  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {

      same_cluster <- labels[i] == labels[j]

      edge_prob <- if (same_cluster) p_in else p_out

      if (runif(1) < edge_prob) {

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

  list(
    adj_w = W,
    labels = labels
  )
}
