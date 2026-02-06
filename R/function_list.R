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
#' N = 30 # nodes
#' n = 100 # time points
#' mu = 0
#' sigma = 1
#' phi = 0.8 # AR parameter
ClustAR1 <- function(TS_by_node, adj_w, ADMM_iter, GD_iter, lr, lambda, gamma, update_gamma, verbose){

  n <- ncol(TS_by_node)
  ar_X_Y <- get_ar_X_Y(TS_by_node, lag_p=1, intercept=FALSE)
  graph_info <- get_graph_info(adj_w)

  output <- GraphClustAR1_cpp(ar_X_Y$X_list, ar_X_Y$Y_list, graph_info$edge_list, graph_info$node_degree,
                             ADMM_iter, lambda, gamma, GD_iter, lr, update_gamma, lag_p=1, verbose)

  return(output)
}
