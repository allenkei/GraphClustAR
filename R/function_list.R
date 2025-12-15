## usethis namespace: start
#' @useDynLib GraphClustAR, .registration = TRUE
## usethis namespace: end
NULL

#' @importFrom Rcpp sourceCpp
NULL



#' Clustering nodes.
#' @description This function clusters nodes.
#' @param TS_by_node Numeric matrix of size N x n. Each row is the time series for a node.
#' @param lag_p Integer lag p for the AR(p) model.
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
#' x_init <- rnorm(N, mean = mu, sd = sigma / sqrt(1 - phi^2))
#' TS_by_node <- matrix(0, nrow = N, ncol = n)
#' TS_by_node[, 1] <- x_init
#' eps <- matrix(rnorm(N * (n-1), 0, 1), nrow = N, ncol = n-1)
#' for (t in 2:n)  TS_by_node[,t] <- mu + phi * (TS_by_node[,t-1] - mu) + eps[,t-1] # AR(1)
#' # clust_result <- ClustAR(TS_by_node, lag_p) # AR(p)
#' # compare TS_by_node[1,] with clust_result$X_list[[1]]
ClustAR <- function(TS_by_node, lag_p){

  output = get_ar_X_Y(TS_by_node, lag_p)

  return(output)
}
