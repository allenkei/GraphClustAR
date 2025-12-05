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
#' @return A list with two elements: X and Y.
#' @export
#'
#' @examples
#' #clust_result <- ClustAR(TS_by_node, lag_p)
ClustAR <- function(TS_by_node, lag_p){

  output = get_ar_X_Y(TS_by_node, lag_p)

  return(output)
}
