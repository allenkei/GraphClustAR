#include <RcppArmadillo.h>
#include <math.h>
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace std;


//' Build AR(p) design matrices
//'
//' Build AR design matrices given time series data for each node.
//'
//' @param TS_by_node Numeric matrix of size N x n. Each row is the time series for a node.
//' @param lag_p Integer lag p for the AR(p) model.
//'
//' @return A list with two elements:
//' \describe{
//'   \item{X_list}{List of length N; \code{X_list[[i]]} is (n - p) x p design matrix for node i.}
//'   \item{Y_list}{List of length N; \code{Y_list[[i]]} is length (n - p) response vector for node i.}
//' }
//' @examples
//' N = 30 # nodes
//' n = 100 # time points
//' mu = 0
//' sigma = 1
//' phi = 0.8 # AR parameter
//' x_init <- rnorm(N, mean = mu, sd = sigma / sqrt(1 - phi^2))
//' TS_by_node <- matrix(0, nrow = N, ncol = n)
//' TS_by_node[, 1] <- x_init
//' eps <- matrix(rnorm(N * (n-1), 0, 1), nrow = N, ncol = n-1)
//' for (t in 2:n)  TS_by_node[,t] <- mu + phi * (TS_by_node[,t-1] - mu) + eps[,t-1] # AR(1)
//' ar_X_Y <- get_ar_X_Y(TS_by_node, lag_p=1)
//' TS_by_node[1,1:10]
//' ar_X_Y$X_list[[1]][1:6,] # design matrix with lag_p
//' ar_X_Y$Y_list[[1]][1:6]  # response at lag_p + 1
//' @export
// [[Rcpp::export]]
Rcpp::List get_ar_X_Y(const arma::mat TS_by_node, const int lag_p) {
  // TS_by_node is mean-centered

  int N = TS_by_node.n_rows;   // number of nodes
  int n = TS_by_node.n_cols;   // length of time series per node

  int T_eff = n - lag_p;  // n - p
  //int d = lag_p + 1;    // NOT USED

  Rcpp::List X_list(N);
  Rcpp::List Y_list(N);

  for (int i = 0; i < N; ++i) {
   arma::rowvec yi = TS_by_node.row(i); // time series for node i

   arma::mat Xi(T_eff, lag_p); // (n - p) x p
   arma::vec Yi(T_eff);    // (n - p)

   // in R: t = p+1,...,n
   // in C++: u = lag_p,...,n-1 (0-based)
   for (int t = 0; t < T_eff; ++t) {
     int u = t + lag_p;   // current time index in 0-based

     // response: y_t
     Yi(t) = yi(u);

     // lagged covariates: y_{t-1}, ..., y_{t-p}
     // Note the order from (t-1) to (t-p) from left to right
     for (int k = 0; k < lag_p; ++k) {
       Xi(t, k) = yi(u - k - 1);
     }
   }

   X_list[i] = Xi;
   Y_list[i] = Yi;
  }

  return Rcpp::List::create(
   Rcpp::Named("X_list") = X_list,
   Rcpp::Named("Y_list") = Y_list
  );
}



//' Get graph information
//'
//' Convert weighted adjacency matrix to (1) edge list and (2) node degrees
//'
//' @param W Numeric matrix (N x N) weighted adjacency matrix. Must be symmetric.
//'
//' @return A list with:
//'   - edge_list: numeric matrix (i, j, w_ij)
//'   - node_degree: integer vector of node degrees
//'
//' @examples
//' # 5-node weighted, symmetric adjacency matrix
//' W <- matrix(0, nrow = 5, ncol = 5)
//' W[1, 2] <- 1.5; W[1, 4] <- 2.0; W[2, 3] <- 0.8; W[4, 5] <- 3.2
//' W <- W + t(W)
//' get_graph_info(W)
//' @export
// [[Rcpp::export]]
Rcpp::List get_graph_info(const arma::mat &W) {


  int N = W.n_rows;
  int E = 0;

  arma::vec node_degree(N, arma::fill::zeros);

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      if (W(i, j) > 0.0) {
        node_degree(i)++;
        node_degree(j)++;
        E++;
      }
    }
  }

  // build edge list
  arma::mat edge_list(E, 3);
  int idx = 0;

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      double wij = W(i, j);
      if (wij > 0.0) {
        edge_list(idx, 0) = i;   // 0-based index
        edge_list(idx, 1) = j;   // 0-based index
        edge_list(idx, 2) = wij;
        idx++;
      }
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("edge_list") = edge_list,
    Rcpp::Named("node_degree") = node_degree
  );
}



//' Update phi
//'
//' ADMM update for phi using edge-wise neighbor aggregation.
//' This function will not be export in the final version.
//'
//' @param X_list List of length N; each \code{X_list[[i]]} is (n-p x p) design matrix X_i.
//' @param Y_list List of length N; each \code{Y_list[[i]]} is length n-p response vector Y_i.
//' @param phi N x d matrix of node parameters.
//' @param nu Edge-wise auxiliary variables, E x d matrix.
//' @param theta Edge-wise scaled dual variables, E x d matrix.
//' @param edge_list edge list (E by 3)
//' @param node_degree Numeric vector of length N. This is |B(i)| for each i.
//' @param gamma penalty for the augmentation term
//'
//' @return An N x d matrix with the updated phi.
//' @export
// [[Rcpp::export]]
arma::mat update_phi(const Rcpp::List &X_list, const Rcpp::List &Y_list,
                      arma::mat  &phi,          // N by d
                      arma::mat  &nu,           // E by d
                      arma::mat  &theta,        // E by d
                      const arma::mat  &edge_list,    // E by 3
                      const arma::vec  &node_degree, // N
                      const double gamma) {
  int N = phi.n_rows;
  int p = phi.n_cols;
  int E = nu.n_rows;

  arma::mat phi_new(N, p, arma::fill::zeros);
  arma::mat neighbor_sum(N, p, arma::fill::zeros);

  // 1. neighbor aggregation
  for (int e = 0; e < E; ++e) {
   int i = edge_list(e, 0);
   int j = edge_list(e, 1);

   // phi_j, nu_e, theta_e are 1 by d
   arma::rowvec term = phi.row(j) + nu.row(e) - theta.row(e);
   neighbor_sum.row(i) += term;
  }

  // 2. Node-wise update
  for (int i = 0; i < N; ++i) {
   arma::mat Xi = Rcpp::as<arma::mat>(X_list[i]);  // (n-p) by p
   arma::vec Yi = Rcpp::as<arma::vec>(Y_list[i]);  // n-p

   // first_term = X_i^T X_i + gamma * |B(i)| I_p
   arma::mat first_term = Xi.t() * Xi;
   first_term.diag() += gamma * node_degree[i];

   // second_term = X_i^T Y_i + gamma * neighbor_sum_i
   arma::vec second_term = Xi.t() * Yi + gamma * neighbor_sum.row(i).t();

   // Solve for phi_i
   arma::vec phi_i = arma::solve(first_term, arma::eye<arma::mat>(p,p)) * second_term;
   phi_new.row(i) = phi_i.t();
  }

  return phi_new;
}





//' Update nu
//'
//' ADMM update for nu using group soft-thresholding.
//' This function will not be export in the final version.
//'
//' @param phi N x p matrix of node parameters (one row per node).
//' @param theta E x p matrix of scaled dual variables.
//' @param edge_list E x 3 matrix. Each row e is (i, j, w_ij)
//' @param lambda GFL penalty parameter.
//' @param gamma penalty parameter for augmentation term.
//'
//' @return Updated nu matrix of size E x p.
//' @export
// [[Rcpp::export]]
arma::mat update_nu(const arma::mat &phi,          // N x p
                    const arma::mat &theta,        // E x p
                    const arma::mat &edge_list,    // E x 3: (i, j, w_ij)
                    const double lambda, const double gamma) {

  int E = edge_list.n_rows;
  int p = phi.n_cols;
  arma::mat nu(E, p, arma::fill::zeros);

  for (int e = 0; e < E; ++e) {

    int i = static_cast<int>(edge_list(e, 0));
    int j = static_cast<int>(edge_list(e, 1));
    double w_ij = edge_list(e, 2);

    arma::rowvec s_ij = phi.row(i) - phi.row(j) + theta.row(e);
    double s_norm = std::sqrt(arma::dot(s_ij, s_ij));

    double condition = 1.0 - (lambda * w_ij) / (gamma * s_norm);
    if (condition <= 0.0) {
      nu.row(e).zeros();
    } else {
      nu.row(e) = condition * s_ij;
    }
  }

  return nu;
}







//' Update theta
//'
//' ADMM update for theta
//' This function will not be export in the final version.
//'
//' @param phi N x p matrix of node parameters (one row per node).
//' @param nu E x d matrix of auxiliary variables.
//' @param theta E x p matrix of scaled dual variables.
//' @param edge_list E x 3 matrix. Each row e is (i, j, w_ij)
//'
//' @return Updated nu matrix of size E x p.
//' @export
// [[Rcpp::export]]
arma::mat update_theta(const arma::mat &phi,          // N x p
                       const arma::mat &nu,           // E x p
                       const arma::mat &theta,        // E x p
                       const arma::mat &edge_list) {

  int E = edge_list.n_rows;
  int p = phi.n_cols;
  arma::mat theta_new(E, p, arma::fill::zeros);

  for (int e = 0; e < E; ++e) {

   int i = static_cast<int>(edge_list(e, 0));
   int j = static_cast<int>(edge_list(e, 1));

   theta_new.row(e) = phi.row(i) - phi.row(j) - nu.row(e) + theta.row(e); // use theta to update theta_new

  }

  return theta_new;
}




//' ADMM for GraphClustAR
//'
//' ADMM for GraphClustAR (cpp version)
//' This function will not be export in the final version.
//'
//' @param X_list List of length N; each \code{X_list[[i]]} is (n-p x p) design matrix X_i.
//' @param Y_list List of length N; each \code{Y_list[[i]]} is length n-p response vector Y_i.
//' @param edge_list edge list (E by 3)
//' @param node_degree Numeric vector of length N. This is |B(i)| for each i.
//' @param lambda GFL penalty parameter.
//' @param gamma penalty for the augmentation term
//' @param lag_p Integer lag p for the AR(p) model.
//'
//' @return An N x d matrix with the updated phi.
//' @export
// [[Rcpp::export]]
arma::mat GraphClustAR_cpp(const Rcpp::List &X_list, const Rcpp::List &Y_list,
                           const arma::mat  &edge_list,    // E by 3
                           const arma::vec  &node_degree, // N
                           const double lambda, const double gamma, const int lag_p) {


  int E = edge_list.n_rows;
  int N = node_degree.size();
  int p = lag_p;

  // initialize parameters
  arma::mat phi(N, p, arma::fill::zeros);
  arma::mat nu(E, p, arma::fill::zeros);
  arma::mat theta(E, p, arma::fill::zeros);




}

