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
//'   \item{X_list}{List of length N; \code{X_list[[i]]} is (n - p) x d design matrix for node i.}
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
//' ar_X_Y <- get_ar_X_Y(TS_by_node, lag_p=4, intercept=FALSE)
//' TS_by_node[1,1:10]         # first 10 time points for node 1
//' ar_X_Y$X_list[[1]][1:6,]   # design matrix with lag_p
//' ar_X_Y$Y_list[[1]][1:6]    # response at lag_p + 1
//' @export
// [[Rcpp::export]]
Rcpp::List get_ar_X_Y(const arma::mat TS_by_node, const int lag_p, bool intercept) {

  int N = TS_by_node.n_rows;   // number of nodes
  int n = TS_by_node.n_cols;   // length of time series per node

  int T_eff = n - lag_p;  // n - p
  //int d_dim = lag_p + 1;

  Rcpp::List X_list(N); Rcpp::List Y_list(N);

  for (int i = 0; i < N; ++i) {

    arma::rowvec yi = TS_by_node.row(i); // time series for node i

    arma::mat Xi; arma::vec Yi;
    Xi.set_size(T_eff, lag_p); // (n - p) x p
    Yi.set_size(T_eff); // (n - p)

    // in R: t = p+1,...,n
    // in C++: u = lag_p,...,n-1 (0-based)
    for (int t = 0; t < T_eff; ++t) {
      int u = t + lag_p;   // current time index in 0-based

      // response: y_t
      Yi(t) = yi(u);

      // lagged covariates: y_{t-1}, ..., y_{t-p}
      // Note the order from (t-1) to (t-p), which is left to right
      for (int k = 0; k < lag_p; ++k) {
        Xi(t, k) = yi(u - k - 1);
      }

    }

    X_list[i] = Xi; Y_list[i] = Yi;
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
//' @param adj_w Numeric matrix (N x N) weighted adjacency matrix. Must be symmetric.
//'
//' @return A list with:
//'   - edge_list: numeric matrix (i, j, w_ij)
//'   - node_degree: integer vector of node degrees
//'
//' @examples
//' # 5-node weighted, symmetric adjacency matrix
//' adj_w <- matrix(0, nrow = 5, ncol = 5)
//' adj_w[1, 2] <- 1.5; adj_w[1, 4] <- 2.0; adj_w[2, 3] <- 0.8; adj_w[4, 5] <- 3.2
//' adj_w <- adj_w + t(adj_w)
//' get_graph_info(adj_w)
//' @export
// [[Rcpp::export]]
Rcpp::List get_graph_info(const arma::mat &adj_w) {

  int N = adj_w.n_rows;
  int E = 0;
  arma::vec node_degree(N, arma::fill::zeros);

  for (int i = 0; i < N; ++i) {
    for (int j = i+1; j < N; ++j) { // store both (1,2) and (2,1) for updates
      if (adj_w(i, j) > 0.0) {
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
    for (int j = i+1; j < N; ++j) {
      double wij = adj_w(i, j);
      if (wij > 0.0) {
        edge_list(idx, 0) = i;   // 0-based index i
        edge_list(idx, 1) = j;   // 0-based index j
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
//' @param X_list List of length N; each \code{X_list[[i]]} is (n-p x d) design matrix X_i.
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
arma::mat update_theta(const Rcpp::List &X_list, const Rcpp::List &Y_list,
                      arma::mat  &theta,        // N by d
                      arma::mat  &nu,           // E by d
                      arma::mat  &eta,          // E by d
                      const arma::mat  &edge_list,   // E by 3
                      const arma::vec  &node_degree, //
                      const int GD_iter, const double lr,
                      const double gamma) {
  int N = theta.n_rows;
  int d_dim = theta.n_cols; // should be 2 for AR(1)
  int E = nu.n_rows;

  for (int i = 0; i < N; ++i) { // for each node i

    arma::vec Xi = Rcpp::as<arma::vec>(X_list[i]);  // length n-1, data from 1 to T-1
    arma::vec Yi = Rcpp::as<arma::vec>(Y_list[i]);  // length n-1, data from 2 to T
    const int n = Yi.n_elem;

    // initialization for node i
    double mu_i  = theta(i, 0);
    double phi_i = theta(i, 1);

    // Gradient Descent (neighbors fixed at current theta_new)
    for (int it = 0; it < GD_iter; ++it) {

      arma::vec Xi_centered = Xi - mu_i;
      arma::vec res_vec = Yi - mu_i - phi_i * Xi_centered;

      // Gradient from loss function
      double sum_res = arma::accu(res_vec);
      double res_dot_Xc = arma::dot(res_vec, Xi_centered);

      double grad_mu  = -2.0 * (1.0 - phi_i) * sum_res;
      double grad_phi = -2.0 * res_dot_Xc;

      // gradient from augmentation term
      // gamma * sum_{j in B(i)} (theta_i - theta_j - nu_ij + eta_ij)
      for (int e = 0; e < E; ++e) {
        int u = edge_list(e, 0);
        int v = edge_list(e, 1);

        double aug_mu  = theta(u, 0) - theta(v, 0) - nu(e, 0) + eta(e, 0);
        double aug_phi = theta(u, 1) - theta(v, 1) - nu(e, 1) + eta(e, 1);

        if (i == u) {
          grad_mu  += gamma * aug_mu;
          grad_phi += gamma * aug_phi;
        } else if (i == v) {
          grad_mu  -= gamma * aug_mu;
          grad_phi -= gamma * aug_phi;
        }
      }

      // stopping check
      //double gnorm = std::sqrt(g_mu * g_mu + g_phi * g_phi);
      //if (gnorm < tol) break;

      //Rcout << "grad_mu = " << grad_mu << "\n";
      //Rcout << "grad_phi = " << grad_phi << "\n";

      // gradient step
      mu_i  -= lr * grad_mu;
      phi_i -= lr * grad_phi;

      // project to causality region |phi| < 1
      //if (phi >  phi_clip) phi =  phi_clip;
      //if (phi < -phi_clip) phi = -phi_clip;

    }

    theta(i, 0) = mu_i;
    theta(i, 1) = phi_i;

  }

  return theta;
}





//' Update nu
//'
//' ADMM update for nu using group soft-thresholding.
//' This function will not be export in the final version.
//'
//' @param phi N by d matrix of node parameters (one row per node).
//' @param theta E by d matrix of scaled dual variables.
//' @param edge_list E by 3 matrix. Each row e is (i, j, w_ij)
//' @param lambda GFL penalty parameter.
//' @param gamma penalty parameter for augmentation term.
//'
//' @return Updated nu matrix of size E by d.
//' @export
// [[Rcpp::export]]
arma::mat update_nu(const arma::mat &theta,          // N by d
                   const arma::mat &eta,        // E by d
                   const arma::mat &edge_list,    // E by 3: (i, j, w_ij)
                   const double lambda, const double gamma) {

 int E = edge_list.n_rows;
 int d_dim = theta.n_cols;
 arma::mat nu(E, d_dim, arma::fill::zeros);

 for (int e = 0; e < E; ++e) {

   int i = edge_list(e, 0);
   int j = edge_list(e, 1);
   double w_ij = edge_list(e, 2);

   arma::rowvec s_ij = theta.row(i) - theta.row(j) + eta.row(e);
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
//' @param phi N by d matrix of node parameters.
//' @param nu E by d matrix of auxiliary variables.
//' @param theta E by d matrix of scaled dual variables.
//' @param edge_list E by 3 matrix. Each row e is (i, j, w_ij)
//'
//' @return Updated nu matrix of size E by p.
//' @export
// [[Rcpp::export]]
arma::mat update_eta(const arma::mat &theta,          // N by d
                      const arma::mat &nu,           // E by d
                      const arma::mat &eta,        // E by d
                      const arma::mat &edge_list) {

 int E = edge_list.n_rows;
 int d_dim = theta.n_cols;
 arma::mat eta_new(E, d_dim, arma::fill::zeros);

 for (int e = 0; e < E; ++e) {
   int i = static_cast<int>(edge_list(e, 0));
   int j = static_cast<int>(edge_list(e, 1));
   eta_new.row(e) = theta.row(i) - theta.row(j) - nu.row(e) + eta.row(e); // use theta to update theta_new
 }

 return eta_new;
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
//' @param ADMM_iter ADMM iteration
//' @param lambda GFL penalty parameter.
//' @param gamma penalty for the augmentation term.
//' @param lag_p Integer lag p for the AR(p) model.
//' @param verbose If TRUE, print info during learning.
//'
//' @return An N x d matrix with the updated phi.
//' @export
// [[Rcpp::export]]
Rcpp::List GraphClustAR1_cpp(const Rcpp::List &X_list, const Rcpp::List &Y_list,
                           const arma::mat  &edge_list,   // E by 3
                           const arma::vec  &node_degree, // N
                           const int ADMM_iter,
                           const double lambda, double gamma,
                           const int GD_iter, const double lr,
                           bool update_gamma, const int lag_p, bool verbose) {


  int E = edge_list.n_rows;
  int N = node_degree.size();
  int d_dim = lag_p + 1; // should be 2 for AR(1)
  double coverged, primal_res, dual_res;

  arma::mat theta_old, nu_old;
  arma::mat theta(N, d_dim, arma::fill::ones);
  arma::mat nu(E, d_dim, arma::fill::ones);
  arma::mat eta(E, d_dim, arma::fill::ones);

  for(int iter = 0; iter < ADMM_iter; ++iter){
    if(verbose){Rcout << "ADMM iter = " << iter+1 << "\n";}

    theta_old = theta;
    theta = update_theta(X_list, Y_list, theta, nu, eta, edge_list, node_degree, GD_iter, lr, gamma);

    //if(verbose){Rcout << "theta = " << theta << "\n";}

    coverged = arma::norm(theta - theta_old, "fro");
    if(verbose){Rcout << "    theta updated" << "\n";}
    if(verbose){Rcout << "        coverged = " << coverged << "\n";}

    nu_old = nu;
    nu = update_nu(theta, eta, edge_list, lambda, gamma);
    if(verbose){Rcout << "    nu updated" << "\n";}

    eta = update_eta(theta, nu, eta, edge_list);
    if(verbose){Rcout << "    eta updated" << "\n";}


    for (int e = 0; e < E; ++e) {
      int i = edge_list(e, 0);
      int j = edge_list(e, 1);
      arma::rowvec r = theta.row(i) - theta.row(j) - nu.row(e);
      primal_res += arma::dot(r, r);
    }

    primal_res = std::sqrt(primal_res);
    dual_res = arma::norm(nu - nu_old, "fro");
    if(verbose){Rcout << "        primal_res = " << primal_res << "\n";}
    if(verbose){Rcout << "        dual_res   = " << dual_res << "\n";}

    if(update_gamma){
      if(primal_res > dual_res * 10.0){
        gamma *= 2.0;
        theta /= 2.0;
        if(verbose){Rcout << "        --- gamma is increased to " << gamma << "\n";}
      }else if(dual_res > primal_res * 10.0){
        gamma /= 2.0;
        theta *= 2.0;
        if(verbose){Rcout << "        --- gamma is decreased to " << gamma << "\n";}
      }
    }

  }

  return Rcpp::List::create(
    Rcpp::Named("theta") = theta,
    Rcpp::Named("nu") = nu,
    Rcpp::Named("eta") = eta
  );

}





