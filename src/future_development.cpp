
#include <RcppArmadillo.h>
#include <math.h>
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace std;




////////////////////////////
// FOR FUTURE DEVELOPMENT //
////////////////////////////



//' Build AR(p) design matrices
//'
//' Build AR design matrices given time series data for each node.
//'
//' @param TS_by_node Numeric matrix of size N x n. Each row is the time series for a node.
//' @param lag_p Integer lag p for the AR(p) model.
//' @param intercept TRUE or FALSE to include intercept
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
Rcpp::List get_ar_X_Y_future(const arma::mat TS_by_node, const int lag_p, bool intercept) {

 int N = TS_by_node.n_rows;   // number of nodes
 int n = TS_by_node.n_cols;   // length of time series per node

 int T_eff = n - lag_p;  // n - p
 int d_dim = lag_p + 1;

 Rcpp::List X_list(N); Rcpp::List Y_list(N);

 for (int i = 0; i < N; ++i) {

   arma::rowvec yi = TS_by_node.row(i); // time series for node i

   arma::mat Xi; arma::vec Yi;

   if (intercept) {
     Xi.set_size(T_eff, d_dim); // (n - p) x d
   } else {
     Xi.set_size(T_eff, lag_p); // (n - p) x p
   }

   Yi.set_size(T_eff); // (n - p)

   // in R: t = p+1,...,n
   // in C++: u = lag_p,...,n-1 (0-based)
   for (int t = 0; t < T_eff; ++t) {
     int u = t + lag_p;   // current time index in 0-based

     // response: y_t
     Yi(t) = yi(u);

     if(intercept){ // include intercept

       // intercept
       Xi(t,0) = 1.0; // USED WITH for (int k = 1; k <= lag_p; ++k)

       // lagged covariates: y_{t-1}, ..., y_{t-p}
       // Note the order from (t-1) to (t-p) from left to right column
       for (int k = 1; k <= lag_p; ++k) {
         Xi(t, k) = yi(u - k);
       }

     }else{ // don't include an intercept

       // lagged covariates: y_{t-1}, ..., y_{t-p}
       // Note the order from (t-1) to (t-p) from left to right column
       for (int k = 0; k < lag_p; ++k) {
         Xi(t, k) = yi(u - k - 1);
       }

     }

   }

   X_list[i] = Xi; Y_list[i] = Yi;
 }

 return Rcpp::List::create(
   Rcpp::Named("X_list") = X_list,
   Rcpp::Named("Y_list") = Y_list
 );
}






//' Update phi of AR(1)
//'
//' ADMM update for phi of AR(1).
//' This function will not be export in the final version.
//'
//' @param X_list List of length N; each \code{X_list[[i]]} is (n-p x d) design matrix X_i.
//' @param Y_list List of length N; each \code{Y_list[[i]]} is length n-p response vector Y_i.
//' @param mu N x d matrix of node parameters.
//'
//' @return An N x d matrix with the updated phi.
//' @export
// [[Rcpp::export]]
arma::vec update_phi_AR1(const Rcpp::List &X_list, const Rcpp::List &Y_list, arma::vec &mu) {
 int N = mu.n_elem;
 arma::vec phi(N, arma::fill::zeros);

 for (int i = 0; i < N; ++i) {
   arma::mat Xi = Rcpp::as<arma::vec>(X_list[i]);  // n-1
   arma::vec Yi = Rcpp::as<arma::vec>(Y_list[i]);  // n-1

   double numer = arma::dot( Xi - mu(i), Yi - mu(i) );
   double deno = arma::dot( Xi - mu(i), Xi - mu(i) );

   phi(i) = numer/deno;
 }

 return phi;
}







//' Update mu of AR(1)
//'
//' ADMM update for mu using edge-wise neighbor aggregation.
//' This function will not be export in the final version.
//'
//' @param X_list List of length N; each \code{X_list[[i]]} is (n-p x d) design matrix X_i.
//' @param Y_list List of length N; each \code{Y_list[[i]]} is length n-p response vector Y_i.
//' @param phi N vector of node parameters.
//' @param nu Edge-wise auxiliary variables, E vector.
//' @param theta Edge-wise scaled dual variables, E vector.
//' @param edge_list edge list (E by 3)
//' @param node_degree Numeric vector of length N. This is |B(i)| for each i.
//' @param gamma penalty for the augmentation term
//' @param T_length length of nodal time series
//'
//' @return An N vector with the updated mu.
//' @export
// [[Rcpp::export]]
arma::vec update_mu_AR1(const Rcpp::List &X_list, const Rcpp::List &Y_list,
                       arma::vec  &phi,   // N
                       arma::vec  &mu,    // N
                       arma::vec  &nu,    // E
                       arma::vec  &theta, // E
                       const arma::mat  &edge_list,   // E by 3
                       const arma::vec  &node_degree, // N
                       const double gamma, const double T_length) {

 int N = phi.n_elem; // num of nodes
 int E = nu.n_elem;  // num of edges

 arma::vec mu_new(N, arma::fill::zeros);
 arma::vec neighbor_sum(N, arma::fill::zeros);

 // 1. neighbor aggregation
 for (int e = 0; e < E; ++e) {
   int i = edge_list(e, 0);
   int j = edge_list(e, 1);
   neighbor_sum(i) += mu(j) + nu(e) - theta(e); // mu_j, nu_e, theta_e are scalar
 }

 // 2. Node-wise update
 for (int i = 0; i < N; ++i) {
   arma::vec Xi = Rcpp::as<arma::vec>(X_list[i]);  // n-1
   arma::vec Yi = Rcpp::as<arma::vec>(Y_list[i]);  // n-1

   double deno = (T_length-1) *(1.0 - phi(i))*(1.0 - phi(i)) + gamma * node_degree(i);

   double numer = (1.0 - phi(i)) * arma::accu(Yi - phi(i) * Xi);
   numer += gamma * neighbor_sum(i);

   mu_new(i) = numer/deno;
 }

 return mu_new;
}




//' Update nu of AR(1)
//'
//' ADMM update for mu using edge-wise neighbor aggregation.
//' This function will not be export in the final version.
//'
//' @param mu N vector of node parameters.
//' @param theta Edge-wise scaled dual variables, E vector.
//' @param edge_list edge list (E by 3)
//' @param node_degree Numeric vector of length N. This is |B(i)| for each i.
//' @param lambda penalty parameter
//' @param gamma penalty for the augmentation term
//'
//' @return An N vector with the updated mu.
//' @export
// [[Rcpp::export]]
arma::vec update_nu_AR1(arma::vec &mu,                // N
                       arma::vec  &theta,             // E
                       const arma::mat  &edge_list,   // E by 3
                       const double lambda, const double gamma) {
 int E = theta.n_elem;  // num of edges
 arma::vec nu_new(E, arma::fill::zeros);

 for (int e = 0; e < E; ++e) {
   int i = edge_list(e, 0);
   int j = edge_list(e, 1);
   double d_ij = mu(i) - mu(j) + theta(e); // mu_i, mu_j, theta_e are scalar
   double w_ij = edge_list(e, 2); // edge weights

   double adjustment = (lambda/gamma) * w_ij;

   if( d_ij > adjustment ){ // greater than positive
     nu_new(e) = d_ij - adjustment;
   }else if( d_ij < -adjustment ){ // less than negative
     nu_new(e) = d_ij + adjustment;
   }else{
     nu_new(e) = 0.0;
   }
 }

 return nu_new;
}




//' Update theta of AR(1)
//'
//' ADMM update for theta
//' This function will not be export in the final version.
//'
//' @param phi N vector of node parameters.
//' @param nu E vector of auxiliary variables.
//' @param theta E vector of scaled dual variables.
//' @param edge_list E by 3 matrix. Each row e is (i, j, w_ij)
//'
//' @return Updated nu matrix of size E by p.
//' @export
// [[Rcpp::export]]
arma::vec update_theta_AR1(const arma::vec &mu, const arma::vec &nu, const arma::vec &theta,
                          const arma::mat &edge_list) {

 int E = edge_list.n_rows;
 arma::vec theta_new(E, arma::fill::zeros);

 for (int e = 0; e < E; ++e) {
   int i = edge_list(e, 0);
   int j = edge_list(e, 1);
   theta_new(e) = mu(i) - mu(j) - nu(e) + theta(e); // use theta to update theta_new
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
//' @param ADMM_iter ADMM iteration
//' @param T_length length of nodal time series
//' @param lambda GFL penalty parameter.
//' @param gamma penalty for the augmentation term.
//' @param update_gamma If TRUE, gamma is updated with a schedule
//' @param verbose If TRUE, print info during learning.
//'
//' @return An N x d matrix with the updated phi.
//' @export
// [[Rcpp::export]]
Rcpp::List GraphClust_AR1_cpp_deprecate(const Rcpp::List &X_list, const Rcpp::List &Y_list,
                             const arma::mat  &edge_list,   // E by 3
                             const arma::vec  &node_degree, // N
                             const int ADMM_iter, const int T_length,
                             const double lambda, double gamma,
                             bool update_gamma, bool verbose) {

 int E = edge_list.n_rows;
 int N = node_degree.size();
 double primal_res, dual_res, converged;

 arma::vec mu_old, nu_old; // arma::mat nu, arma::mat theta,
 arma::vec phi(N, arma::fill::zeros);
 arma::vec mu(N, arma::fill::zeros);
 arma::vec nu(E, arma::fill::zeros);
 arma::vec theta(E, arma::fill::zeros);

 for(int iter = 0; iter < ADMM_iter; ++iter){

   // one ADMM iteration
   if(verbose){Rcout << "ADMM iter = " << iter+1 << "\n";}

   phi = update_phi_AR1(X_list, Y_list, mu);
   if(verbose){Rcout << "    phi updated" << "\n";}

   mu_old = mu;
   mu = update_mu_AR1(X_list, Y_list, phi, mu_old, nu, theta, edge_list, node_degree, gamma, T_length);

   converged = arma::norm(mu - mu_old, 2);
   if(verbose){Rcout << "    mu updated" << "\n";}
   if(verbose){Rcout << "        coverged = " << converged << "\n";}

   nu_old = nu;
   nu = update_nu_AR1(mu, theta, edge_list, lambda, gamma);
   if(verbose){Rcout << "    nu updated" << "\n";}

   theta = update_theta_AR1(mu, nu, theta, edge_list);
   if(verbose){Rcout << "    theta updated" << "\n";}

   for (int e = 0; e < E; ++e) {
     int i = edge_list(e, 0);
     int j = edge_list(e, 1);
     primal_res += (mu(i) - mu(j) - nu(e)) * (mu(i) - mu(j) - nu(e));
   }
   primal_res = std::sqrt(primal_res);
   dual_res = arma::norm(nu - nu_old, 2);
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
   Rcpp::Named("phi") = phi,
   Rcpp::Named("mu") = mu,
   Rcpp::Named("nu") = nu,
   Rcpp::Named("theta") = theta
 );

}


























