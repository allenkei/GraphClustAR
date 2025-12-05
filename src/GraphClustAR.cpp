#include <RcppArmadillo.h>
#include <math.h>
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace std;


//' Build AR(p) design matrices for each node
//'
//' @param TS_by_node Numeric matrix of size N x n. Each row is the time series for a node.
//' @param lag_p Integer lag p for the AR(p) model.
//'
//' @return A list with two elements:
//' \describe{
//'   \item{X_list}{List of length N; each X is (n - p) x (p + 1) design matrix for node i.}
//'   \item{Y_list}{List of length N; each Y is length (n - p) response vector for node i.}
//' }
//' @export
// [[Rcpp::export]]
Rcpp::List get_ar_X_Y(arma::mat TS_by_node, int lag_p) {

 int N = TS_by_node.n_rows;   // number of nodes
 int n = TS_by_node.n_cols;   // length of time series per node

 int T_eff = n - lag_p;  // n - p
 int d = lag_p + 1;      // p + 1

 Rcpp::List X_list(N);
 Rcpp::List Y_list(N);

 for (int i = 0; i < N; ++i) {
   arma::rowvec yi = TS_by_node.row(i);   // time series for node i (1 x n)

   arma::mat Xi(T_eff, d);          // (n - p) x (p + 1)
   arma::vec Yi(T_eff);             // length (n - p)

   // t index in R: t = p+1,...,n
   // in 0-based C++: u = lag_p,...,n-1
   for (int t = 0; t < T_eff; ++t) {
     int u = t + lag_p;   // current time index in 0-based

     // response: y_t
     Yi(t) = yi(u);

     // intercept term
     Xi(t, 0) = 1.0;

     // lagged covariates: y_{t-1}, ..., y_{t-p}
     for (int k = 1; k <= lag_p; ++k) {
       Xi(t, k) = yi(u - k);
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
