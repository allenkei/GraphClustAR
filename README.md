# R package: GraphClustAR

# Install the package in R
```r
library(devtools)
install_github("allenkei/GraphClustAR")
library(GraphClustAR)
```

# Demonstration
```r
ts_data <- gen_ar1_by_cluster(cluster_sizes = c(50, 50, 50), n = 100, seed=123)
graph_data <- gen_weighted_block_graph(cluster_sizes = c(50, 50, 50), seed=123)
TS_by_node <- ts_data$TS_by_node
adj_w <- graph_data$weights
labels <- ts_data$labels


result <- ClustARp(TS_by_node, adj_w, lag_p=1, ADMM_iter=200, lambda=1, gamma=1, 
                    update_gamma=FALSE, verbose=TRUE)
plot(result$phi, col=labels) # Visualization of the two AR(1) parameters
```


