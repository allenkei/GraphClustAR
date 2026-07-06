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
adj_w <- graph_data$adj_w
result <- ClustARp(TS_by_node, adj_w, lag_p=1, num_clust_list=2:7,
                    lambda_list=c(0.1,0.25,0.5,0.75,1), ADMM_iter=100)
result$lambda
result$K
result$cluster
plot(result$ADMM_output$phi, col=result$cluster)
```


