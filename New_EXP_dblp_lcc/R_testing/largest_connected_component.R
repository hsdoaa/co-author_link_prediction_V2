library(igraph)
library(linkprediction)
library(MASS)
library(reshape2)


 
e3<-read.csv(file="co_authership_testing.csv", header=FALSE) #
#e3<-read.csv(file="Largest_connected_component_edgelist.csv", header=FALSE) 
#G<-graph.adjacency(e3)
e3[,1]<-as.character(e3[,1]) 
e3[,2]<-as.character(e3[,2])
e3<-as.matrix(e3)

g<-graph.edgelist(e3[,1:2],directed=FALSE)




# This is a multigraph! There are multiple edges in some pairs
#has.multiple(g)
#g <- simplify(g) # simplify, i.e. remove multiple edges into one.Should not change the results though.
#typeof(g)

# Large connected components
k <- clusters(g)
# Largest connected component
lcc <- induced_subgraph(g, V(g)[k$membership == which.max(k$csize)])
typeof(lcc)
V(lcc)
#train_nodes_lcc<-V(lcc)$name# To get node names
#write.matrix(train_nodes_lcc,file="train_nodes_ArnetMiner_lcc.txt") #To retrieve the names of all the vertices in the graph g
#x<-get.edgelist(lcc)
#write.matrix(x,file="Largest_connected_component_edgelist.txt")

##
ad_matrix_testing<-get.adjacency(lcc,sparse=FALSE)
write.csv(ad_matrix_testing,file="ad_matrix_training_lcc_330000.csv")

#library(reshape2) # To import melt function
#x_train<-melt(ad_matrix_training) # melt to create an edge list from the matrix ad_matrix_tesing
#write.csv(x_train,file="Labeled_full_edgelist_lcc_train_years.csv")

#x_train[0,]

#get a graph from line adjacency matrix.  line_ad_matrix
#library(igraph)

#dat=read.csv("square_line_ad_matrix_train_330000_lcc.csv") # read .csv file
#m=as.matrix(dat)
#mode(m) <- "numeric"
#g_line<-graph.adjacency(adjmatrix=m,mode="undirected",weighted=TRUE,diag=FALSE)

#typeof(g_line)
#V(g_line)
#train_nodes<-V(g_line)$name# To get node names
#write.matrix(train_nodes,file="test_nodes_ArnetMiner.txt") #To retrieve the names of all the vertices in the graph g
#ad_matrix_line_train<-get.adjacency(g_line,sparse=FALSE)
#x_train_line<-melt(ad_matrix_line_train)
#write.csv(x_train_line,file="Labeled_edge_list_test_years_line_330000_lcc.csv")


clusters(lcc)
plot(lcc)

#exracting node features in BUP_tain.attributes.csv
train_nodes_degree<-degree(lcc) #degree of each vertex
train_nodes_closeness<-closeness(lcc) #closeness of each vertex
train_nodes_betweenness<-betweenness(lcc)
write.csv(train_nodes_degree,file="test_nodes_degree.csv")
write.csv(train_nodes_closeness,file="test_nodes_closeness.csv")
write.csv(train_nodes_betweenness,file="test_nodes_betweenness.csv")


# Adamic-Adar
#aa<-proxfun(lcc, method="aa", value="edgelist")
# to write output jaccard to CSV file
#write.table(aa, file = "aa.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
#write.csv(aa,file="aa_values_training.csv")

aa2<-proxfun(lcc, method="aa", value="matrix")
aa2<-melt(aa2) # melt to create an edge list from the matrix jaccard, Please refer to:
zaa<-aa2$value
write.csv(zaa,file="aa_test.csv")



# Common Neighbours
#cn<-proxfun(lcc, method="cn", value="edgelist")
#write.table(cn, file = "cn.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
#write.csv(cn,file="cn_values_training.csv")

cn2<-proxfun(lcc, method="cn", value="matrix")
cn2<-melt(cn2) # melt to create an edge list from the matrix cn2
zcn<-cn2$value
write.csv(zcn,file="cn_test.csv")

# pa_preferential attachment
#pa<-proxfun(lcc, method="pa", value="edgelist")
# to write output pa to CSV file
#write.table(pa, file = "pa.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
#write.csv(pa,file="pa_train.txt")

pa2<-proxfun(lcc, method="pa", value="matrix")
pa2<-melt(pa2) # melt to create an edge list from the matrix pa, Please refer to:
zpa<-pa2$value
write.csv(zpa,file="pa_test.csv")


# katz
#katz<-proxfun(lcc, method="katz", value="edgelist")
# to write output katz to CSV file
#write.table(katz, file = "katz.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
#write.matrix(katz,file="katz_values_training.txt")

katz2<-proxfun(lcc, method="katz", value="matrix")
katz2<-melt(katz2) # melt to create an edge list from the matrix katz, Please refer to:
zkatz<-katz2$value
write.csv(zkatz,file="katz_test.csv")



#jaccard
#jaccard<-proxfun(lcc, method="jaccard", value="edgelist")
# to write output jaccard to CSV file
#write.table(jaccard, file = "jaccard_linkprediction_package.csv",row.names=FALSE, sep=",")
#write.matrix(jaccard,file="jaccard_values_training_edgelist_LP_Package.txt")

#jaccard2<-proxfun(lcc, method="jaccard", value="matrix")
#j<-melt(jaccard2) # melt to create an edge list from the matrix jaccard, Please refer to:
#zj<-j$value
#write.csv(zj,file="jaccard_values_training.csv")

#jaccard_igraph
j_igraph <- similarity.jaccard(lcc)
j_igraph <-melt(j_igraph ) # melt to create an edge list from the matrix jaccard, Please refer to:
zj_igraph <-j_igraph $value
write.csv(zj_igraph ,file="jaccard_test.csv")

#dice_igraph
d_igraph <- similarity.dice(lcc)
d_igraph <-melt(d_igraph ) # melt to create an edge list from the matrix jaccard, Please refer to:
zd_igraph <-d_igraph $value
write.csv(zd_igraph ,file="dice_test.csv")

#invlogweighted_igraph
invlogweighted<-similarity.invlogweighted(lcc)
i<-melt(invlogweighted) # melt to create an edge list from the matrix invlogweighted, Please refer to:
zi<-i$value
write.csv(zi,file="invlogweighted_test.csv")
