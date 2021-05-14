installed.packages("stringr")
installed.packages("text2vec")
library(stringr)
library(text2vec)
library(MASS)
library(reshape2) # To import melt function

# For measuring various distances/similarity between two documents (i.e., two reseach interests of two authors)
#http://text2vec.org/similarity.html
# Read author_dataset.csv into R
MyData <- read.csv(file="co_author_R_intersts_test.csv", header=TRUE, sep=",")

prep_fun = function(x) {
  x %>% 
    # make text lower case
    str_to_lower %>% 
    # remove non-alphanumeric symbols
    str_replace_all("[^[:alnum:]]", " ") %>% 
    # collapse multiple spaces
    str_replace_all("\\s+", " ")
}
MyData$R_intersts_clean = prep_fun(MyData$R_intersts )


#define common space to compare documents in vector space and project documents to it. We will use vocabulary-based vectorization vectorization for better interpretability:
it = itoken(MyData$R_intersts_clean, progressbar = FALSE)
v = create_vocabulary(it) %>% prune_vocabulary(doc_proportion_max = 0.1, term_count_min = 5)
vectorizer = vocab_vectorizer(v)

# they will be in the same space because we use same vectorizer
# hash_vectorizer will also work fine
dtm = create_dtm(it, vectorizer)
dim(dtm)


#compute cosine similarity 
cos_sim_RS= sim2(dtm, dtm, method = "cosine", norm = "l2")
typeof(cos_sim_RS )
write.matrix(cos_sim_RS ,file="cos_sim_RS_test.csv",sep=",")

# calculation of jaccard similarity 





