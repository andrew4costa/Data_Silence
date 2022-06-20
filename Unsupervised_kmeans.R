#This code allows the clustering of our dataset with the k-means unsupervised algorithm
library(tidyverse)
library(fastDummies)
library(dplyr)
library(cluster)
library(compareGroups)
library(NbClust)
library(Rtsne) # for t-SNE plot
library(ggplot2)

#useful tool------------------------------------------------------------------------------
extract_results <- function (dat,res){
  qual<-c("high", "low")
  typ<-c("white", "red")
  a<-1
  #loop for the combinations
  for(c in 1:2){
    for (t in 1:2){
      for(q in 1:2){
        res[a,c]<-length(which(dat$quality==qual[q]&dat$type==typ[t]&dat$cluster==c))
        a<-a+1
      }
      
    }
  a<-1}
  #loop for the totals
  a<-5
  for(c in 1:2){
    for (t in 1:2){
      res[a,c]<-length(which(dat$type==typ[t]&dat$cluster==c))
      a<-a+1}
    for(q in 1:2){
      res[a,c]<-length(which(dat$quality==qual[q]&dat$cluster==c))
      a<-a+1
    }
    a<-5}
  print(res)
  return(res)

}
#-------------------------------------------------------------------------------------

##Importing & Preprocessing####
rw = read.table("./winequality-red.csv",sep=",",header=T)
ww = read.table("./winequality-white.csv",sep=";",header=T)
rw['type'] <- c("red")
ww['type'] <- c("white")
wdb <- rbind(ww,rw)
wdb<-as.data.frame(wdb)
wdb <- wdb %>%
  mutate(
    quality = ifelse(quality<=5, "low", "high"),
    quality = factor(quality)
  ) %>%
  mutate(across(where(is.character), as.factor))
wdb <- wdb[!duplicated(wdb),]
std <- wdb %>% mutate_if(is.numeric, ~(scale(.) %>% as.vector))#k-means computes euclidean distance, so we standardize
#std %>% 
#  DataExplorer::create_report()

#How will 2-Means Classify our dataset?####--------------------------------------------------------------
library(ggpubr)
library(factoextra)

set.seed(42)
res.km <- kmeans(std[,-c(12,13)], 2, nstart = 150)
print(res.km)
std <- cbind(std, cluster = res.km$cluster)
std$cluster<-as.factor(std$cluster)
library(caret)
summary(std[,c(12,13,14)])

#extracting results-----------------------------------------------------------------------------------
r<-matrix(c(1:16),nrow=8)
r<-as.data.frame(r)
rownames(r)<-c("WH","WL","RH","RL","W","R","H","L")
r<-extract_results(std,r)
colnames(r)<-c("C1","C2")
kmeans_res<-r%>%
  mutate(tot = C1+C2)%>%
  mutate(pc_1 = round(C1*100/tot,2))%>%
  mutate(pc_2 = round(C2*100/tot,2))
kmeans_res

dat <- std%>%
  filter((cluster==1&type=="white")|(cluster==2&type=="red"))
highlight_df <- std%>%
  filter((cluster==2&type=="white")|(cluster==1&type=="red"))

##Plotting Results---------------------------------------------------------------------------------------
atop<- dat%>%
  ggplot(aes(x=type, y=quality))+ 
  scale_y_discrete(limits = c("low","high"))+
  geom_jitter(aes(color=cluster))+
  scale_color_manual(values=c("#DBDAB5","#94345D"))+
  geom_point(data=highlight_df,aes(x=type, y=quality), size=1.5, position="jitter", color="black") +
  ggtitle("K-means Classification of Wine","Based on Eleven Features Standardized") 

  
atop


#---------------------------------------------------------------

#PAM

# Buid distance matrix
pam_df <- std[1:500,-c(12,13)]  # Adjust number of dimensions for computing power reasons
df_dist <- as.matrix(dist(pam_df, method = "euclidean"))

# Find best optimal parameter using elbow and silhouette methods
fviz_nbclust(df_dist, pam, method = "wss") +
  geom_vline(xintercept = 8, linetype = 2)+
  labs(subtitle = "Elbow method")

fviz_nbclust(df_dist, pam, method = "silhouette")+
  labs(subtitle = "Silhouette method")

# Fit pam with 2 subgroups
pam_fit <- pam(df_dist,
               diss = TRUE,
               k = 8)


# Create clusters
tsne_obj <- Rtsne(df_dist, is_distance = TRUE)

tsne_pam_data <- tsne_obj$Y %>%
  data.frame() %>%
  setNames(c("X", "Y")) %>%
  mutate(cluster = factor(pam_fit$clustering))


# Plot clusters
ggplot(aes(x = X, y = Y), data = tsne_pam_data) +
  geom_point(aes(color = cluster))
