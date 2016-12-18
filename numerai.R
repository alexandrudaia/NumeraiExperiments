train=read.csv('/home/machine_learning/Downloads/numerai_datasets/numerai_training_data.csv',header=T,sep=',')
sums=c()
for( i  in 1:nrow(train))
{
  row=as.numeric(train[i,1:(ncol(train)-1)])
  s=sum(row^2)
  sums=c(sums,s)
}
train['sums']=sums

test=read.csv('/home/machine_learning/Downloads/numerai_datasets/numerai_tournament_data.csv',header=T,sep=',')
sums=c()
for( i  in 1:nrow(test))
{
  row=as.numeric(test[i,2:(ncol(test))])
  s=sum(row^2)
  sums=c(sums,s)
}
test['sums']=sums


#write.csv(train,'/home/machine_learning/Downloads/numerai_datasets/train1.csv',row.names = F)
#write.csv(test,'/home/machine_learning/Downloads/numerai_datasets/test1.csv',row.names = F)



#macking  things  categorical

for (i in 1:nrow(train)){
  train[i,1:(ncol(train)-2)]=round( as.numeric(train[i,1:(ncol(train)-2)])*10)
}
for (i in 1:nrow(test)){
  test[i,2:(ncol(test)-1)]=round( as.numeric(test[i,2:(ncol(test)-1)])*10)
}
######################computing energies 

kinetic=c()
for( i in  1:nrow(train))
{
  row=as.numeric(train[i,1:(ncol(train)-2)])
  probs=table(row)/(length(row))
  energy=sum(probs^2)
  kinetic=c(kinetic,energy)
}

train["kinetic"]=kinetic

kinetic=c()
for( i in  1:nrow(test))
{
  row=as.numeric(test[i,2:(ncol(test)-1)])
  probs=table(row)/(length(row))
  energy=sum(probs^2)
  kinetic=c(kinetic,energy)
}

test["kinetic"]=kinetic
write.csv(train,'/home/machine_learning/Downloads/numerai_datasets/trainCat.csv',row.names = FALSE)
write.csv(test,'/home/machine_learning/Downloads/numerai_datasets/testCat.csv',row.names = FALSE)

