# Load Data
MtM_data = read.csv("first_train_data.csv",header = FALSE)
MtM_data_val = read.csv("first_val_data.csv",header = FALSE)

training.matrix =  as.matrix(MtM_data[,-1])
val_data = as.matrix(MtM_data_val[,-1])
## patches of 10 by 10 pixels
## Set up the autoencoder architecture:
nl=3 ## number of layers (default is 3: input, hidden, output)
unit.type = "logistic" ## specify the network unit type, i.e., the unit's
## activation function ("logistic" or "tanh")

N.input = 143 ## number of units (neurons) in the input layer (one unit per pixel)
N.hidden = 20 ## number of units in the hidden layer
lambda = 0.0002 ## weight decay parameter
beta = 6 ## weight of sparsity penalty term
rho = 0.001 ## desired sparsity parameter
epsilon <- 0.001 ## a small parameter for initialization of weights
## as small gaussian random numbers sampled from N(0,epsilon^2)
max.iterations = 100 ## number of iterations in optimizer
## Train the autoencoder on training.matrix using BFGS optimization method
## (see help('optim') for details):
## Not run:
autoencoder.object <- autoencode(X.train=training.matrix,nl=nl,X.test = val_data,N.hidden=c(8),
                                 unit.type=unit.type,lambda=lambda,beta=beta,rho=rho,epsilon=epsilon,
                                 optim.method="BFGS",max.iterations=max.iterations,
                                 rescale.flag=TRUE,rescaling.offset=0.001)

X.output <- predict(autoencoder.object, X.input=training.matrix, hidden.output=TRUE)
Output = as.data.frame(X.output$X.output)
cor(Output)
