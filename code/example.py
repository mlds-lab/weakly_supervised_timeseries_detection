import numpy as np
from noisy_lr import NoiseyLogisticRegression

# Load data
data = np.load("../data/toy_data.npy")[0]
X = data["features"]
T = data["timestamps"]
Z = data["observations"]
Y = data["labels"]

# get data stats
n_sessions = len(X)
n_features = X[0].shape[1]

# Build model
nlr = NoiseyLogisticRegression(prior_model="logistic_regression",n_features=n_features,lambda_0=1.0,verbose=1,max_iter=1000,tol=1e-5)

# Fit model
nlr.fit(X,T,Z)

# Make prior predictions
# \hat{y} = \argmax_y p(y|x)
Y_hat_prior = nlr.predict(X)
print "Train set prior predictive accuracy:", np.mean(np.hstack(Y) == np.hstack(Y_hat_prior))

# Make posterior predictions
# \hat{o},\hat{y} = \argmax_{o,y} p(z,o,y|x,t)
O_hat_posterior,Y_hat_posterior,MAP_scores = zip(*nlr.map_inference(X,T,Z))
print "Train set posterior predictive accuracy:", np.mean(np.hstack(Y) == np.hstack(Y_hat_posterior))
