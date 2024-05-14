import os
import pickle
from sklearn.mixture import GMM


def save_gmm(gmm, name):
    """ Save Gaussian mixture model using pickle.
        Args:
            gmm        : Gaussian mixture model.
            name (str) : File name.
    """
    filename = name + ".gmm"
    with open(filename, 'wb') as gmm_file:
        pickle.dump(gmm, gmm_file)
    print ("%5s %10s" % ("SAVING", filename,))

...
# get gender_voice_features using FeaturesExtraction
# generate gaussian mixture models
gender_gmm = GMM(n_components = 16, n_iter = 200, covariance_type = 'diag', n_init = 3)
# fit features to models
gender_gmm.fit(gender_voice_features)
# save gmm
save_gmm(gender_gmm, "gender")