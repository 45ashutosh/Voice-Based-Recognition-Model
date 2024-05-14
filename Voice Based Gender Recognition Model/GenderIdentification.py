import pickle
import numpy as np
from FeaturesExtractor import FeaturesExtractor

def identify_gender(vector):
    # female hypothesis scoring
    is_female_scores         = np.array(self.females_gmm.score(vector))
    is_female_log_likelihood = is_female_scores.sum()

    # male hypothesis scoring
    is_male_scores         = np.array(self.males_gmm.score(vector))
    is_male_log_likelihood = is_male_scores.sum()

    # print scores
    print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
    print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood,3))))

    # find the winner aka the probable gender of the speaker
    if is_male_log_likelihood > is_female_log_likelihood: winner = "male"
    else                                                : winner = "female"
    return winner


# init instances and load models
features_extractor  = FeaturesExtractor()
females_gmm         = pickle.load(open(females_model_path, 'rb'))
males_gmm           = pickle.load(open(males_model_path, 'rb'))

# read the test directory and get the list of test audio files
file   = "speaker-test-file.wav"
vector = features_extractor.extract_features(file)
winner = identify_gender(vector)
expected_gender = file.split("/")[1][:-1]

print("%10s %6s %1s" %  ("+ EXPECTATION",":", expected_gender))
print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))