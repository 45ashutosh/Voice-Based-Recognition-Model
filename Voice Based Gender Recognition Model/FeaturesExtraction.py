import numpy as np
 from sklearn import preprocessing
 from scipy.io.wavfile import read
 from python_speech_features import mfcc
 from python_speech_features import delta

 def extract_features(audio_path):
     """
     Extract MFCCs, their deltas and double deltas from an audio, performs CMS.

     Args:
         audio_path (str) : path to wave file without silent moments.
     Returns:
         (array) : Extracted features matrix.
     """
     rate, audio  = read(audio_path)
     mfcc_feature = mfcc(audio, rate, winlen = 0.05, winstep = 0.01, numcep = 5, nfilt = 30,
                         nfft = 512, appendEnergy = True)

     mfcc_feature  = preprocessing.scale(mfcc_feature)
     deltas        = delta(mfcc_feature, 2)
     double_deltas = delta(deltas, 2)
     combined      = np.hstack((mfcc_feature, deltas, double_deltas))
 return combined