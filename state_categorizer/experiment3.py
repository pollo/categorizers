from experiment import ExperimentBase
from sys import argv
import os.path

"""
In this experiment the features extracted will be the 3 components and the module of velocity and accelleration of each point. The svm will be used with gaussian kernel instead of linear.
"""

class Experiment(ExperimentBase):
    @property
    def KERNEL_TYPE(self):
        return "rbf"

    @property
    def FEATURES_PER_SAMPLE(self):
        return self.WINDOW_SIZE*8

    @property
    def WINDOW_SIZE(self):
        return 11

    def _extract_features(self, points):
        features = []
        for point in points:
            try:
                features.append(float(point['categorizers']['vel']))
                features.append(float(point['categorizers']['velx']))
                features.append(float(point['categorizers']['vely']))
                features.append(float(point['categorizers']['velz']))
                features.append(float(point['categorizers']['acc']))
                features.append(float(point['categorizers']['accx']))
                features.append(float(point['categorizers']['accy']))
                features.append(float(point['categorizers']['accz']))
            except KeyError as e:
                print e
                return []
        return features

if __name__ == '__main__':
    auth_ids = [32, 51]
    experiment_name = os.path.splitext(os.path.basename(argv[0]))[0]
    try:
        subsampling = float(argv[1])
        experiment_name += "_s"+str(subsampling)
    except (ValueError, IndexError) as e:
        subsampling = 1.0
    Experiment().run(auth_ids, experiment_name,
                     subsampling)

