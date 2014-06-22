from experiment import ExperimentBase

"""
In this experiment the features extracted will be the 3 components and the module of velocity and accelleration of each point
"""

class Experiment(ExperimentBase):
    @property
    def FEATURES_PER_POINT(self):
        return 8

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
    Experiment().run(auth_ids, 'experiment1/clf.dump')

