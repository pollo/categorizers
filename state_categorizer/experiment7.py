from experiment import ExperimentBase
from sys import argv
import os.path
import numpy as np
import utm

"""
In this experiment the features extracted will be the 3 components and the module of velocity and accelleration of each point
"""

def point2vector(point):
    x, y, _, _ = utm.from_latlon(point['lat'], point['lon'])

    vector = np.array(
        (x,
         y,
         point['z'])
    )
    return vector

class Experiment(ExperimentBase):
    @property
    def C(self):
        return 1

    @property
    def KERNEL_TYPE(self):
        return "linear"

    @property
    def FEATURES_PER_SAMPLE(self):
        return self.WINDOW_SIZE*8 + (self.WINDOW_SIZE-1)*12

    @property
    def WINDOW_SIZE(self):
        return 11

    def _extract_features(self, points):
        features = []

        assert self.WINDOW_SIZE%2==1

        central_time = points[self.WINDOW_SIZE/2]['ts']
        central_vector = point2vector(points[self.WINDOW_SIZE/2])

        for i,point in enumerate(points):
            try:
                if i != self.WINDOW_SIZE/2:
                    vector = point2vector(point)

                    delta_p = central_vector - vector
                    delta_t = central_time - point['ts']

                    velocity = delta_p / delta_t
                    acceleration = velocity / delta_t

                    features.append(np.linalg.norm(delta_p))
                    features.append(delta_p[0])
                    features.append(delta_p[1])
                    features.append(delta_p[2])

                    features.append(np.linalg.norm(velocity))
                    features.append(velocity[0])
                    features.append(velocity[1])
                    features.append(velocity[2])

                    features.append(np.linalg.norm(acceleration))
                    features.append(acceleration[0])
                    features.append(acceleration[1])
                    features.append(acceleration[2])

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

