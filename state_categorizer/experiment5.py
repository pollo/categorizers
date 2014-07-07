from experiment import ExperimentBase
from sys import argv
import os.path
import utm, numpy

"""
In this experiment the features extracted will be the 3 components and the module of velocity and accelleration of each point. In addition the classification of the features is used as a feature. Also the velocity over the whole window has been added.
"""

def distance_3d(lon1, lat1, h1, lon2, lat2, h2):
    ym1, xm1, _, _ = utm.from_latlon(lat1, lon1)
    ym2, xm2, _, _ = utm.from_latlon(lat2, lon2)

    arr1 = numpy.array((xm1, ym1, h1))
    arr2 = numpy.array((xm2, ym2, h2))

    dist = numpy.linalg.norm(arr1-arr2)
    return dist

class Experiment(ExperimentBase):
    @property
    def C(self):
        return 1

    @property
    def KERNEL_TYPE(self):
        return "linear"

    @property
    def FEATURES_PER_SAMPLE(self):
        return self.WINDOW_SIZE*8+self.WINDOW_SIZE/2+1

    @property
    def WINDOW_SIZE(self):
        return 11

    def _extract_features(self, points):
        features = []
        for i,point in enumerate(points):
            try:
                features.append(float(point['categorizers']['vel']))
                features.append(float(point['categorizers']['velx']))
                features.append(float(point['categorizers']['vely']))
                features.append(float(point['categorizers']['velz']))
                features.append(float(point['categorizers']['acc']))
                features.append(float(point['categorizers']['accx']))
                features.append(float(point['categorizers']['accy']))
                features.append(float(point['categorizers']['accz']))
                if i<len(points)/2:
                    features.append(int(point['classification']))
            except KeyError as e:
                print e
                return []

        distance = distance_3d(
            points[0]['x'], points[0]['y'], points[0]['z'],
            points[-1]['x'], points[-1]['y'], points[-1]['z']
        )
        time_delta = abs(points[-1]['ts'] - points[0]['ts'])
        speed = (distance) / time_delta
        features.append(speed)

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

