from experiment import ExperimentBase
from sys import argv
import os.path
import numpy as np
import math

"""
In this experiment the features extracted will be the z components and the module of velocity and accelleration of each point, plus the steering ange of velocity and acceleration from one point to the next one
"""

def _angle_between(a,b):
  arccosInput = np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
  arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
  arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
  return math.acos(arccosInput)

class Experiment(ExperimentBase):
    @property
    def KERNEL_TYPE(self):
        return "linear"

    @property
    def FEATURES_PER_SAMPLE(self):
        return (self.WINDOW_SIZE-1)*6

    @property
    def WINDOW_SIZE(self):
        return 11

    def _extract_features(self, points):
        features = []
        for i in range(len(points)-1):
            try:
                vel_xy = np.array(
                  (float(points[i]['categorizers']['velx']),
                   float(points[i]['categorizers']['vely']))
                )
                acc_xy = np.array(
                  (float(points[i]['categorizers']['accx']),
                   float(points[i]['categorizers']['accy']))
                )
                next_vel_xy = np.array(
                  (float(points[i+1]['categorizers']['velx']),
                   float(points[i+1]['categorizers']['vely']))
                )
                next_acc_xy = np.array(
                  (float(points[i+1]['categorizers']['accx']),
                   float(points[i+1]['categorizers']['accy']))
                )
                vel_angle = _angle_between(vel_xy,
                                           next_vel_xy)
                acc_angle = _angle_between(acc_xy,
                                           next_acc_xy)
                features.append(float(points[i]['categorizers']['vel']))
                features.append(float(points[i]['categorizers']['velz']))
                features.append(float(points[i]['categorizers']['acc']))
                features.append(float(points[i]['categorizers']['accz']))
                features.append(vel_angle)
                features.append(acc_angle)
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

