#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## SKELETONS DEFINITIONS                                                 ##
###########################################################################

The definition and hierarchy of the following skeletons are available:
- OpenPose BODY_25B, BODY_25, BODY_135, COCO, MPII
- Mediapipe BLAZEPOSE
- AlphaPose HALPE_26, HALPE_68, HALPE_136, COCO_133, COCO, MPII
(for COCO and MPII, AlphaPose must be run with the flag "--format cmu")
- DeepLabCut CUSTOM: the skeleton will be defined in Config.toml

N.B.: Not all face and hand keypoints are reported in the skeleton architecture,
since some are redundant for the orientation of some bodies.

N.B.: The corresponding OpenSim model files are provided in the "Pose2Sim\Empty project" folder.
If you wish to use any other, you will need to adjust the markerset in the .osim model file,
as well as in the scaling and IK setup files.
'''

## INIT
from anytree import Node


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


'''BODY_25B (full-body without hands, experimental, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md'''
BODY_25B = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=17, children=[
        Node("Head", id=18, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])


'''BODY_25 (full-body without hands, standard, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
BODY_25 = Node("CHip", id=8, children=[
    Node("RHip", id=9, children=[
        Node("RKnee", id=10, children=[
            Node("RAnkle", id=11, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=12, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=14, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''BODY_135 (full-body with hands and face, experimental, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md)'''
BODY_135 = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=17, children=[
        Node("Head", id=18, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=48),
                    Node("RIndex", id=51),
                    Node("RPinky", id=63),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=27),
                    Node("LIndex", id=30),
                    Node("LPinky", id=42),
                ]),
            ]),
        ]),
    ]),
])


'''BLAZEPOSE (full-body with simplified hand and foot, from mediapipe)
https://google.github.io/mediapipe/solutions/pose'''
BLAZEPOSE = Node("root", id=None, children=[
    Node("right_hip", id=24, children=[
        Node("right_knee", id=26, children=[
            Node("right_ankle", id=28, children=[
                Node("right_heel", id=30),
                Node("right_foot_index", id=32),
            ]),
        ]),
    ]),
    Node("left_hip", id=23, children=[
        Node("left_knee", id=25, children=[
            Node("left_ankle", id=27, children=[
                Node("left_heel", id=29),
                Node("left_foot_index", id=31),
            ]),
        ]),
    ]),
    Node("nose", id=0, children=[
        Node("right_eye", id=5),
        Node("left_eye", id=2),
    ]),
    Node("right_shoulder", id=12, children=[
        Node("right_elbow", id=14, children=[
            Node("right_wrist", id=16, children=[
                Node("right_pinky", id=18),
                Node("right_index", id=20),
                Node("right_thumb", id=22),
            ]),
        ]),
    ]),
    Node("left_shoulder", id=11, children=[
        Node("left_elbow", id=13, children=[
            Node("left_wrist", id=15, children=[
                Node("left_pinky", id=17),
                Node("left_index", id=19),
                Node("left_thumb", id=21),
            ]),
        ]),
    ]),
])


'''HALPE_26 (full-body without hands, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
HALPE_26 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Head", id=17, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])


'''HALPE_68 (full-body with hands without face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
HALPE_68 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=49),
                    Node("RIndex", id=52),
                    Node("RPinky", id=64),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=28),
                    Node("LIndex", id=31),
                    Node("LPinky", id=43),
                ])
            ]),
        ]),
    ]),
])


'''HALPE_136 (full-body with hands and face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
HALPE_136 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=117),
                    Node("RIndex", id=120),
                    Node("RPinky", id=132),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=96),
                    Node("LIndex", id=99),
                    Node("LPinky", id=111),
                ])
            ]),
        ]),
    ]),
])


'''COCO_133 (full-body with hands and face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
COCO_133 = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=20, children=[
                    Node("RSmallToe", id=21),
                ]),
                Node("RHeel", id=22),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=17, children=[
                    Node("LSmallToe", id=18),
                ]),
                Node("LHeel", id=19),
            ]),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=0, children=[
            Node("right_eye", id=2),
            Node("left_eye", id=1),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=114),
                    Node("RIndex", id=117),
                    Node("RPinky", id=129),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=93),
                    Node("LIndex", id=96),
                    Node("LPinky", id=108),
                ])
            ]),
        ]),
    ]),
])


'''COCO (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, MMPose, etc.)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
COCO = Node("CHip", id=None, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=12, children=[
            Node("LAnkle", id=13),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''MPII (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, MMPose, etc.)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
MPII = Node("CHip", id=14, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=12, children=[
            Node("LAnkle", id=13),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''COCO_17 (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, MMPose, etc.)
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
COCO_17 = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])



"""Custom Skeletons by Johann Ullrich"""

'''Coco17_UpperBody

    COCO keypoints::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
        
https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html'''
Coco17_UpperBody = Node("Nose", id=0, children=[
    Node("LEye", id=1, children=[
        Node("LEar", id=3, children=[
            Node("LShoulder", id=5, children=[
                Node("LElbow", id=7, children=[
                    Node("LWrist", id=9),
                ]),
            ]),
            Node("LHip", id=11),
        ]),
    ]),
    Node("REye", id=2, children=[
        Node("REar", id=4, children=[
            Node("RShoulder", id=6, children=[
                Node("RElbow", id=8, children=[
                    Node("RWrist", id=10),
                ]),
            ]),
        Node("RHip", id=12),
        ]),
    ]),
])


'''B25_UPPER_BODY

Body25 only Upper Body

    B25_UPPER_BODY keypoints::

        0: 'Nose',
        1: 'Neck',
        2: 'RShoulder',
        3: 'RElbow',
        4: 'RWrist',
        5: 'LShoulder',
        6: 'LElbow',
        7: 'LWrist',
        8: 'CHip',
        9: 'RHip',
        12: 'LHip',
        15: 'REye',
        16: 'LEye'
'''
B25_UPPER_BODY = Node("CHip", id=8, children=[
    Node("RHip", id=9),
    Node("LHip", id=12),
    Node("Neck", id=1, children=[
        Node("Nose", id=0, children=[
            Node("REye", id=15),
            Node("LEye", id=16),
        ]),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])

'''B25_UPPER_EARS

Body25 only Upper Body with ears

    B25_UPPER_EARS keypoints::

        0: 'Nose',
        1: 'Neck',
        2: 'RShoulder',
        3: 'RElbow',
        4: 'RWrist',
        5: 'LShoulder',
        6: 'LElbow',
        7: 'LWrist',
        8: 'CHip',
        9: 'RHip',
        12: 'LHip',
        15: 'REye',
        16: 'LEye'
        17: 'REar'
        18: 'LEar'
'''
B25_UPPER_Ears = Node("CHip", id=8, children=[
    Node("RHip", id=9),
    Node("LHip", id=12),
    Node("Neck", id=1, children=[
        Node("Nose", id=0, children=[
            Node("REye", id=15, children=[
                Node("REar", id=17),
            ]),
            Node("LEye", id=16, children=[
                Node("LEar", id=18),
            ]),
        ]),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])