from __future__ import absolute_import

import os

from got10k.experiments import *

import sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

from siamfc import TrackerSiamFC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.add_dll_directory("G:\\InstallSoftware\\Anaconda\\envs\\python_env\\Library\\bin\\geos_c.dll")

# if __name__ == '__main__':
#     net_path = '../不shuffle的VGG models/siamfc_alexnet_e50.pth'
#     tracker = TrackerSiamFC(net_path=net_path)
#

#     root_dir = os.path.expanduser('/home/liang/Downloads/data/OTB100')
#     results = '/home/liang/Downloads/siamfc-pytorch-master/results'
#     report = '/home/liang/Downloads/siamfc-pytorch-master/report'
#     e = ExperimentOTB(root_dir, version='tb100', result_dir=results, report_dir=report)
#     e.run(tracker, visualize=True)
#     e.report([tracker.name])
if __name__ == '__main__':
    # net_path = 'models-1-100/siamfc_alexnet_e50.pth'
    # tracker = TrackerSiamFC(net_path=net_path)
    # root_dir = os.path.expanduser('/home/asus/ly/SiamDUL/OTB100')
    # #e = ExperimentOTB(root_dir, version='tb100')
    # e = ExperimentOTB(root_dir, version=2015)
    # e.run(tracker, visualize=True)
    # e.report([tracker.name])

    for i in range(46,47):
        net_path = '/home/adam/Desktop/UniDualTrack/tools/models/siamfc_alexnet_e{}.pth'.format(i)
        tracker = TrackerSiamFC(net_path=net_path)
        root_dir = os.path.expanduser('/home/adam/Desktop/testdata/OTB100')
        #e = ExperimentOTB(root_dir, version='tb50'
        e = ExperimentOTB(root_dir, version=2015,result_dir='result-13/result-{}'.format(i),report_dir='result-13/report-{}'.format(i))
        #e = ExperimentDTB70(root_dir,result_dir='result-10-DTB/result-{}'.format(i),report_dir='result-10-DTB/report-{}'.format(i))
        #e = ExperimentUAV123(root_dir,version='UAV123',result_dir='result-10-uav/result-{}'.format(i),report_dir='result-10-uav/report-{}'.format(i))
        #e = ExperimentGOT10k(root_dir,subset = 'test',result_dir='result-10-got10k/result-{}'.format(i),report_dir='result-10-got10k/report-{}'.format(i))
        #e = ExperimentTColor128(root_dir,result_dir='result-10-color/result-{}'.format(i),report_dir='result-10-color/report-{}'.format(i))
        e.run(tracker, visualize=True)
        e.report([tracker.name])

