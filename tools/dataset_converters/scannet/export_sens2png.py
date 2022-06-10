import argparse
import os, sys

from SensorData import SensorData
from zipfile import ZipFile

#### python2.7 ####

# params
parser = argparse.ArgumentParser()
# data paths
# parser.add_argument('--filename', required=True, help='path to sens file to read')
# parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--root_path', required=True, help='root path of scannet')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)


def main():
  # if not os.path.exists(opt.output_path):
  #   os.makedirs(opt.output_path)
  # # load the data
  # sys.stdout.write('loading %s...' % opt.filename)
  # sd = SensorData(opt.filename)
  # sys.stdout.write('loaded!\n')
  # if opt.export_depth_images:
  #   sd.export_depth_images(os.path.join(opt.output_path, 'depth'))
  # if opt.export_color_images:
  #   sd.export_color_images(os.path.join(opt.output_path, 'color'))
  # if opt.export_poses:
  #   sd.export_poses(os.path.join(opt.output_path, 'pose'))
  # if opt.export_intrinsics:
  #   sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))
  scene_list = sorted(os.listdir(opt.root_path))
  for scene in scene_list:
    scene_path = os.path.join(opt.root_path, scene)

    # unzip
    zip_dir = os.path.join(scene_path, scene + "_2d-instance-filt")
    if not os.path.exists(zip_dir):
      zip_path = os.path.join(scene_path, scene + "_2d-instance-filt.zip")
      with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_dir)
    else:
      print "pass ", zip_dir

    zip_dir = os.path.join(scene_path, scene + "_2d-label-filt")
    if not os.path.exists(zip_dir):
      zip_path = os.path.join(scene_path, scene + "_2d-instance-filt.zip")
      with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_dir)
    else:
      print "pass ", zip_dir

    if os.path.exists(os.path.join(scene_path, 'intrinsic')):
      print "pass parsing ", scene_path, " sens"
      continue

    sens_path = os.path.join(opt.root_path, scene, scene + ".sens")
    sys.stdout.write('loading %s...' % scene)
    sd = SensorData(sens_path)
    sys.stdout.write('loaded!\n')
    if opt.export_depth_images:
      sd.export_depth_images(os.path.join(scene_path, 'depth'))
    if opt.export_color_images:
      sd.export_color_images(os.path.join(scene_path, 'color'))
    if opt.export_poses:
      sd.export_poses(os.path.join(scene_path, 'pose'))
    if opt.export_intrinsics:
      sd.export_intrinsics(os.path.join(scene_path, 'intrinsic'))

if __name__ == '__main__':
    main()