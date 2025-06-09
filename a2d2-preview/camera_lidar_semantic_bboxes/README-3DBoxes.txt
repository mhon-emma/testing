## 3D Bounding Boxes

The 3D bounding box subset contains 12,497 frames. Each frame has the
following items:

- RGB image
- 3D point cloud
- annotated semantic segmentation label
- annotated 3D bounding box label

Frames are grouped in 18 different scenes, with each scene contained in
its own directory. Scene directory names are in 'YYYYMMDD_hhmmss'
format, denoting the time and date of the recording. Each scene
directory is further divided into four subdirectories:

- 'camera': images and json info files
- 'lidar': 3D point clouds
- 'label': annotated label images
- 'label3D': annotated 3D bounding boxes 

Each of these directories contains further subdirectories for each
camera. There are six cameras available in the vehicle, corresponding
to the following subdirectories:

- 'cam_front_center'
- 'cam_front_left'
- 'cam_front_right'
- 'cam_side_left'
- 'cam_side_right'
- 'cam_rear_center'

These are the filename formats for the item of a single frame:

RGB image           : YYMMDDDDhhmmss_camera_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].png
info                : YYMMDDDDhhmmss_camera_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].json
LiDAR point cloud   : YYMMDDDDhhmmss_lidar_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].npz
semantic label image: YYMMDDDDhhmmss_label_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].png
3D bounding boxes   : YYMMDDDDhhmmss_label3D_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].json

For example, a frame with ID 1617 from a scene recorded on 2018-08-07
14:50:28 from the front center camera would be have the following items
in these locations:

RGB image           : 20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000001617.png
info                : 20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000001617.json
LiDAR point cloud   : 20180807_145028/lidar/cam_front_center/20180807145028_lidar_frontcenter_000001617.npz
semantic label image: 20180807_145028/label/cam_front_center/20180807145028_label_frontcenter_000001617.png
3D bounding boxes   : 20180807_145028/label3D/cam_front_center/20180807145028_label3D_frontcenter_000001617.json

For further explanations regarding the format of each of these items,
please refer to the tutorial in our dataset web page.
