# Monocular Video Odometry Using OpenCV
This is an Python OpenCV based implementation of visual odometery. (Slightly modified from original from where I forked)
## Basicaly you don't need to provide any pose data (ground truth) to run the program and have added an example file, will update soon so it can take videos directly as inputs.

An invaluable resource I used in building the visual odometry system was Avi Singh's blog post: http://avisingh599.github.io/vision/monocular-vo/ as well as his C++ implementation found [here](https://github.com/avisingh599/mono-vo).

Datasets that can be used: [http://www.cvlibs.net/datasets/kitti/eval_odometry.php](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)



# Running Program
1. First clone repository
2. In `test.py` change `img_path` and `pose_path` to correct image sequences and pose file paths
3. Ensure focal length and principal point information is correct
4. Adjust Lucas Kanade Parameters as needed
5. Run command `python ./test.py`
