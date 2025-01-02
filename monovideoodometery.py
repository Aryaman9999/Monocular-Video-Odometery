import numpy as np
import cv2
import os

class MonoVideoOdometery:
    def __init__(self, img_file_path, focal_length=718.8560, pp=(607.1928, 185.2157),
                 lk_params=dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
                 detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        """
        Initialize the MonoVideoOdometery class.
        
        Args:
            img_file_path (str): Path to the directory containing image files.
            focal_length (float): Focal length of the camera.
            pp (tuple): Principal point (cx, cy) of the camera.
            lk_params (dict): Parameters for Lucas-Kanade optical flow.
            detector (cv2.Feature2D): OpenCV feature detector object.
        """
        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.eye(3)  # Start with identity matrix for rotation
        self.t = np.zeros((3, 1))  # Start with zero translation
        self.id = 0
        self.n_features = 0

        # Validate the image directory
        if not os.path.isdir(img_file_path):
            raise ValueError(f"The image path '{img_file_path}' does not exist.")

        # Collect and sort image files
        self.image_files = sorted([f for f in os.listdir(img_file_path) if f.endswith('.png')],
                                   key=lambda x: int(x.split('.')[0].replace("-", "")))
        if len(self.image_files) < 2:
            raise ValueError("The image directory must contain at least two images.")

        # Initialize the first frame
        self.process_frame()

    def has_next_frame(self):
        """
        Check if there are more frames to process.

        Returns:
            bool: True if more frames are available, False otherwise.
        """
        return self.id < len(self.image_files) - 1

    def detect(self, img):
        """
        Detect features in the input image.

        Args:
            img (np.ndarray): Grayscale image.

        Returns:
            np.ndarray: Array of detected feature points.
        """
        keypoints = self.detector.detect(img)
        return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    def visual_odometry(self):
        """
        Perform visual odometry by estimating motion between two frames.
        """
        # Detect new features if needed
        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        # Calculate optical flow
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        # Compute the Essential Matrix and recover pose
        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0)
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp)

        # Update translation and rotation with a fixed scale
        scale = 1.0  # Fixed scale factor
        self.t += scale * self.R.dot(t)
        self.R = R.dot(self.R)

        # Update the number of features
        self.n_features = self.good_new.shape[0]

    def process_frame(self):
        """
        Load and process the next frame in the sequence.
        """
        if self.id == 0:
            # Load the first two frames
            self.old_frame = cv2.imread(os.path.join(self.file_path, self.image_files[self.id]), cv2.IMREAD_GRAYSCALE)
            self.current_frame = cv2.imread(os.path.join(self.file_path, self.image_files[self.id + 1]), cv2.IMREAD_GRAYSCALE)
            self.visual_odometry()
            self.id += 1
        else:
            # Shift frames and load the next
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(os.path.join(self.file_path, self.image_files[self.id]), cv2.IMREAD_GRAYSCALE)
            self.visual_odometry()
            self.id += 1

    def get_coordinates(self):
        """
        Get the current estimated coordinates.

        Returns:
            np.ndarray: Translation vector [x, y, z].
        """
        return self.t.flatten()
