import numpy as np
import cv2
import argparse
from monovideoodometery import MonoVideoOdometery

def main():
    parser = argparse.ArgumentParser(description='Visual odometry on image sequence.')
    parser.add_argument('--img_path', type=str, default='./data', help='Path to the image directory')
    args = parser.parse_args()

    img_path = args.img_path
    focal = 718.8560
    pp = (607.1928, 185.2157)
    lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    vo = MonoVideoOdometery(img_path, focal, pp, lk_params)
    traj = np.zeros((600, 800, 3), dtype=np.uint8)

    while vo.has_next_frame():
        vo.process_frame()
        coords = vo.get_coordinates()
        x, z = int(coords[0]) + 400, int(coords[2]) + 100

        traj = cv2.circle(traj, (x, z), 1, (0, 255, 0), 2)
        cv2.putText(traj, "Estimated Trajectory", (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Trajectory', traj)
        cv2.imshow('Frame', vo.current_frame)
        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    cv2.imwrite("./trajectory.png", traj)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
