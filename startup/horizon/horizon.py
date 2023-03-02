import cv2
import numpy as np

from scipy.spatial.transform import Rotation

# Project world -> camera -> image
# X: [x, y, z] (3, N)
def project_points(X, R_mat, P):
    r, c = X.shape
    # Transform points from world coordinates to camera coordinate frame 
    Xr = R_mat @ X

    # Switch ONR coordinate system to opencv coordinate frame
    Xc = Xr[[1, 2, 0], :]
    Xc = np.vstack([Xc, np.ones((1, c))])

    # Project to image coordinates with camera projection matrix
    coords = P @ Xc
    
    # Homogenize points
    coords_xy = coords[0:2, :] / coords[2, :]
    return coords_xy


def estimate_horizon_mask(q_xyzw, Pn, return_points=False, shape=(512, 640)):
    N = 4
    y = 2000
    x = 5000
    z = 80

    # q = quaternion.as_quat_array(q_wxyz)
    # dummy_p = np.quaternion(0, 1, 0, 0)
    # res = np.sum(q*dummy_p)*q.conjugate()
    # theta_yaw = np.arctan2(res.y, res.x) # Yaw induced on dummy vector by quaternion
    # inv_R_yaw = quaternion.from_rotation_vector(np.array([0, 0, 1])*(-theta_yaw))
    # R_mat = quaternion.as_rotation_matrix(inv_R_yaw*q)
    # print(np.rad2deg(theta_yaw))

    r = Rotation.from_quat(q_xyzw)
    yaw, pitch, roll =  r.as_euler('ZYX', degrees=True)
    r = Rotation.from_euler('ZYX', [0, pitch, -roll], degrees=True)
    R_mat = r.as_matrix()

    X = np.ones((3, N))
    X[0, :] = x
    X[1, :] = np.linspace(-y, y, N)
    X[2, :] = z

    Xn = project_points(X, R_mat, Pn).T
    line_params = cv2.fitLine(Xn, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)

    (y_grid, x_grid) = np.indices(shape, sparse=True)
    sky_mask = y_grid - line_params[3, 0] >= line_params[1, 0] / line_params[0, 0] * (x_grid - line_params[2, 0])
    if return_points:
        return 255*sky_mask.astype(np.uint8), Xn.astype(int)
    
    return 255*sky_mask.astype(np.uint8)

def thermal2rgb(img):
    if img is not None:
        img = img - np.percentile(img, 1)
        img = img / np.percentile(img, 99)
        img = np.clip(img, 0, 1)
        img = np.stack([img]*3, axis=2)
        return np.uint8(255*img) 
    return np.zeros((512, 640, 3), dtype=np.uint8)   

def draw_overlay_and_pts(img, sky_mask, points, rotate_180):
    img = thermal2rgb(img)
    # for i in range(points.shape[0]):
    #     xi, yi = points[i]
    #     cv2.circle(img, (xi, yi), radius=3, color=(0, 0, 255), thickness=-1)

    # sky_mask = cv2.rotate(sky_mask, cv2.ROTATE_180)
    sky_mask_bool = sky_mask == 255
    sky_mask = np.stack([sky_mask]*3, axis=2)
    sky_mask[:,:,:2] = 0

    if rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    img_masked = cv2.addWeighted(img, 0.8, sky_mask, 0.1, 0)
    img[sky_mask_bool, :] = img_masked[sky_mask_bool, :]
    return img.astype(np.uint8)