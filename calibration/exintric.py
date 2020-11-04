import cv2
import numpy as np
import argparse

class Camera:
    """
    Camera Class for camera calibration
    """

    def __init__(self, mtx = None, dist = None):
        """
        Camera Initilization
        mtx 
        dist
        """
        # intrisic parameters of the camera
        self.mtx = mtx
        self.dist = dist

        # exintric parameters of the camera
        self.rvec = None
        self.tvec = None

        # 设置世界坐标的坐标
        self.axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    
    def load_intrinsic(self, filename):
        """
        load intrinsic parameters of the camera
        filename: file that stores the intrinsic parameters of the camera
        
        """
        with np.load(filename) as f:
            self.mtx, self.dist = [f[i] for i in ('mtx','dist')]

    def calicalibration_photo(self, image, x_nums, y_nums, vis=False):
        """
        calibration photo
        Input
            image: image to be calibrated
            x_nums : corners along x axis
            y_nums : corners along y axis
        """
        # 设置(生成)标定图在世界坐标中的坐标
        # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
        world_point = np.zeros((x_nums * y_nums,3), np.float32)
        # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
        world_point[:,:2] = np.mgrid[:x_nums,:y_nums].T.reshape(-1, 2)

        # 设置角点查找限制
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    
        if len(image.shape) == 3 and image.shape[2] != 1 :
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # find corners
        ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), )
        text = "successfully find corners!" if ok else "Fail to find the given number of corners"
        print(text)

        if ok:
            #获取更精确的角点位置
            exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
            #获取外参
            _, self.rvec, self.tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)

            # visualize the results
            if vis:
                self.plot_image(image, corners)

    def plot_image(self, img, corners):
        """
        visualiz the calibration results
        """
        imgpts, jac = cv2.projectPoints(self.axis, self.rvec, self.tvec, self.mtx, self.dist)

        # visualize the corners
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
def test(photo_path, filename):
    """
    Test function
    """
    img = cv2.imread(photo_path)
    camera = Camera()
    camera.load_intrinsic(filename)
    camera.calicalibration_photo(img, 9, 3, vis=True)
    
if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--img", type=str, default="./test/test1.jpg", help="img for calibration")
    arg.add_argument("--intrinsic", type=str, default="intrinsic_parameters.npz", help="intrinsic parameters of camera")
    opt = arg.parse_args()


    test(opt.img, opt.intrinsic)



