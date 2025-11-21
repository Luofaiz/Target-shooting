import cv2
import rospy
import time
import numpy as np
from threading import Thread
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from ar_track_alvar_msgs.msg import AlvarMarkers
from collections import Counter


class Image_():
    def __init__(self, img) -> None:
        """
        :param img: 原始图像
        .crop_center : 从中心进行裁剪
        """

        self.data = img
        self.h, self.w = img.shape[:2]
        self.x = self.w // 2
        self.y = self.h // 2

    def __call__(self):  # 调用实例返回图像
        return self.data

    def crop_center(self, crop_width, crop_height):  # 从中心裁剪图像

        """
        :param crop_width: 裁剪宽度
        :param crop_height: 裁剪高度
        """

        start_x = max(0, self.x - crop_width // 2)
        start_y = max(0, self.y - crop_height // 2)
        end_x = min(self.w, self.x + crop_width // 2)
        end_y = min(self.h, self.y + crop_height // 2)
        return self.data[start_y:end_y, start_x:end_x]


class Camera():
    def __init__(self, source, P=[516.422974, 0.000000, 372.887246, 0.000000,
                                  0.000000, 520.513672, 226.444701, 0.000000,
                                  0.000000, 0.000000, 1.000000, 0.000000],
                 R=[1.000000, 0.000000, 0.000000,
                    0.000000, 1.000000, 0.000000,
                    0.000000, 0.000000, 1.000000],
                 K=[514.453865, 0.000000, 368.311232,
                    0.000000, 514.428903, 224.580904,
                    0.000000, 0.000000, 1.000000],
                 D=[0.031723, -0.045138, 0.004146, 0.006205, 0.000000]) -> None:

        """
        :param source: 视频流地址
        .open   :打开视频流线程
        .close  :关闭视频流线程
        .show   :展示视频内容
        .get    :获取视频内容
        """
        self.image_pub = rospy.Publisher("/usb_cam/image_raw", Image, queue_size=1)
        self.image_pub1 = rospy.Publisher("/usb_cam/camera_info", CameraInfo, queue_size=1)
        self.P = P
        self.R = R
        self.K = K
        self.D = D
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.frame = None
        self.img = None
        self.stop = False
        self.id = []
        self.pose = []
        self.sub = None
        self.center = []

    def quaternion_to_euler(self, q):

        q_w, q_x, q_y, q_z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
        cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q_w * q_y - q_z * q_x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q_w * q_z + q_x * q_y)
        cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def project_point(self, point):
        # 将三维点转换为齐次坐标
        point_h = np.append(point, 1)
        # 投影点

        projected_point_h = np.array(self.K).reshape(3, 3) @ np.array(self.R).reshape(3, 3) @ point_h[:3]

        # 齐次坐标归一化
        u = projected_point_h[0] / projected_point_h[2]
        v = projected_point_h[1] / projected_point_h[2]

        return np.array([u, v])

    def cv2_to_imgmsg(self, cv_image):
        B = Header()
        B.frame_id = "usb_cam"
        img_msg = Image()
        img_msg.header = B
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(
            img_msg.data) // img_msg.height  # That double line is actually integer division, not a comment
        return img_msg
   
    def circle_center(self):
        """圆形检测函数 - 不显示形态学窗口"""
        if self.frame is None:
            return None
        
        image = self.frame()
        kernel = np.ones((8, 8), np.uint8)
        
        # 优化后的颜色检测参数
        mask = cv2.inRange(image, np.array([104, 67, 0]), np.array([238, 89, 57]))
        
        # 形态学处理
        closing1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 检测区域处理
        # 裁剪模式 (640×480)
        opening = closing1[156:277, 245:357]
        opening = cv2.resize(opening, (640, 640))

        # 👇 注释掉这一行，不再显示形态学窗口
        # cv2.imshow("th", opening)
        
        # 中心点检测
        centers = []
        for i in range(191, 264):
            white = np.where(opening[i, :] == 255)[0]
            if len(white) == 0:
                continue
            left = white[0]
            right = white[-1]
            centers.append((left + right) / 2)
        
        if len(centers) == 0:
            return None
        
        from collections import Counter
        midpoint_counts = Counter(centers)
        mode_midpoint = midpoint_counts.most_common(1)[0][0]
        center = [mode_midpoint, 200]
        return center

    def show_with_circle_detection(self):
        """显示带圆形检测标记的摄像头画面（预览模式）"""
        if self.img is not None:
            img = self.img.copy()
            
            # 1. 绘制AR码（红点）
            if self.pose is not None:
                for i in self.pose:
                    cv2.circle(img, (int(i[0][0]), int(i[0][1])), 1, (0, 0, 255), 3)
                    # 添加AR码ID标签
                    cv2.putText(img, "AR", (int(i[0][0]) + 5, int(i[0][1]) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 2. 绘制裁剪区域（绿框）- 根据circle_center的参数
            crop_x1, crop_y1 = 245, 156
            crop_x2, crop_y2 = 357, 277
            cv2.rectangle(img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
            cv2.putText(img, "Detection Area", (crop_x1, crop_y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 3. 检测并绘制圆心
            center = self.circle_center()
            if center is not None:
                # 将检测坐标映射回原图
                crop_width = crop_x2 - crop_x1
                crop_height = crop_y2 - crop_y1
                real_x = int(crop_x1 + (center[0] / 640.0) * crop_width)
                real_y = int(crop_y1 + (center[1] / 640.0) * crop_height)
                
                # 绘制圆心（黄色大圆）
                cv2.circle(img, (real_x, real_y), 8, (0, 255, 255), 3)
                cv2.circle(img, (real_x, real_y), 2, (0, 255, 255), -1)  # 中心点
                cv2.putText(img, "Center: ({:.0f}, {:.0f})".format(center[0], center[1]), 
                           (real_x + 15, real_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 255), 2)
                
                # 显示检测成功状态
                cv2.putText(img, "Circle Detected!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 计算并显示与目标的偏差
                target_x = 350
                error = abs(center[0] - target_x)
                error_text = "Error: {:.1f}px".format(error)
                error_color = (0, 255, 0) if error <= 7 else (0, 165, 255)
                cv2.putText(img, error_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
            else:
                # 显示未检测到状态
                cv2.putText(img, "No Circle Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 4. 绘制目标线（蓝色竖线）- target_x=350的位置
            target_x = 350
            crop_width = crop_x2 - crop_x1
            target_real_x = int(crop_x1 + (target_x / 640.0) * crop_width)
            cv2.line(img, (target_real_x, crop_y1), (target_real_x, crop_y2), (255, 0, 0), 2)
            cv2.putText(img, "Target", (target_real_x + 5, crop_y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # 5. 添加图例说明
            legend_y = img.shape[0] - 100
            cv2.putText(img, "Legend:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(img, (20, legend_y + 20), 3, (0, 255, 0), 2)
            cv2.putText(img, "Detection Area", (30, legend_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.line(img, (15, legend_y + 40), (25, legend_y + 40), (255, 0, 0), 2)
            cv2.putText(img, "Target Line", (30, legend_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(img, (20, legend_y + 60), 3, (0, 255, 255), -1)
            cv2.putText(img, "Circle Center", (30, legend_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(img, (20, legend_y + 80), 3, (0, 0, 255), -1)
            cv2.putText(img, "AR Marker", (30, legend_y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("Camera Preview", img)
  
    def ar_marker_subscriber(self):
        def ar_marker_callback(msg):
            id = []
            pose = []
            for marker in msg.markers:
                id.append(marker.id)
                p = marker.pose.pose
                ss = self.project_point([p.position.x, p.position.y, p.position.z])
                # print(p.orientation.x)
                yaw = self.quaternion_to_euler([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z])[2]
                pose.append([ss, yaw])
            self.id = id
            self.pose = pose

        self.sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, ar_marker_callback, queue_size=7)
        rospy.spin()

    def thread(self, cap):
        while not self.stop:
            _, frame = cap.read()
            try:
                self.image_pub.publish(self.cv2_to_imgmsg(frame))
                msg = CameraInfo()
                A = Header()
                A.frame_id = "usb_cam"
                msg.header = A
                msg.height = 720
                msg.width = 1280
                msg.P = self.P
                msg.R = self.R
                msg.K = self.K
                msg.D = self.D
                self.image_pub1.publish(msg)
            except Exception as e:
                print(e)

            self.frame = Image_(frame)
            self.img = self.frame().copy()
            time.sleep(0.02)

    def open(self):  # 打开摄像头线程
        thread_cam = Thread(target=self.thread, args=(self.cap,))
        thread_ar = Thread(target=self.ar_marker_subscriber)
        thread_cam.start()
        thread_ar.start()

    def close(self):  # 关闭摄像头线程
        if self.sub is not None:
            self.sub.unregister()
        self.stop = True

    def show(self):  # 展示摄像头图像（用于task_2和task_3）
        if self.img is not None:
            img = self.img
            # print(self.pose)
            if self.pose is not None:
                for i in self.pose:
                    cv2.circle(img, (int(i[0][0]), int(i[0][1])), 1, (0, 0, 255), 3)
            if self.center is not None:
                # print("centers:",self.center)
                for i in self.center:
                    cv2.circle(img, (int(i[0]), int(i[1])), 1, (0, 255, 0), 3)
            cv2.imshow("raw", img)

    def get(self):  # 获取摄像头图像
        return self.frame

    def get_ar(self):
        return self.id, self.pose


if __name__ == "__main__":
    cam = Camera(0)
    cam.open()
    while 1:
        cam.show_with_circle_detection()  # 使用预览模式测试
        if cv2.waitKey(10) == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.close()
