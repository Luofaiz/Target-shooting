#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
from playsound import playsound
import time
import os
from basecar.camera import Camera
from basecar.action import Action
import argparse
import tf
import threading
import rospy
import sys
import math
from sensor_msgs.msg import LaserScan
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import roslib
import tf2_ros
from robot_voice.msg import Targettt
from geometry_msgs.msg import TransformStamped
import numpy as np
from std_msgs.msg import String, Float32MultiArray
import ctypes
import cv2
from ctypes.util import find_library
import os

import serial

key = 0
F_list = []
robot_orientation_yaw, robot_pos_x, robot_pos_y = 0, 0, 0
angular_accumulated_error, angular_previous_error, angular_control_output = 0, 0, 0
angular_proportional_gain = -7
angular_integral_gain = 0
angular_derivative_gain = -3

global xy
global task_xy
global qtn_list_xy
global qtn_list_task_xy
global pose_num_xy
global pose_num_task_xy
global yaw
global now_pose_X
global now_pose_Y

xy = [[2.65, -0.2, 0], [2.97, -1.24, 0], [2.90, -2.48, -0.9], [0.2503, -2.66, -2.71], [0.125, -1.45, 2.5],
      [0.4, 0, -3.1]]
task_xy = [[1.25, 0.60, 180], [1.25, -0.60, 180], [-1.25, 0.60, 0], [1.25, 0.6, 0], [1.25, 0.60, 0], [1.25, 0.60, 0]]
global target_2
global target_3
# 在这里设置射击目标（请根据实际需要修改数字）
target_2 = 1  # 二号靶场目标（旋转靶）
target_3 = 7  # 三号靶场目标（移动靶）

target_position1 = [-1.350, 0.600, 180]
target_position12 = [-0.45, 1.3, 180]
target_position2 = [-1.342, -0.606, 180]
target_position23 = [-0.45, 2.6, 180]
target_position3 = [-1.338, -0.630, 180]
# ========================================================

qtn_list_xy = []
qtn_list_task_xy = []
pose_num_xy = len(xy)
pose_num_task_xy = len(task_xy)
yaw = 0
global move_base
now_pose_X = 0
now_pose_Y = 0

global w_kp
global w_ki
global w_kd
global w_target
global w_e_all
global w_last_e

w_kp = 2
w_ki = 0.001
w_kd = 0.005
w_e_all = 0
w_last_e = 0
global x_f, x_b, y_l, y_r
x_f = 0.0
x_b = 0.0
y_l = 0.0
y_r = 0.0


def w_pid_cal(pid_target, dis):
    global w_kp
    global w_ki
    global w_kd
    global w_e_all
    global w_last_e
    e = dis - pid_target
    w_e_all = w_e_all + e
    pid = w_kp * e + w_ki * w_e_all + w_kd * (e - w_last_e)
    w_last_e = e
    return pid


global p_kp
global p_ki
global p_kd
global p_e_all
global p_last_e
global p_pid
p_kp = -8
p_ki = 0
p_kd = -3
p_e_all = 0
p_last_e = 0
p_pid = 0


def p_pid_cal(pid_target, pose):
    global p_kp
    global p_ki
    global p_kd
    global p_e_all
    global p_last_e
    ture_pose = (pose / 3.14159265359 * 180.0 + 180.0) % 360
    if pid_target == 0:
        if ture_pose > 0 and ture_pose < 180:
            pid_target = 0
        if ture_pose > 180 and ture_pose < 360:
            pid_target = 360

    e = ture_pose - pid_target
    p_e_all = p_e_all + e
    pid = p_kp * e + p_ki * p_e_all + p_kd * (e - p_last_e)
    p_last_e = e
    return pid


global point_kp
global point_ki
global point_kd
global point_e_all
global point_last_e
global point_pid
point_kp = -3
point_ki = 0
point_kd = 0
point_e_all = 0
point_last_e = 0
point_pid = 0


def point_pid(pid_target_x, ture):
    global point_kp
    global point_ki
    global point_kd
    global point_e_all
    global point_last_e
    e = ture - pid_target_x
    point_e_all = point_e_all + e
    pid = point_kp * e + point_ki * point_e_all + point_kd * (e - point_last_e)
    point_last_e = e
    return pid


def limt(limt, target):
    if limt > target:
        limt = target
    if limt < -target:
        limt = -target
    return limt


def pid_stop2(target_x, target_y, target_yaw):
    global w_kp, w_ki, w_kd, w_e_all, y_l
    w_kp = 0.9
    w_ki = 0.000
    w_kd = 0.001
    w_e_all = 0
    count = 0
    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(7)
    speed = Twist()
    wich_x = 0
    wich_y = 0
    time = 0
    while not rospy.is_shutdown():
        rate_loop_pid.sleep()
        time += 1

        # 初始化PID值，避免target为0时未定义
        pid_x = 0
        pid_y = 0

        if target_x > 0:
            pid_x = w_pid_cal(target_x, x_f)
            wich_x = x_f
        elif target_x < 0:
            pid_x = w_pid_cal(target_x, -x_b)
            wich_x = -x_b
        # else: target_x == 0, pid_x保持为0，wich_x也保持为0

        if target_y > 0:
            pid_y = w_pid_cal(target_y, y_l)
            wich_y = y_l
        elif target_y < 0:
            pid_y = w_pid_cal(target_y, -y_r)
            wich_y = -y_r
        # else: target_y == 0, pid_y保持为0，wich_y也保持为0

        p_pid = p_pid_cal(target_yaw, yaw)
        speed.linear.y = pid_y
        speed.linear.x = pid_x
        speed.angular.z = p_pid / 180.0 * 3.14159265359
        w_e_all = limt(w_e_all, 5)
        if abs(wich_x - target_x) <= 0.03 and abs(wich_y - target_y) <= 0.03 and abs(
                target_yaw - (yaw / 3.1415926 * 180 + 180)) <= 3:
            w_e_all = 0
            count += 1
        if count >= 6:
            speed.linear.x = 0
            speed.linear.y = 0
            speed.linear.z = 0
            pid_vel_pub.publish(speed)
            w_e_all = 0
            break
        pid_vel_pub.publish(speed)


def pid_stop(target_x, target_y, target_yaw):
    global w_kp, w_ki, w_kd, w_e_all, y_l
    w_kp = 1.8
    w_ki = 0.00
    w_kd = 0.01
    w_e_all = 0
    count = 0
    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(7)
    speed = Twist()
    wich_x = 0
    wich_y = 0
    time = 0
    while not rospy.is_shutdown():
        rate_loop_pid.sleep()
        time += 1

        # 初始化PID值，避免target为0时未定义
        pid_x = 0
        pid_y = 0

        if target_x > 0:
            pid_x = w_pid_cal(target_x, x_f)
            wich_x = x_f
        elif target_x < 0:
            pid_x = w_pid_cal(target_x, -x_b)
            wich_x = -x_b

        if target_y > 0:
            pid_y = w_pid_cal(target_y, y_l)
            wich_y = y_l
        elif target_y < 0:
            pid_y = w_pid_cal(target_y, -y_r)
            wich_y = -y_r

        p_pid = p_pid_cal(target_yaw, yaw)
        if abs(wich_x) > 0.6:
            speed.linear.y = 0.05 * pid_y
        else:
            speed.linear.y = pid_y

        speed.linear.x = pid_x
        speed.angular.z = p_pid / 180.0 * 3.14159265359
        w_e_all = limt(w_e_all, 5)
        print("w_e_all:", w_e_all)
        print("wich_x:", wich_x, "wich_y:", wich_y)
        if abs(wich_x - target_x) <= 0.2 and abs(wich_y - target_y) <= 0.25:
            w_e_all = 0
            count += 1
        if count >= 3:
            speed.linear.x = 0
            speed.linear.y = 0
            speed.linear.z = 0
            pid_vel_pub.publish(speed)
            w_e_all = 0
            break
        pid_vel_pub.publish(speed)


def pid_bask(target_x, target_y, target_yaw):
    global w_kp, w_ki, w_kd, w_e_all, y_l
    w_kp = 0.6
    w_ki = 0.00
    w_kd = 0.01
    w_e_all = 0
    count = 0
    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(10)
    speed = Twist()
    wich_x = 0
    wich_y = 0
    time = 0
    while not rospy.is_shutdown():
        rate_loop_pid.sleep()
        time += 1

        # 初始化PID值，避免target为0时未定义
        pid_x = 0
        pid_y = 0

        if target_x > 0:
            pid_x = w_pid_cal(target_x, x_f)
            wich_x = x_f
        elif target_x < 0:
            pid_x = w_pid_cal(target_x, -x_b)
            wich_x = -x_b

        if target_y > 0:
            pid_y = w_pid_cal(target_y, y_l)
            wich_y = y_l
        elif target_y < 0:
            pid_y = w_pid_cal(target_y, -y_r)
            wich_y = -y_r

        p_pid = p_pid_cal(target_yaw, yaw)
        speed.linear.y = pid_y
        speed.linear.x = pid_x
        speed.angular.z = p_pid / 180.0 * 3.14159265359
        w_e_all = limt(w_e_all, 5)
        print("w_e_all:", w_e_all)
        print("wich_x:", wich_x, "wich_y:", wich_y)
        if abs(wich_x - target_x) <= 0.08 and abs(wich_y - target_y) <= 0.08 and abs(
                target_yaw - (yaw / 3.1415926 * 180 + 180)) <= 3:
            w_e_all = 0
            count += 1
        if count >= 6:
            speed.linear.x = 0
            speed.linear.y = 0
            speed.linear.z = 0
            pid_vel_pub.publish(speed)
            w_e_all = 0
            break
        pid_vel_pub.publish(speed)


def pid_go(target_x, target_y, target_yaw):
    global point_kp, vision_result, dis_trun_off, point_ki
    global w_kp, w_ki, w_kd
    w_kp = 2
    w_ki = 0
    w_kd = 0.008
    count = 0
    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(7)
    speed = Twist()
    wich_x = 0
    wich_y = 0
    while not rospy.is_shutdown():
        if target_x > 0:
            pid_x = w_pid_cal(target_x, x_f)
            wich_x = x_f
        if target_x < 0:
            pid_x = w_pid_cal(target_x, -x_b)
            wich_x = -x_b
        if target_y > 0:
            pid_y = w_pid_cal(target_y, y_l)
            wich_y = y_l
        if target_y < 0:
            pid_y = w_pid_cal(target_y, -y_r)
            wich_y = -y_r
        p_pid = p_pid_cal(target_yaw, yaw)
        if abs(target_x - wich_x) < 0.2 and abs(target_y - wich_y) < 0.2:
            speed.linear.x = 0
            speed.linear.y = 0
        else:
            speed.linear.y = pid_y
            speed.linear.x = pid_x
        speed.angular.z = p_pid / 180.0 * 3.14159265359
        pid_vel_pub.publish(speed)
        rate_loop_pid.sleep()
        if vision_result != 0:
            break


def pid_turn(target_x, target_y, target_yaw):
    global point_kp
    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(7.2)
    speed = Twist()
    while not rospy.is_shutdown():
        map_pid_x = point_pid(target_x, now_pose_X)
        map_pid_y = point_pid(target_y, now_pose_Y)
        p_pid = p_pid_cal(target_yaw, yaw)
        if target_yaw != 180:
            speed.linear.x = 0
            speed.linear.y = 0
        else:
            speed.linear.x = 0
            speed.linear.y = 0
        speed.angular.z = 1.5 * p_pid / 180.0 * 3.14159265359
        pid_vel_pub.publish(speed)
        rate_loop_pid.sleep()
        if abs(target_yaw - (yaw / 3.1415926 * 180 + 180)) <= 50:
            break


serial_port = serial.Serial("/dev/shoot", 9600)
sleep_time = 1


def shoot_action():
    buf = bytearray([0x55, 0x01, 0x12, 0x00, 0x00, 0x00, 0x01, 0x69])
    serial_port.write(buf)
    rospy.sleep(0.08)
    buf = bytearray([0x55, 0x01, 0x11, 0x00, 0x00, 0x00, 0x01, 0x68])
    serial_port.write(buf)


def vision_callback(msg):
    a = 1


def vision_listen():
    rospy.Subscriber('/vision_detect', Float32MultiArray, vision_callback, queue_size=10)
    rospy.spin()


global scan_data
scan_data = []


def get_valid_distance(scan, start_index, direction, angle_resolution):
    max_angle = len(scan.ranges)
    for i in range(max_angle):
        index = (start_index + i * direction) % max_angle
        if scan.ranges[index] != float('inf'):
            distance = scan.ranges[index]
            angle = np.radians((index - start_index) * angle_resolution)
            distance_corrected = distance * np.cos(angle)
            return distance_corrected
    return float('inf')


def get_laserscan(scan):
    global x_f, x_b, y_l, y_r, yaw, scan_data
    scan_data = scan.ranges

    front_index = 360
    angle_resolution = 0.5
    x_f = get_valid_distance(scan, front_index, 1, angle_resolution)
    x_b = get_valid_distance(scan, 0, 1, angle_resolution)
    y_l = get_valid_distance(scan, 540, -1, angle_resolution)
    y_r = get_valid_distance(scan, 180, 1, angle_resolution)


def laser_listen():
    rospy.Subscriber('/scan_filtered', LaserScan, get_laserscan, queue_size=7)
    rospy.spin()


# ========== 修改2：同步更新robot_orientation_yaw ==========
def now_pose_xy():
    global now_pose_X, now_pose_Y, yaw, robot_orientation_yaw  # 添加robot_orientation_yaw
    now_pose = rospy.Rate(10)
    listener = tf.TransformListener()
    while not rospy.is_shutdown():
        now_pose.sleep()
        try:
            (trans, rot) = listener.lookupTransform("map", "base_link", rospy.Time(0))
            # 小车坐标
            now_pose_X = trans[0]
            now_pose_Y = trans[1]
            euler = tf.transformations.euler_from_quaternion(rot)
            yaw = euler[2]  # 第三个元素是yaw角
            robot_orientation_yaw = euler[2]  # 同步更新robot_orientation_yaw
        except Exception as e:
            print("连接tf中.......")


# ==========================================================


def play_mp3(file_path):
    """使用playsound播放mp3文件"""
    try:
        if os.path.exists(file_path):
            # print(f"正在播放: {file_path}")
            playsound(file_path)
            # print(f"播放完成: {file_path}")
        else:
            print(f"❌ 文件不存在: {file_path}")
    except Exception as e:
        print(f"播放音频出错: {e}")

def shoot_task_1(target_point, kp, ki, kd):
    """
    改进版射击函数 - 每次射击前重新瞄准，确保精度
    """
    global w_kp, w_ki, w_kd
    w_kp = kp
    w_ki = ki
    w_kd = kd

    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(10)
    speed = Twist()

    max_shots = 2  # 最大射击次数
    shot_count = 0

    print(f"开始精确射击任务，目标点: {target_point}")

    # 初始检测状态
    initial_center = cam.circle_center()
    if initial_center is not None:
        initial_error = initial_center[0] - target_point
        print(f"初始圆心: X={initial_center[0]:.1f} Y={initial_center[1]:.1f} | 初始偏差: {initial_error:+.1f} 像素")
    else:
        print("初始状态：未检测到圆心，开始搜索...")

    while not rospy.is_shutdown() and shot_count < max_shots:
        # 每次射击前重新瞄准
        w_e_all = 0  # 重置PID积分项
        count = 0

        print(f"=== 第 {shot_count + 1} 次射击瞄准 ===")

        # 瞄准阶段
        while not rospy.is_shutdown():
            # 👇 添加这行：在瞄准过程中持续显示预览窗口
            cam.show_with_circle_detection()
            cv2.waitKey(1)

            center = cam.circle_center()
            if center == None:
                print("未检测到圆心，等待检测中...")
                continue

            # 实时显示圆心坐标和调整信息
            current_error = center[0] - target_point
            shoot_turn = w_pid_cal(target_point, center[0])

            print(
                f"圆心坐标: X={center[0]:.1f} Y={center[1]:.1f} | 目标: {target_point} | 误差: {current_error:+.1f} | 转速: {shoot_turn * 0.5:.4f}")

            speed.linear.x = 0
            speed.linear.y = 0
            speed.angular.z = shoot_turn * 0.5
            pid_vel_pub.publish(speed)

            if abs(target_point - center[0]) <= 7:
                count += 1
                print(f" 精度达标 (稳定计数: {count}/8)")
            else:
                count = 0
                print(f"继续调整...")

            if count >= 5:
                print(f"瞄准完成！最终圆心: X={center[0]:.1f} Y={center[1]:.1f}")
                break

        speed.linear.x = 0
        speed.linear.y = 0
        speed.angular.z = 0
        pid_vel_pub.publish(speed)
        rospy.sleep(0.3)

        center = cam.circle_center()
        if center is not None:
            current_error = center[0] - target_point
            print(f"微调前圆心: X={center[0]:.1f} Y={center[1]:.1f} | 偏差: {current_error:+.1f} 像素")

            if abs(current_error) <= 2:
                correction_angle = 0.25
                correction_time = 0.15
                print("  微调级别: 精确对准")
            elif current_error > 2:
                correction_angle = 0.3
                correction_time = 0.18
                print("  微调级别: 圆心偏右，增加左转")
            else:
                correction_angle = 0.2
                correction_time = 0.12
                print("  微调级别: 圆心偏左，减少左转")

            print(f"  微调参数: 角速度={correction_angle}, 时间={correction_time}s")

            speed.angular.z = correction_angle
            pid_vel_pub.publish(speed)
            rospy.sleep(correction_time)

            # 停止并最终稳定
            speed.angular.z = 0
            pid_vel_pub.publish(speed)
            rospy.sleep(0.2)

            # 检查微调后的结果
            final_center = cam.circle_center()
            if final_center is not None:
                final_error = final_center[0] - target_point
                print(
                    f"微调后圆心: X={final_center[0]:.1f} Y={final_center[1]:.1f} | 最终偏差: {final_error:+.1f} 像素")
            else:
                print("微调后未检测到圆心")
        else:
            print("微调阶段未检测到圆心，直接射击")

        # 射击
        print(f"\n🎯 第 {shot_count + 1} 次射击准备完毕！")
        # 射击前最后确认圆心位置
        pre_shoot_center = cam.circle_center()
        if pre_shoot_center is not None:
            pre_shoot_error = pre_shoot_center[0] - target_point
            print(
                f"射击时圆心: X={pre_shoot_center[0]:.1f} Y={pre_shoot_center[1]:.1f} | 射击偏差: {pre_shoot_error:+.1f} 像素")
        else:
            print("射击时未检测到圆心")

        action.shoot(1)
        shot_count += 1
        print(f"💥 第 {shot_count} 次射击完成！")

        # 射击后稳定恢复时间
        if shot_count < max_shots:
            print(f"等待射击反冲稳定... ({1.5}秒)")
            rospy.sleep(1)  # 增加射击间隔，让机器人完全稳定

    print(f"\n=== 射击任务完成 ===")
    print(f"总射击次数: {shot_count}")

    # 最终状态检查
    final_center = cam.circle_center()
    if final_center is not None:
        final_error = final_center[0] - target_point
        print(f"最终圆心: X={final_center[0]:.1f} Y={final_center[1]:.1f} | 最终偏差: {final_error:+.1f} 像素")
    else:
        print("任务结束时未检测到圆心")


def debug_circle_center(target_point):
    """
    调试圆形靶射击的函数 - 用于shoot_task_1
    显示圆心坐标，支持手动射击测试和机器人微调

    参数:
    - target_point: 目标圆心X坐标（通常是333）
    """
    print("=== 进入圆形靶调试模式 ===")
    print("操作说明：")
    print("- 将圆形靶放在视野中")
    print("- 按 's' 键射击测试")
    print("- 按 'a' 键向左微调角度")
    print("- 按 'd' 键向右微调角度")
    print("- 按 'w' 键向前微调位置")
    print("- 按 'x' 键向后微调位置")
    print("- 按 'q' 键退出调试")
    print("- ESC键退出")
    print()
    print("圆形靶射击参考信息：")
    print(f"- 目标X坐标: {target_point}")
    print("- shoot_task_1 射击范围: ±7像素")
    print("- 图像尺寸: 640x480")
    print()

    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    speed = Twist()

    # 停止机器人
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    print("开始检测圆形靶...")

    while not rospy.is_shutdown():
        # 显示摄像头画面
        cam.show()
        key = cv2.waitKey(1) & 0xFF

        # 获取圆心坐标
        center = cam.circle_center()

        if center is not None:
            x_coord = center[0]
            y_coord = center[1]

            # 计算与目标点的偏差
            x_error = x_coord - target_point

            # 判断是否在射击范围内
            in_shoot_range = abs(x_error) <= 7
            shoot_status = "✅" if in_shoot_range else "❌"

            # 判断偏向
            if x_error > 0:
                direction = "向右偏"
            elif x_error < 0:
                direction = "向左偏"
            else:
                direction = "居中"

            # 实时显示圆心坐标信息
            print(f"\r圆心坐标: X={x_coord:.1f} Y={y_coord:.1f}", end="")
            print(f" | 目标: {target_point}", end="")
            print(f" | 偏差: {x_error:+.1f} ({direction})", end="")
            print(f" | 射击状态: {shoot_status}", end="")
            print(f" | 范围: ±7      ", end="")

        else:
            print(f"\r未检测到圆形靶 - 等待检测中...                                                     ", end="")

        # 键盘控制
        if key == ord('s') or key == ord('S'):
            if center is not None:
                print(f"\n>>> 射击测试！圆心坐标: ({center[0]:.1f}, {center[1]:.1f})")
            else:
                print("\n>>> 射击测试！(未检测到圆形靶，但仍可射击)")
            action.shoot(1)  # shoot_task_1使用action.shoot(1)
            rospy.sleep(1)

        elif key == ord('a') or key == ord('A'):
            print("\n>>> 向左微调角度")
            speed.angular.z = 0.1  # 向左转
            pid_vel_pub.publish(speed)
            rospy.sleep(0.2)
            speed.angular.z = 0
            pid_vel_pub.publish(speed)

        elif key == ord('d') or key == ord('D'):
            print("\n>>> 向右微调角度")
            speed.angular.z = -0.1  # 向右转
            pid_vel_pub.publish(speed)
            rospy.sleep(0.2)
            speed.angular.z = 0
            pid_vel_pub.publish(speed)

        elif key == ord('w') or key == ord('W'):
            print("\n>>> 向前微调位置")
            speed.linear.x = 0.1  # 向前
            pid_vel_pub.publish(speed)
            rospy.sleep(0.3)
            speed.linear.x = 0
            pid_vel_pub.publish(speed)

        elif key == ord('x') or key == ord('X'):
            print("\n>>> 向后微调位置")
            speed.linear.x = -0.1  # 向后
            pid_vel_pub.publish(speed)
            rospy.sleep(0.3)
            speed.linear.x = 0
            pid_vel_pub.publish(speed)

        elif key == ord('q') or key == ord('Q') or key == 27:  # 27是ESC键
            print("\n>>> 退出调试模式")
            break

    # 确保机器人停止
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    print("\n=== 圆形靶调试完成 ===")
    print("调试结果分析：")
    if center is not None:
        final_x = center[0]
        final_y = center[1]
        final_error = final_x - target_point
        print(f"最终圆心坐标: X={final_x:.1f}, Y={final_y:.1f}")
        print(f"与目标点偏差: {final_error:+.1f} 像素")
        print(f"是否在射击范围: {'是' if abs(final_error) <= 7 else '否'}")
        if abs(final_error) > 7:
            if final_error > 0:
                print("建议: 机器人需要向左调整")
            else:
                print("建议: 机器人需要向右调整")
        else:
            print("建议: 当前位置适合射击")
    else:
        print("未检测到圆形靶，请检查：")
        print("1. 圆形靶是否在视野范围内")
        print("2. 光照条件是否合适")
        print("3. 圆形靶颜色是否符合检测要求")
    print("请记录击中目标时的圆心坐标用于参数调整！")


def task_1(target_point, kp, ki, kd):
    pid_stop2(target_position1[0], target_position1[1], 180)
    print("______________________________到达目标点1______________________________")
    # debug_camera_center(1)
    # debug_circle_center(345)
    # action.shoot(1)
    shoot_task_1(target_point, kp, ki, kd)
    print("任务一结束")
    print("✓ 已关闭圆形检测预览窗口")


def shoot_task_2(task_id):
    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(15)
    speed = Twist()

    # 射击控制参数
    shoot_count = 0  # 已射击次数
    max_shots = 3  # 最大射击次数

    # 检测失败退出机制
    no_detection_count = 0  # 连续未检测到目标的次数
    max_no_detection = 10  # 最大连续未检测次数

    print(f"开始旋转靶射击任务，最大射击次数: {max_shots}")
    print(f"连续未检测到目标{max_no_detection}次将自动退出")

    while not rospy.is_shutdown():
        # 退出条件1：达到最大射击次数
        if shoot_count >= max_shots:
            print(f"已达到最大射击次数({max_shots}次)，任务完成")
            break

        # 退出条件2：连续未检测到目标达到上限
        if no_detection_count >= max_no_detection:
            print(f"连续{max_no_detection}次未检测到目标{task_id}，任务退出")
            break

        cam.show()  # 👈 显示实时摄像头画面和AR码红点
        cv2.waitKey(1)
        ids, poses = cam.get_ar()
        print(
            f"检测到: {ids}, 目标: {task_id}, 射击进度: {shoot_count}/{max_shots}, 未检测计数: {no_detection_count}/{max_no_detection}")

        if task_id in ids:
            # 检测到目标，重置未检测计数器
            no_detection_count = 0

            pose = poses[ids.index(task_id)]
            error = pose[0][0] - 301

            print(f"AR码坐标: X={pose[0][0]:.1f}, Y={pose[0][1]:.1f}, 误差: {error:.1f}")

            # 控制机器人转向
            shoot_turn = -error * 0.006
            speed.linear.x = 0
            speed.linear.y = 0
            speed.angular.z = shoot_turn
            pid_vel_pub.publish(speed)

            # 射击条件判断
            if abs(error) < 10 and pose[0][1] > 173 and pose[0][1] < 185:
                print(f"瞄准成功，第{shoot_count + 1}次射击")
                action.shoot(1)
                # rospy.sleep(0.05)
                # action.shoot(1)
                shoot_count += 1

                # 射击后短暂停顿
                rospy.sleep(0.5)
        else:
            # 未检测到目标，增加未检测计数器
            no_detection_count += 1
            print(f"未检测到目标{task_id} (连续未检测: {no_detection_count}/{max_no_detection})")

            # 停止机器人转动，等待目标出现
            speed.linear.x = 0
            speed.linear.y = 0
            speed.angular.z = 0
            pid_vel_pub.publish(speed)

        rate_loop_pid.sleep()

    # 停止机器人
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    # 任务结束统计
    if shoot_count >= max_shots:
        print(f"旋转靶任务完成，总射击次数: {shoot_count} (达到最大次数)")
    elif no_detection_count >= max_no_detection:
        print(f"旋转靶任务提前结束，总射击次数: {shoot_count} (连续未检测)")
    else:
        print(f"旋转靶任务结束，总射击次数: {shoot_count}")


def debug_camera_center(task_id):
    """
    调试相机视野中心的函数 - 增强版，显示完整的X和Y坐标信息
    使用方法：
    1. 将AR码放在炮口正前方（机器人能击中的位置）
    2. 运行此函数，观察AR码坐标
    3. 按 's' 射击测试，看是否击中
    4. 按 'a'/'d' 微调机器人角度
    5. 按 'w'/'s' 微调机器人前后位置（如果需要）
    6. 按 'q' 退出调试
    """
    print("=== 进入视野中心标定模式（增强版）===")
    print("操作说明：")
    print("- 将AR码放在炮口正前方")
    print("- 按 's' 键射击测试")
    print("- 按 'a' 键向左微调角度")
    print("- 按 'd' 键向右微调角度")
    print("- 按 'w' 键向前微调位置")
    print("- 按 'x' 键向后微调位置")
    print("- 按 'q' 键退出调试")
    print("- ESC键退出")
    print()
    print("参考信息：")
    print("- 理论视野中心: X=320, Y=240")
    print("- 当前使用中心: X=300")
    print("- task_2 Y参考值: 208±20 (范围188-228)")
    print("- 图像尺寸: 640x480")
    print()

    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    speed = Twist()

    # 停止机器人
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    while not rospy.is_shutdown():
        # 显示摄像头画面
        cam.show()
        key = cv2.waitKey(1) & 0xFF

        # 获取AR码信息
        ids, poses = cam.get_ar()

        if ids and task_id in ids:
            pose = poses[ids.index(task_id)]
            x_coord = pose[0][0]
            y_coord = pose[0][1]

            # 计算各种偏差
            x_offset_320 = x_coord - 320  # 距离理论中心的偏差
            x_offset_300 = x_coord - 300  # 距离当前使用中心的偏差
            y_offset_240 = y_coord - 240  # 距离理论Y中心的偏差
            y_offset_208 = y_coord - 208  # 距离task_2参考值的偏差

            # 判断是否在task_2的射击范围内
            task2_x_ok = abs(x_offset_300) < 10
            task2_y_ok = abs(y_offset_208) < 20
            task2_ready = "✅" if (task2_x_ok and task2_y_ok) else "❌"

            # 判断是否在task_3的射击范围内
            task3_x_ok = abs(x_offset_300) <= 3
            task3_ready = "✅" if task3_x_ok else "❌"

            # 实时显示详细坐标信息
            print(f"\r目标AR码 {task_id} 坐标详情:", end="")
            print(f" X={x_coord:.1f} Y={y_coord:.1f}", end="")
            print(f" | X偏差: 理论={x_offset_320:+.1f} 使用={x_offset_300:+.1f}", end="")
            print(f" | Y偏差: 理论={y_offset_240:+.1f} task2基准={y_offset_208:+.1f}", end="")
            print(f" | 射击状态: task2={task2_ready} task3={task3_ready}      ", end="")

        else:
            print(f"\r未检测到目标AR码 {task_id} - 等待检测中...                                                     ",
                  end="")

        # 键盘控制
        if key == ord('s') or key == ord('S'):
            print("\n>>> 射击测试！")
            action.shoot(1)
            rospy.sleep(1)

        elif key == ord('a') or key == ord('A'):
            print("\n>>> 向左微调角度")
            speed.angular.z = 0.1  # 向左转
            pid_vel_pub.publish(speed)
            rospy.sleep(0.2)
            speed.angular.z = 0
            pid_vel_pub.publish(speed)

        elif key == ord('d') or key == ord('D'):
            print("\n>>> 向右微调角度")
            speed.angular.z = -0.1  # 向右转
            pid_vel_pub.publish(speed)
            rospy.sleep(0.2)
            speed.angular.z = 0
            pid_vel_pub.publish(speed)

        elif key == ord('w') or key == ord('W'):
            print("\n>>> 向前微调位置")
            speed.linear.x = 0.1  # 向前
            pid_vel_pub.publish(speed)
            rospy.sleep(0.3)
            speed.linear.x = 0
            pid_vel_pub.publish(speed)

        elif key == ord('x') or key == ord('X'):
            print("\n>>> 向后微调位置")
            speed.linear.x = -0.1  # 向后
            pid_vel_pub.publish(speed)
            rospy.sleep(0.3)
            speed.linear.x = 0
            pid_vel_pub.publish(speed)

        elif key == ord('q') or key == ord('Q') or key == 27:  # 27是ESC键
            print("\n>>> 退出标定模式")
            break

    # 确保机器人停止
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    print("\n=== 标定完成 ===")
    print("标定结果分析：")
    if ids and task_id in ids:
        pose = poses[ids.index(task_id)]
        final_x = pose[0][0]
        final_y = pose[0][1]
        print(f"最终AR码坐标: X={final_x:.1f}, Y={final_y:.1f}")
        print(f"建议视野中心: X={final_x:.0f}")
        print(f"task_2基准Y值: 当前={final_y:.1f}, 建议范围={final_y - 20:.0f}-{final_y + 20:.0f}")
        print(f"task_3无Y限制，当前Y={final_y:.1f}应该可用")
    else:
        print("未检测到目标，请重新标定")
    print("请记录击中目标时的坐标值用于代码调整！")


def task_2(task_id):
    pid_stop(target_position12[0], target_position12[1], 180)
    print("----------退出一号靶场-----------")
    pid_stop2(target_position2[0], target_position2[1], 180)
    # pid_stop2(-1.336, -0.652, 180)
    print("----------到达二号靶场-----------")
    # debug_camera_center(task_id)
    rospy.sleep(0.5)
    shoot_task_2(task_id)
    print("----------射击完成-----------")


def debug_3d_coordinates(task_id):
    """
    调试3D坐标的函数 - 专门用于shoot_task_3
    显示AR码的3D世界坐标，与shoot_task_3使用相同的坐标系统
    """
    from ar_track_alvar_msgs.msg import AlvarMarkers
    import time

    print("=== 进入3D坐标调试模式 ===")
    print("操作说明：")
    print("- 将AR码放在炮口正前方")
    print("- 按 's' 键射击测试")
    print("- 按 'a' 键向左微调角度")
    print("- 按 'd' 键向右微调角度")
    print("- 按 'w' 键向前微调位置")
    print("- 按 'x' 键向后微调位置")
    print("- 按 'q' 键退出调试")
    print("- ESC键退出")
    print()
    print("3D坐标参考信息：")
    print("- X轴：左右方向（负值=左，正值=右）")
    print("- Y轴：前后方向（负值=后，正值=前）")
    print("- Z轴：上下方向（负值=下，正值=上）")
    print("- shoot_task_3 射击范围：-0.19 < x < 0.085")
    print("- 下降趋势射击：-0.19 < x < -0.085")
    print("- 上升趋势射击：-0.19 < x < 0.085")
    print()

    # 3D坐标变量
    current_3d_poses = {}
    last_detection_time = {}
    coordinate_history = []  # 存储坐标历史，用于显示趋势

    def ar_3d_callback(msg):
        nonlocal current_3d_poses, last_detection_time
        current_time = time.time()

        # 清空当前帧的检测结果
        current_frame_ids = set()

        # 更新3D坐标数据
        for marker in msg.markers:
            marker_id = marker.id
            current_frame_ids.add(marker_id)
            current_3d_poses[marker_id] = {
                'x': marker.pose.pose.position.x,
                'y': marker.pose.pose.position.y,
                'z': marker.pose.pose.position.z
            }
            last_detection_time[marker_id] = current_time

        # 移除超过1秒未检测到的目标
        ids_to_remove = []
        for marker_id in current_3d_poses.keys():
            if marker_id not in current_frame_ids:
                if current_time - last_detection_time.get(marker_id, 0) > 1.0:
                    ids_to_remove.append(marker_id)

        for marker_id in ids_to_remove:
            del current_3d_poses[marker_id]
            if marker_id in last_detection_time:
                del last_detection_time[marker_id]

    # 订阅AR话题
    ar_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, ar_3d_callback, queue_size=7)

    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    speed = Twist()

    # 停止机器人
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    rospy.sleep(0.5)
    print("等待3D AR数据...")

    while not rospy.is_shutdown():
        # 显示摄像头画面
        cam.show()
        key = cv2.waitKey(1) & 0xFF

        if task_id in current_3d_poses:
            pos = current_3d_poses[task_id]
            x, y, z = pos['x'], pos['y'], pos['z']

            # 记录坐标历史
            coordinate_history.append(x)
            if len(coordinate_history) > 3:
                coordinate_history.pop(0)

            # 计算趋势
            trend = "稳定"
            if len(coordinate_history) >= 3:
                if coordinate_history[2] < coordinate_history[1] < coordinate_history[0]:
                    trend = "下降⬇️"
                elif coordinate_history[2] > coordinate_history[1] > coordinate_history[0]:
                    trend = "上升⬆️"

            # 判断是否在射击范围内
            in_down_range = -0.2334 < x < -0.22
            in_up_range = -0.2084 < x < -0.2513
            shoot_status_down = "✅下降可射" if in_down_range else "❌"
            shoot_status_up = "✅上升可射" if in_up_range else "❌"

            # 实时显示3D坐标信息
            print(f"\r目标AR码 {task_id} 3D坐标:", end="")
            print(f" X={x:.4f} Y={y:.4f} Z={z:.4f}", end="")
            print(f" | 趋势:{trend}", end="")
            print(f" | {shoot_status_down} {shoot_status_up}        ", end="")

        else:
            coordinate_history.clear()
            print(
                f"\r未检测到目标AR码 {task_id} 的3D坐标 - 等待检测中...                                                          ",
                end="")

        # 键盘控制
        if key == ord('s') or key == ord('S'):
            print("\n>>> 射击测试！")
            if task_id in current_3d_poses:
                pos = current_3d_poses[task_id]
                print(f"射击时3D坐标: X={pos['x']:.4f}, Y={pos['y']:.4f}, Z={pos['z']:.4f}")
            else:
                print("射击时未检测到3D坐标")
            action.shoot(1)
            rospy.sleep(1)

        elif key == ord('a') or key == ord('A'):
            print("\n>>> 向左微调角度")
            speed.angular.z = 0.1
            pid_vel_pub.publish(speed)
            rospy.sleep(0.2)
            speed.angular.z = 0
            pid_vel_pub.publish(speed)

        elif key == ord('d') or key == ord('D'):
            print("\n>>> 向右微调角度")
            speed.angular.z = -0.1
            pid_vel_pub.publish(speed)
            rospy.sleep(0.2)
            speed.angular.z = 0
            pid_vel_pub.publish(speed)

        elif key == ord('w') or key == ord('W'):
            print("\n>>> 向前微调位置")
            speed.linear.x = 0.1
            pid_vel_pub.publish(speed)
            rospy.sleep(0.3)
            speed.linear.x = 0
            pid_vel_pub.publish(speed)

        elif key == ord('x') or key == ord('X'):
            print("\n>>> 向后微调位置")
            speed.linear.x = -0.1
            pid_vel_pub.publish(speed)
            rospy.sleep(0.3)
            speed.linear.x = 0
            pid_vel_pub.publish(speed)

        elif key == ord('q') or key == ord('Q') or key == 27:
            print("\n>>> 退出调试模式")
            break

    # 清理
    ar_sub.unregister()

    # 确保机器人停止
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    print("\n=== 3D坐标调试完成 ===")
    print("调试结果分析：")
    if task_id in current_3d_poses:
        pos = current_3d_poses[task_id]
        final_x = pos['x']
        final_y = pos['y']
        final_z = pos['z']
        print(f"最终3D坐标: X={final_x:.4f}, Y={final_y:.4f}, Z={final_z:.4f}")
        print(f"下降趋势射击范围: -0.2334 < x < -0.22, 当前={final_x:.4f}")
        print(f"上升趋势射击范围: -0.2084 < x < -0.2513, 当前={final_x:.4f}")
        if -0.2334 < final_x < -0.22:
            print("✅ 当前位置在下降趋势射击范围内")
        elif -0.2084 < final_x < -0.2513:
            print("✅ 当前位置在上升趋势射击范围内")
        else:
            print("❌ 当前位置不在任何射击范围内")
    else:
        print("未检测到目标的3D坐标")
    print("请记录击中目标时的3D坐标值用于参数调整！")


def shoot_task_3(task_id):
    """
    3D趋势射击算法 - 使用AR 3D坐标进行移动靶射击
    """
    from ar_track_alvar_msgs.msg import AlvarMarkers
    import time

    # 射击参数
    max_shots = 3  # 最大射击次数
    shot_count = 0

    # 目标丢失参数
    max_lost_count = 30  # 目标丢失30次后判定为击倒
    target_lost_count = 0

    # 3D坐标变量
    current_3d_poses = {}
    last_detection_time = {}

    # 趋势跟踪变量
    last_offset_x = 0
    last_last_offset_x = 0
    offset_x = 0

    def ar_callback(msg):
        nonlocal current_3d_poses, last_detection_time
        current_time = time.time()

        # 清空当前帧的检测结果
        current_frame_ids = set()

        # 更新3D坐标数据
        for marker in msg.markers:
            marker_id = marker.id
            current_frame_ids.add(marker_id)
            current_3d_poses[marker_id] = {
                'x': marker.pose.pose.position.x,
                'y': marker.pose.pose.position.y,
                'z': marker.pose.pose.position.z
            }
            last_detection_time[marker_id] = current_time

        # 移除超过1秒未检测到的目标
        ids_to_remove = []
        for marker_id in current_3d_poses.keys():
            if marker_id not in current_frame_ids:
                if current_time - last_detection_time.get(marker_id, 0) > 1.0:
                    ids_to_remove.append(marker_id)

        for marker_id in ids_to_remove:
            del current_3d_poses[marker_id]
            if marker_id in last_detection_time:
                del last_detection_time[marker_id]

    # 订阅AR话题获取3D坐标
    ar_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, ar_callback, queue_size=7)

    # 等待数据
    rospy.sleep(0.1)
    print("等待AR 3D数据...")
    while not current_3d_poses and not rospy.is_shutdown():
        rospy.sleep(0.1)
    print("开始3D趋势射击算法...")

    pid_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    rate_loop_pid = rospy.Rate(20)
    speed = Twist()

    # 停止机器人移动
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

    # 第一个循环：等待检测到目标
    while not rospy.is_shutdown():
        cam.show()
        cv2.waitKey(1)

        print(f"检测到的3D AR码: {list(current_3d_poses.keys())}, 目标: {task_id}")

        if not current_3d_poses:
            print("没检测到任何AR码")
        elif task_id not in current_3d_poses:
            print(f"没检测到目标{task_id}")
        else:
            print(f"✅ 检测到目标AR码 {task_id}")
            # 初始化坐标
            offset_x = current_3d_poses[task_id]['x']
            print(f"初始3D位置 x: {offset_x}")
            target_lost_count = 0  # 重置丢失计数
            break

        rospy.sleep(0.1)

    # 第二个循环：基于3D趋势进行射击
    while not rospy.is_shutdown() and shot_count < max_shots:
        cam.show()
        cv2.waitKey(1)

        if task_id in current_3d_poses:
            # 重置丢失计数
            target_lost_count = 0

            # 更新坐标趋势
            last_last_offset_x = last_offset_x
            last_offset_x = offset_x
            offset_x = current_3d_poses[task_id]['x']

            # print(f"3D坐标趋势: {last_last_offset_x:.3f} -> {last_offset_x:.3f} -> {offset_x:.3f}")
            print(offset_x)

            condition_1 = (offset_x < -0.22 and offset_x > -0.2334 and
                           last_offset_x > offset_x and
                           last_last_offset_x > offset_x)

            # 条件2：上升趋势射击
            condition_2 = (offset_x > -0.2513 and offset_x < -0.2084 and
                           last_offset_x < offset_x and
                           last_last_offset_x < offset_x)

            if condition_1:
                print("🎯 检测到下降趋势，射击！")
                print(f"下降趋势: {last_last_offset_x:.3f} > {last_offset_x:.3f} > {offset_x:.3f}")

                # 射击序列
                action.shoot(1)
                print("开炮")
                shot_count += 1
                rospy.sleep(0.02)

                # 检查是否击倒目标 - 增加检查时间
                print("检查目标是否被击倒...")
                rospy.sleep(1.0)  # 增加等待时间

                # 连续检查3次，确保目标真的消失
                target_still_exists = False
                for check_count in range(5):
                    rospy.sleep(0.5)
                    if task_id in current_3d_poses:
                        target_still_exists = True
                        break

                if not target_still_exists:
                    print("---------已经击倒目标----------")
                    break
                else:
                    print("目标仍然存在，继续射击...")

            elif condition_2:
                print("🎯 检测到上升趋势，射击！")
                print(f"上升趋势: {last_last_offset_x:.3f} < {last_offset_x:.3f} < {offset_x:.3f}")

                # 射击序列
                action.shoot(1)
                print("开炮")
                shot_count += 1
                rospy.sleep(0.02)

                # 检查是否击倒目标 - 增加检查时间
                print("检查目标是否被击倒...")
                rospy.sleep(1.0)  # 增加等待时间

                # 连续检查3次，确保目标真的消失
                target_still_exists = False
                for check_count in range(5):
                    rospy.sleep(0.5)
                    if task_id in current_3d_poses:
                        target_still_exists = True
                        break

                if not target_still_exists:
                    print("---------已经击倒目标----------")
                    break
                else:
                    print("目标仍然存在，继续射击...")
            else:
                print(f"等待合适的射击时机... x={offset_x:.3f}")
        else:
            # 目标丢失处理
            target_lost_count += 1
            print(f"丢失目标{task_id} (丢失计数: {target_lost_count}/{max_lost_count})")

            # 如果目标丢失次数过多，可能已经被击倒
            if target_lost_count >= max_lost_count:
                print("---------目标长时间丢失，可能已被击倒----------")
                break

            # 短暂等待，给目标重新出现的机会
            rospy.sleep(0.1)

        rate_loop_pid.sleep()

    # 清理
    ar_sub.unregister()
    print(f"射击任务完成，共射击 {shot_count} 次")

    # 确保机器人停止
    speed.linear.x = 0
    speed.linear.y = 0
    speed.angular.z = 0
    pid_vel_pub.publish(speed)

def compute_angular_pid_control(target_orientation, current_robot_pose):
    """Calculate PID output for angular control"""
    global angular_proportional_gain, angular_integral_gain, angular_derivative_gain
    global angular_accumulated_error, angular_previous_error

    normalized_robot_pose = (current_robot_pose / 3.14159265359 * 180.0 + 180.0) % 360
    adjusted_target = target_orientation

    if adjusted_target == 0:
        if 0 < normalized_robot_pose < 180:
            adjusted_target = 0
        if 180 < normalized_robot_pose < 360:
            adjusted_target = 360

    angular_error = normalized_robot_pose - adjusted_target
    angular_accumulated_error = angular_accumulated_error + angular_error
    pid_output_result = (angular_proportional_gain * angular_error +
                         angular_integral_gain * angular_accumulated_error +
                         angular_derivative_gain * (angular_error - angular_previous_error))
    angular_previous_error = angular_error
    return pid_output_result


def execute_angle_only_calibration(target_orientation_angle):
    """只校准角度的PID，不调整位置，在当前位置精确调整到目标角度"""
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=7)
    loop_rate = rospy.Rate(10)
    movement_command = Twist()
    consecutive_success_count = 0
    iteration_counter = 0

    print(f"开始角度校准，目标角度: {target_orientation_angle}°")

    while not rospy.is_shutdown():
        loop_rate.sleep()
        iteration_counter += 1

        # 只进行角度控制
        angular_pid_output = compute_angular_pid_control(target_orientation_angle, robot_orientation_yaw)

        # 不调整位置，只调整角度
        movement_command.linear.x = 0
        movement_command.linear.y = 0
        movement_command.angular.z = angular_pid_output / 180.0 * 3.14159265359

        # 超时保护
        if iteration_counter >= 100:
            break

        # 角度精度检查
        current_angle_normalized = (robot_orientation_yaw / 3.1415926 * 180 + 180) % 360
        angle_ok = abs(target_orientation_angle - current_angle_normalized) <= 2

        if angle_ok:
            consecutive_success_count += 1
        else:
            consecutive_success_count = 0

        if consecutive_success_count >= 5:
            movement_command.linear.x = 0
            movement_command.linear.y = 0
            movement_command.angular.z = 0
            velocity_publisher.publish(movement_command)
            print(f"角度校准完成，当前角度: {current_angle_normalized:.1f}°")
            break

        velocity_publisher.publish(movement_command)


def task_3(task_id):
    pid_stop(target_position23[0], target_position23[1], 180)
    print("----------退出二号靶场-----------")
    pid_stop2(target_position3[0], target_position3[1], 180)
    # debug_3d_coordinates(task_id)
    # pid_stop2(-1.382,0.596,180)
    print("----------到达三号靶场-----------")
    execute_angle_only_calibration(180)
    shoot_task_3(task_id)
    print("----------完成射击--------------")


def init_fun():
    # 转换点
    global qtn_list_xy, qtn_list_task_xy, pose_num_xy, pose_num_task_xy, move_base
    for i in range(pose_num_xy):
        qtn = tf.transformations.quaternion_from_euler(0, 0, xy[i][2])
        qtn_list_xy.append(qtn)
    i = 0
    for i in range(pose_num_task_xy):
        qtn = tf.transformations.quaternion_from_euler(0, 0, task_xy[i][2])
        qtn_list_task_xy.append(qtn)

    # 连接move_base—actition
    move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    while move_base.wait_for_server(rospy.Duration(5.0)) == 0:
        rospy.loginfo("Request to connect to move_base server")
    rospy.loginfo("Be connected successfully")

    thread_lidar = threading.Thread(target=laser_listen)
    thread_lidar.start()

    thread_vision = threading.Thread(target=vision_listen)
    thread_vision.start()

    thread_now_pose = threading.Thread(target=now_pose_xy)
    thread_now_pose.start()


def comprehensive_camera_check(cam):
    """简化的摄像头检测功能 - 仅检查基本状态"""
    print("开始摄像头检测...")

    try:
        # 检查Camera对象是否有cap属性
        if hasattr(cam, 'cap'):
            cap = cam.cap
            print("✅ 摄像头对象: 正常初始化")

            # 检查摄像头是否打开
            if cap.isOpened():
                print("✅ 摄像头状态: 已成功打开")

                # 获取摄像头尺寸
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"📏 摄像头尺寸: {width} x {height}")

                return True
            else:
                print("❌ 摄像头状态: 未能打开")
                return False

        else:
            print("❌ 摄像头对象: 缺少cap属性")
            return False

    except Exception as e:
        print(f"❌ 摄像头检查失败: {e}")
        return False

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('move_test', anonymous=True)
    action = Action("/dev/shoot", 9600)
    cam = Camera(0)
    cam.open()

    comprehensive_camera_check(cam)

    # 添加输出摄像头尺寸的代码
    try:
        # 如果Camera类有获取尺寸的方法
        if hasattr(cam, 'get_frame_size'):
            width, height = cam.get_frame_size()
            print(f"摄像头尺寸: {width} x {height}")

        # 或者直接访问OpenCV的VideoCapture对象
        elif hasattr(cam, 'cap'):
            width = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"摄像头尺寸: {width} x {height}")

        # 或者通过读取一帧来获取尺寸
        else:
            # 读取一帧获取尺寸
            ret, frame = cam.cap.read() if hasattr(cam, 'cap') else (False, None)
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"摄像头尺寸: {width} x {height}")
            else:
                print("无法获取摄像头尺寸")

    except Exception as e:
        print(f"获取摄像头尺寸时出错: {e}")

    init_fun()

    print("=== 机器人射击比赛系统启动 ===")
    print(f"目标设置: 二号靶场(旋转靶)={target_2}, 三号靶场(移动靶)={target_3}")

    # ========== 修改部分：添加简单的启动确认 ==========
    print("\n请输入 1 开始比赛:")
    user_input = raw_input() if sys.version_info[0] < 3 else input()

    if user_input.strip() == "1":
        print("=== 开始执行射击任务 ===")
    else:
        print("输入无效，退出程序")
        sys.exit(0)

    task_shoot_ar2 = target_2
    task_shoot_ar3 = target_3
    b = 275
    kp = -0.005
    ki = -0.00002
    kd = 0
    begin_time = rospy.Time.now()

    task_1(350, -0.005, -0.00002, 0)
    task_2(int(task_shoot_ar2))
    task_3(int(task_shoot_ar3))
    pid_bask(-0.23, -0.22, 180)

    print("=== 导航到终点 ===")

    print("=== 比赛结束 ===")

    finish_time = rospy.Time.now()
    print("任务完成，总用时:", (finish_time - begin_time).to_sec(), "秒")
