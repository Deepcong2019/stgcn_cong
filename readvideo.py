import cv2
cap = cv2.VideoCapture('E:\\捷安项目\\datasets\\record_night\\1\\姚雅倩1.mkv')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame_count:", frame_count)
# 通过循环读取视频的每一帧
count = 0
while True:
    # 读取一帧
    ret, frame = cap.read()
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    print(count)
    print(timestamp)
    count += 1
    # 如果正确读取帧，ret为True
    if not ret :
        print("count:", count)
        print("无法读取视频流或文件结束")
        break

    # 显示帧
    cv2.imshow('Video', frame)

    # 如果按下'q'键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象
cap.release()
# 销毁所有窗口
cv2.destroyAllWindows()