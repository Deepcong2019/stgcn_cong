import onnxruntime
import numpy as np
import json
import os
# 从文件中读取
jsons_path = 'D:\\st-gcn\\data\\chongqing\\kinetics-skeleton\\kinetics_val'
for path in os.listdir(jsons_path):
    file_path = os.path.join(jsons_path, path)
    with open(file_path, "r") as f:
        video_info = json.load(f)
    label = video_info['label_index']
    print('label:', label)
    data_numpy = np.zeros((3, 300, 26, 1))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            # 1:n_in
            if m >= 1:
                break
            pose = skeleton_info['pose']
            # print('pose:', pose)
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]  # 取奇数位
            data_numpy[1, frame_index, :, m] = pose[1::2]  # 取出偶数位置的元素
            data_numpy[2, frame_index, :, m] = score

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
    data_numpy = data_numpy[:, :, :, 0:2].astype(np.float32)
    data_numpy = data_numpy.reshape(-1, 3, 300, 26, 1)

    # 创建一个运行时会话并加载ONNX模型
    session = onnxruntime.InferenceSession('stgcn.onnx')

    # 准备输入数据，这里需要根据你的模型的输入进行修改
    # input_feed = np.random.randn(1, 3, 300, 18, 2).astype(np.float32)  # 假设输入形状为 NCHW

    # 运行模型进行推理
    output_names = [node.name for node in session.get_outputs()]
    output_dict = session.run(output_names, input_feed={session.get_inputs()[0].name: data_numpy})

    # 处理输出
    # 这里假设模型只有一个输出，并且我们需要它
    predictions = list(output_dict[0][0]).index(max(output_dict[0][0]))

    print('pre:', predictions)