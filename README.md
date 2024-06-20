# 基于Alphapose生成的skeleton point,整理成stgcn需要的的格式（类似kinetics-skeleton），进入stgcn训练<br>
参考：https://blog.csdn.net/qq_43019451/article/details/118787580
1、修改tools\kinetics_gendata.py中：


    '--data_path', default='data/KTH/kinetics-skeleton'
    '--out_folder', default='data/KTH/kinetics-skeleton'

    def gendata(
            data_path,
            label_path,
            data_out_path,
            label_out_path,
            num_person_in=1,  #observe the first 5 persons
            num_person_out=1,  #then choose 2 persons with the highest score
            max_frame=300):

	shape=(len(sample_name), 3, max_frame, 18, num_person_out))
<br>
2、修改feeder/feeder_kinetics.py中：

    # sample_id = [name.split('.')[0] for name in self.sample_name]
    sample_id = [name.split('.json')[0] for name in self.sample_name]
    self.V = 26  # joint,根据不同的skeleton获得的节点数而定，coco是18个节点

<br>

3、python tools/kinetics_gendata.py, 生成:

    train_data.npy
    train_label.pkl
    val_data.npy
    val_label.pkl

4、在net/utils/graph.py文件里面get_edge函数中增加一个elif，num_node为 关键点个数、self_link为连接关系。如下添加的是一个‘my_pose’Layout，关键点个数为20（默认的pose点数是18）。

        elif layout == 'my_pose':
            self.num_node = 26
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 3), (0, 2), (2, 4), (17, 18), (18, 6),
                             (18, 5), (6, 8), (8, 10), (5, 7),
                             (7, 9), (11, 19), (18, 19), (19, 12), (14, 12), (11, 13),
                             (14, 16), (16, 25), (25, 21), (25, 23), (13, 15), (15, 24),
                             (24, 20),(24,22)]
            self.edge = self_link + neighbor_link
            self.center = 18-19


5、修改config/st_gcn/kinetics-skeleton/train.yaml中的相关参数：


    work_dir: ./work_dir/recognition/kinetics_skeleton/ST_GCN
    
    # feeder
    feeder: feeder.feeder.Feeder
    train_feeder_args:
      random_choose: True
      random_move: True
      window_size: 150
      data_path: ./data/KTH/kinetics-skeleton/train_data.npy
      label_path: ./data/KTH/kinetics-skeleton/train_label.pkl
    test_feeder_args:
      data_path: ./data/KTH/kinetics-skeleton/val_data.npy
      label_path: ./data/KTH/kinetics-skeleton/val_label.pkl
    
    # model
    model: net.st_gcn.Model
    model_args:
      in_channels: 3
      num_class: 6
      edge_importance_weighting: True
      graph_args:
        layout: 'my_pose'
        strategy: 'spatial'
    
    # training
    device: [0]
    batch_size: 32
    test_batch_size: 32
    
    #optim
    base_lr: 0.1
    step: [20, 30, 40, 50]
    num_epoch: 500

6、执行训练：python main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml

7、在resource/kinetics_skeleton中，新建lable_name.txt,类别写入。
   修改test.py中：

       def __init__(self, in_channels=3, num_class=6,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        # load graph
        self.graph = Graph(layout='my_pose', strategy='spatial',)


    weights_path = 'work_dir/recognition/kinetics_skeleton/ST_GCN/epoch30_model.pt'
        # label_list = ['standing', 'walking', 'laying']
        label_list = []
        with open('resource/kinetics_skeleton/label_name_KTH.txt', 'r') as f:
            for line in f:
                label_list.append(line.strip('\n'))
        data_path = "data/KTH/kinetics-skeleton/val_data.npy"
        label_path = "data/KTH/kinetics-skeleton/val_label.pkl"
   测试：python test.py,