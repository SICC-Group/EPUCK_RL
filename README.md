# Webots_RL 框架

# 安装
1.安装webots（2023b版本）：
- [Linux](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-linux)

2.建立虚拟环境：
```
conda create -n webots python=3.8
```

3.安装deepbots:
```
pip install deepbots
```

4.安装依赖包：
```
pip install -r requirements.txt
```

5.安装pytorch及相关包：

下载对应python版本和对应电脑cuda版本的pytorch、torch-cluster、torch_geometric、torch-sactter、torch-sparse、torch-spline-conv（https://download.pytorch.org/whl/torch_stable.html）

5.配置文件：
在epuck_controller、target_controller、supervisor_controller文件夹下，找到对应文件夹名字的.py文件，并在第一行指定环境：
```
#!<自己安装路径>/miniconda3/envs/webots/bin/python
```

# 运行
1.打开webots载入worlds.wbt

2.执行simulation

3.进入虚拟环境：
```
conda activate webots
```

4.在supervisor_controller文件夹下，运行train_prey_rmdfg.sh文件
```
python train_prey_rmdfg.sh
```