# Badnets

## Environment
See requirements.txt

## Run
使用默认的cifar10数据集和PreActResNet18模型，运行如下命令：
```bash
python main.py
```
更改datasets和model可以用命令行：
```bash
python train.py dataset=cifar10 model=densenet121
```
或者更改`config.yaml`文件
## Result
投毒预览情况：

默认配置运行 5 epoch, 最终最佳结果为
训练train的loss和acc, val的clean_acc和asr(poison_acc)原始值见`metrics_raw.csv`
(在跑的时候第4个epoch写入的时候发生的点异常，所以是手动写的终端里面的输出结果，所以保留位数不太统一)

各曲线如下：
<img src="./docs/train_curve.png" width="80%" title="train_curve" />
<img src="./docs/train_step_curve.png" width="50%" title="train_step_curve" />
<img src="./docs/eval_curve.png" width="80%" title="eval_curve" />

## Example
<img src="./docs/example_dog.png" width="200%" title="example" />
<img src="./docs/example_result.png" width="50%" title="example_result" />


