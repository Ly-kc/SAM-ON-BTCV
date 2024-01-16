

## Code Structure

```bash
├── ckpts
│   ├── sam_vit_h_4b8939.pth   #原始预训练权重
│   ├── vit_h_mask_2024-01-15-05-42-30   #finetune后的mask decoder权重以及训练log
│   ├── vit_h_semantic_mask_2024-01-15-22-51-30    #训练后的semantic mask decoder权重以及训练log
├── data
│   ├── processed         #预处理后的数据集
│   ├── sam_embedding     #保存以便多次使用的image embedding
│   ├── Testing        #原数据集Testing
│   ├── Training	   #原数据集Training
│   ├── centers.txt      #训练集的窗口中心
│   └── widths.txt       #训练集的窗口宽度
├── id_to_color.txt       #器官类别到color的映射
└── sam_on_btcv
    ├── segment_anything             #相较原版sam代码新增了build_sem_sam.py以及SemanticMaskDecoder
    ├── btcv_dataset.py         #Dataset类
    ├── criterion.py            #Dice loss
    ├── grid_sam.py             #以grid points作为prompts，展开一系列应用
    ├── myAutomaticMaskGenerator.py         #以grid points为输入的semantic mask预测
    ├── my_pridictor.py                     #用于finetune decoder以及训练semantic mask decoder
    ├── preprocess_dataset.py           #数据预处理
    ├── finetune.py                 #finetune mask decoder
    ├── train_semantic.py        #train semantic mask decoder
    └── visualize.py             #各种可视化应用
```

## Usage

### Installation

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python,tqdm,SimpleITK,open3d,matplotlib
```

### Data preparation

1. 下载预训练权重至ckpts/，下载数据集保存至data/Testing与data/Training
2. 运行```python preprocess_dataset.py ```预处理数据



### 

1. 运行```python finetune.py ```来finetune mask decoder
2. 运行```python train_semantic.py ```来训练semantic mask decoder （需要修改底部from_pretrain参数）

