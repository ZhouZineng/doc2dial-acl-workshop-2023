# Doc2dial

nsddd队伍方案

## Preparation

1. Git clone repo

   ```shell
   git clone https://github.com/NSDDD-ict/doc2dial-acl-workshop-2023.git
   ```

2. 下载 model 文件夹

   因为有模型集成，所有模型文件夹大小在40G左右

   ```
   下载链接: https://pan.baidu.com/s/1y_WIh9u0GuEQldDtTZk_dA?pwd=34g6 提取码: 34g6 
   --来自百度网盘超级会员v1的分享
   ```

   linux终端下百度云盘稳定下载方法

   1. 安装bypy

      ```shell
      # pip 安装
      pip install bypy
      ```

      第一次使用的时候，因为需要注册访问百度网盘api，所以随便输入一个bypy命令，如：

      ```shell
      bypy info
      ```

      这时会命令行返回一个网址，打开此网址，登录百度网盘之后，得到验证码，返回命令行输入此验证码，即可使bypy得到百度网盘api使用权限，接下来就可以使用bypy管理百度网盘文件的上传下载了。

      

   2. （百度云盘app中）登陆后，百度云盘会自动建立bypy文件夹，将model文件夹保存到bypy文件夹下面

      ![image-20230406153852217](https://gitee.com/Ljunius/image-bed/raw/master/img/202304061538561.webp)

   3. （linux服务器中）进入repo 目录下并下载model文件夹到当前目录

      ```
      cd doc2dial-acl-workshop-2023
      bypy downfile model
      ```

2. Install requirements.

```bash
pip install -r requirements.txt
```

## Testing

### bash

```shell
bash run_infer.sh
```

运行推理脚本，直接生成结果，结果在`./model_outputStandardFileBaseline.json` 



**分开推理**

### Retrieval

```bash
python inference_retrieval.py
```


### Rerank

```bash
python inference_rerank.py
```

This produces the rerank result **rerank_output.jsonl**

### Generation

```bash
python inference_generation.py
```

This produces the generation result **outputStandardFile.json**



## Training

### Retrieval

```bash
python init_model.py
python train_retrieval.py
```


### Rerank

```bash
python train_rerank.py
```


### Generation

```bash
python train_generation.py
```



