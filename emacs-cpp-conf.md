# emacs cpp configuration

本文档纪录emacs中c++的配置和使用方法, 以便后来者参考.

## 依赖库安装

```shell
sudo apt install clang-6.0 clang-format-6.0

# global需要6.2.3及以上版本, Ubuntu提供的版本为5.7.1, 所以需要源码安装
wget http://tamacom.com/global/global-6.6.4.tar.gz
tar zxvf global-6.6.4.tar.gz && cd cd global-6.6.4
./configure --prefix=${HOME}/Documents/tools/global
make -j4 && make install
# 设置环境变量

export PATH=${TOOLS}/global/bin:$PATH
```

## keymap

1. basic navigation, 在括号的开头和结尾, 引号的开头和结尾, 函数定义的开头和结尾,  等等之间跳转:

| key               | function           |
|:-----------------:|:------------------:|
| C-M-f             | forward-sexp       |
| C-M-b             | backward-sexp      |
| C-M-k             | kill-sexp          |
| C-M-<SPC> / C-M-@ | mark-sexp          |
| C-M-a             | beginning-of-defun |
| C-M-e             | end-of-defun       |
| C-M-h             | mark-defun         |

2. 在函数申明, 定义, 和引用之间跳转:

| key           | function                         |
|:-------------:|:--------------------------------:|
| M-.           | helm-gtags-dwim                  |
| M-,           | helm-gtags-pop-stack             |
| C-c g r       | helm-gtags-find-rtag             |
| C-c g s       | helm-gtags-find-symbol           |
| C-c g a       | helm-gtags-tags-in-this-function |
| C-c g <left>  | helm-gtags-previous-history      |
| C-c g <right> | helm-gtags-next-history          |
| C-c g o       | helm-gtags-show-stack            |

其中, `helm-gtags-dwim` 做如下事情:

1) If the tag at point is a definition, jumps to a reference;

2) If the tag at point is a reference, jumps to tag definition;

3) If the tag at point is an include header, it jumps to that header.
