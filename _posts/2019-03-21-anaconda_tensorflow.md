---
layout: post
title: Docker에서 Anaconda 가상환경에 tensorflow, tensorflow-gpu 설치하기
subtitle: 
tags: [develop, docker, anaconda, tensorflow, ubuntu]
comments: true
---

Docker 우분투 컨테이너 위에서 새롭게 머신러닝 환경을 세팅하는데 나와 같은 고생을 하지 않기를 바라면서 글을 작성합니다.  

### Docker 환경 : ubuntu 14.04  
### 파이썬 버전 : 3.6

## Anaconda 가상환경 설치

```
cd ./home/anaconda

curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh

sha256sum Anaconda3-5.2.0-Linux-x86_64.sh

bash Anaconda3-5.2.0-Linux-x86_64.sh
```
<hr>  

```
Do you wish to proceed with the installation of Microsoft VSCode? [yes|no]
```
이 과정에서는 no를 누르시면 됩니다.  
(docker 환경에서 vscode를 쓰지 않기 때문에 설치하지 않습니다.)


이 외의 모든 설치 과정에서 yes를 하시면 됩니다.

설치 완료 후 아래 구문을 실행합니다.
```
source ~/.bashrc

//아나콘다 버전확인
conda --version

//아나콘다 업데이트
conda update conda
```
<hr>

## 가상환경 만들기

### 가상환경 생성
```
//cpu 
conda create --name [가상환경이름] python=3.X(원하는 버전) tensorflow
//gpu 
conda create --name [가상환경이름] python=3.X(원하는 버전) tensorflow-gpu
```

### gpu 선택시 해줘야할 것
```
//아래 문구를  ~/.bashrc 마지막에 입력
export CUDA_VISIBLE_DEVICES=''

// .bashrc 수정 후
source ~/.bashrc
```

```
//가상환경 실행
source activate [가상환경이름]
```
### 여기까지 하시면 세팅은 완료됩니다.
<hr>
<br><br><br>


## 가상환경 삭제 (필요시)

```
conda remove --name [가상환경명] --all
```

## 아나콘다 clean (필요시)
인덱스 캐시, 잠긴 파일, 소스캐시 등을 삭제할 수 있습니다.
```
conda clean --all
//or
conda clean -a
```

## CUDA 삭제 (필요시)
삭제할 때 아래와 같이 진행하시면 됩니다.
```
// 간단한 방법 
sudo apt-get remove --auto-remove nvidia-cuda-toolkit
// 좀 더 깨끗하게 
sudo apt-get --purge remove 'cuda*' 
sudo apt-get autoremove --purge 'cuda*' 

sudo rm -rf /usr/local/cuda-[your version] 
sudo rm -rf /usr/local/cuda
```

잘못된 부분이 있다면 언제든 댓글 혹은 이메일로 피드백 해주시면 빠르게 수정하겠습니다.

참조 링크 1: https://ufris.tistory.com/5  
참조 링크 2: https://niceman.tistory.com/86?category=940952  
참조 링크 3: https://goodtogreate.tistory.com/entry/TensorFlow-GPU-%EB%B2%84%EC%A0%84-%EC%9A%B0%EB%B6%84%ED%88%AC-1604%EC%97%90-%EC%84%A4%EC%B9%98-%ED%95%98%EA%B8%B0  