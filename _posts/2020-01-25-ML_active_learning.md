---
layout: post
title: MNIST로 Active Learning 하기
subtitle: Labeling 시간을 줄여보자
tags: [study, machinelearning, deeplearning, ML, active learning, data science]
comments: true
use_math: true
---



> 이 포스트는 tensorflow 1.15.0 버전을 기반으로 작성되었습니다.

# MNIST로 Active Learning 하기

Active Learning은 학습 과정 (loss) 관점에서 가장 중요한 샘플을 선택하여 적은 데이터에 레이블을 지정할 수있는 semi-supervised 기법입니다. 데이터 양이 많고 라벨링을 해야하는 비율이 많은 경우 프로젝트 비용에 큰 영향을 줄 수 있습니다. 비율이 높습니다. 예를 들어, Object Detection 및 NLP-NER 문제가 있습니다.
이 포스트는 다음 코드를 기반으로합니다. [**Active Learning on MNIST**](https://github.com/andy-bosyi/articles/blob/master/ActiveLearning-MNIST.ipynb)



## 실험을 위한 데이터

```python
# load 4000 of MNIST data for train and 400 for testing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_full = x_train[:4000] / 255
y_full = y_train[:4000]
x_test = x_test[:400] /255
y_test = y_test[:400]
x_full.shape, y_full.shape, x_test.shape, y_test.shape

# 출력 값
# ((4000, 28, 28), (4000,), (400, 28, 28), (400,))
```

```python
plt.imshow(x_full[3999])
```

<img src="data:image/png;base64,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%0A" />



레이블과 10K개 테스트 샘플이있는 60K개 숫자의 MNIST 데이터 세트를 사용합니다. 보다 빠른 훈련을 위해 훈련에 4000 개의 샘플 (사진)이 필요하고 테스트에 400 개가 필요합니다 (신경망은 훈련 중에는 이를 보지 못합니다). 정규화를 위해 그레이 스케일 이미지 포인트를 255로 나눕니다.



## 모델, 훈련, 라벨링 과정

```python
# build computation graph
x = tf.placeholder(tf.float32, [None, 28, 28])
x_flat = tf.reshape(x, [-1, 28 * 28])
y_ = tf.placeholder(tf.int32, [None])
W = tf.Variable(tf.zeros([28 * 28, 10]), tf.float32)
b = tf.Variable(tf.zeros([10]), tf.float32)
y = tf.matmul(x_flat, W) + b
y_sm = tf.nn.softmax(y)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
train = tf.train.AdamOptimizer(0.1).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_, tf.cast(tf.argmax(y, 1), tf.int32)), tf.float32))
```

softmax 출력 y_sm은 숫자의 확률을 나타낸다. Loss는 예측 된 데이터와 라벨링 된 데이터 사이의 전형적인 "소프트 맥스"교차 엔트로피가 될 것입니다. 유명한 Adam을 optimizer로 선택했습니다(원문에서 유명해서 선택했다고 합니다..). Learning rate는 기본 0.1로 설정했습니다. 위의 정확도(accuracy)를 테스트 데이터셋에서도 사용합니다.



````python
def reset():
    '''Initialize data sets and session'''
    global x_labeled, y_labeled, x_unlabeled, y_unlabeled
    x_labeled = x_full[:0]
    y_labeled = y_full[:0]
    x_unlabeled = x_full
    y_unlabeled = y_full
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() 

def fit():
    '''Train current labeled dataset until overfit.'''
    trial_count = 10
    acc = sess.run(accuracy, feed_dict={x:x_test, y_:y_test})
    weights = sess.run([W, b])
    while trial_count > 0:
        sess.run(train, feed_dict={x:x_labeled, y_:y_labeled})
        acc_new = sess.run(accuracy, feed_dict={x:x_test, y_:y_test})
        if acc_new <= acc:
            trial_count -= 1
        else:
            trial_count = 10
            weights = sess.run([W, b])
            acc = acc_new

    sess.run([W.assign(weights[0]), b.assign(weights[1])])    
    acc = sess.run(accuracy, feed_dict={x:x_test, y_:y_test})
    print('Labels:', x_labeled.shape[0], '\tAccuracy:', acc)

def label_manually(n):
    '''Human powered labeling (actually copying from the prelabeled MNIST dataset).'''
    global x_labeled, y_labeled, x_unlabeled, y_unlabeled
    x_labeled = np.concatenate([x_labeled, x_unlabeled[:n]])
    y_labeled = np.concatenate([y_labeled, y_unlabeled[:n]])
    x_unlabeled = x_unlabeled[n:]
    y_unlabeled = y_unlabeled[n:]
````

여기서는 보다 편리한 코딩을 위해이 세 가지 절차를 정의합니다.
**reset ()** — 라벨이 지정된 데이터 세트를 비우고 라벨이없는 데이터 세트에 모든 데이터를 넣고 세션 변수를 재설정합니다

**fit ()** — 최고의 정확도에 도달하기 위해 훈련을 실행합니다. 처음 10 번의 시도 중에도 향상되지 않으면 훈련은 마지막 최상의 결과에서 멈춥니다. 모델이 빠르게 과적합(overfit)되거나 집중적인(intensive) L2 정규화가 필요하기 때문에 많은 훈련 시대(training epochs)를 사용할 수 없습니다.

**label_manually ()** — 이것은 휴먼 데이터 레이블의 에뮬레이션입니다. 실제로, 우리는 이미 레이블이 지정된 MNIST 데이터 세트에서 레이블을 가져옵니다.



## Ground Truth

```python
# train full dataset of 1000
reset()
label_manually(4000)
fit()

# 출력
# Labels: 4000 	Accuracy: 0.9225
```

운이 좋으면 전체 데이터 세트에 레이블을 지정할만큼 충분한 리소스가 있으면 92.25 %의 정확도를 얻게됩니다.



## Clustering

```python
# apply clustering
kmeans = tf.contrib.factorization.KMeansClustering(10, use_mini_batch=False)
kmeans.train(lambda: tf.train.limit_epochs(x_full.reshape(4000, 784).astype(np.float32), 10))

# 출력 값
'''
INFO:tensorflow:Using default config.
WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpuk2rv4h5
INFO:tensorflow:Using config: {'_session_config': None, '_evaluation_master': '', '_tf_random_seed': None, '_save_checkpoints_steps': None, '_model_dir': '/tmp/tmpuk2rv4h5', '_master': '', '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f58d0363d30>, '_service': None, '_num_ps_replicas': 0, '_task_id': 0, '_save_summary_steps': 100, '_keep_checkpoint_every_n_hours': 10000, '_keep_checkpoint_max': 5, '_num_worker_replicas': 1, '_is_chief': True, '_global_id_in_cluster': 0, '_train_distribute': None, '_task_type': 'worker', '_log_step_count_steps': 100, '_save_checkpoints_secs': 600}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpuk2rv4h5/model.ckpt.
INFO:tensorflow:loss = 274107.66, step = 1
INFO:tensorflow:Saving checkpoints for 10 into /tmp/tmpuk2rv4h5/model.ckpt.
INFO:tensorflow:Loss for final step: 154262.86.
Out[8]:
<tensorflow.contrib.factorization.python.ops.kmeans.KMeansClustering at 0x7f59087b3780>
'''
```

```python
centers = kmeans.cluster_centers().reshape([10, 28, 28])
plt.imshow(np.concatenate([centers[i] for i in range(10)], axis=1))
```

<img src="data:image/png;base64,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%0A" />

여기서 k- 평균 군집화를 사용하여 숫자 그룹을 찾고 이 정보를 자동 레이블링에 사용하려고합니다. Tensorflow clustering estimator를 실행한 다음 결과 10개의 중심(centroids)을 시각화합니다. 보다시피 결과는 완벽하지 않습니다. 숫자 "9"가 세 번 나타나고  "8"과 "3"은 섞여 나타납니다.



## Random Labeling

```python
#try to run on random 400
reset()
label_manually(400)
fit()

# 출력 값
# Labels: 400 	Accuracy: 0.8375
```

10 %의 데이터 (400 개 샘플)에 레이블을 지정하려고 시도하면 ground truth의 정확도인 92.25%와는 거리가 먼 83.75%의 정확도를 얻게됩니다.



## Active Learning

```python
#now try to run on 10
reset()
label_manually(10)
fit()

# 출력 값
# Labels: 10 	Accuracy: 0.38
```

```python
# pass unlabeled rest 3990 through the early model
res = sess.run(y_sm, feed_dict={x:x_unlabeled})
#find less confident samples
pmax = np.amax(res, axis=1)
pidx = np.argsort(pmax)
#sort the unlabeled corpus on the confidency
x_unlabeled = x_unlabeled[pidx]
y_unlabeled = y_unlabeled[pidx]
plt.plot(pmax[pidx])
```

<img src="https://miro.medium.com/max/750/1*z_fjP6IfjqJJy56CBb5fBQ.png" />

이제 Active learning을 사용하여 동일한 10%의 데이터 (400 개 샘플)에 레이블을 지정합니다. 이를 위해, 우리는 10개의 샘플 중 하나의 배치를 취하여 매우 원시적인 모델(primitive model)을 훈련시킵니다. 그런 다음이 모델을 통해 나머지 데이터 (3990 샘플)를 전달하고 최대 softmax 출력을 평가합니다. 선택한 클래스가 정답일 확률을 보여줍니다 (즉, 신경망의 신뢰도: the confidence of neural network). 정렬 후, 신뢰 분포가 20 %에서 100 %로 다양하다는 것을 플롯에서 볼 수 있습니다. LESS CONFIDENT 샘플에서 라벨을 지정할 다음 배치를 선택하는 것이 좋습니다.



```python
# do the same in a loop for 400 samples
for i  in range(39):
    label_manually(10)
    fit()
    
    res = sess.run(y_sm, feed_dict={x:x_unlabeled})
    pmax = np.amax(res, axis=1)
    pidx = np.argsort(pmax)
    x_unlabeled = x_unlabeled[pidx]
    y_unlabeled = y_unlabeled[pidx]
    
'''
Labels: 20 	Accuracy: 0.4975
Labels: 30 	Accuracy: 0.535
Labels: 40 	Accuracy: 0.5475
Labels: 50 	Accuracy: 0.59
Labels: 60 	Accuracy: 0.64
Labels: 70 	Accuracy: 0.6475
Labels: 80 	Accuracy: 0.6925
Labels: 90 	Accuracy: 0.6975

...

Labels: 350 	Accuracy: 0.865
Labels: 360 	Accuracy: 0.8775
Labels: 370 	Accuracy: 0.8825
Labels: 380 	Accuracy: 0.8825
Labels: 390 	Accuracy: 0.885
Labels: 400 	Accuracy: 0.8975
'''
```

10개 샘플의 40개 배치에 대해 이러한 절차를 실행하면 결과 정확도가 거의 90%임을 알 수 있습니다. 이는 무작위로 레이블이 지정된 데이터의 경우 달성 한 83.75 %보다 훨씬 큽니다.



## 나머지 라벨링 되어있지 않은 데이터로 할 일

> 원문: What to do with the rest of the unlabeled data

```python
# pass rest unlabeled data through the model and try to autolabel
res = sess.run(y_sm, feed_dict={x:x_unlabeled})
y_autolabeled = res.argmax(axis=1)
x_labeled = np.concatenate([x_labeled, x_unlabeled])
y_labeled = np.concatenate([y_labeled, y_autolabeled])
# train on 400 labeled by active learning and 3600 stochasticly autolabeled data
fit()

# 출력 값
# Labels: 4000	Accuracy: 0.8975
```

고전적인 방법은 기존 모델을 통해 나머지 데이터 세트를 실행하고 데이터에 자동으로 레이블을 지정하는 것입니다. 그런 다음 해당 데이터 세트를 훈련 과정에 넣는 것은 모델을 더 잘 조정할 수 있습니다. 위의 경우에는 더 나은 결과를 얻지 못했습니다.

My approach is to do the same but, as in the active learning, taking in consideration the confidence (이 부분을 이해하지 못했는데,, 아시는 분은 댓글 남겨주시면 감사할 것 같아요 ㅠㅠ)

```python
# pass rest of unlabeled (3600) data trough the model for automatic labeling and show most confident samples
res = sess.run(y_sm, feed_dict={x:x_unlabeled})
y_autolabeled = res.argmax(axis=1)
pmax = np.amax(res, axis=1)
pidx = np.argsort(pmax)
# sort by confidency
x_unlabeled = x_unlabeled[pidx]
y_autolabeled = y_autolabeled[pidx]
plt.plot(pmax[pidx])
```

<img src="https://miro.medium.com/max/750/1*37XHL8q9kDv2PZKrmLXvpA.png"/>

```python
# automatically label 10 most confident sample and train for it
x_labeled = np.concatenate([x_labeled, x_unlabeled[-10:]])
y_labeled = np.concatenate([y_labeled, y_autolabeled[-10:]])
x_unlabeled = x_unlabeled[:-10]
fit()
```

여기서 우리는 모델 평가를 통해 레이블이없는 나머지 데이터를 실행하고 나머지 샘플에 대한 신뢰도가 여전히 다르다는 것을 알 수 있습니다. 따라서 10개의 MOST CONFIDENT 샘플을 일괄적으로 뽑아서 모델을 교육하는 것이 좋을것 같다.



```python
# run rest of unlabelled samples starting from most confident
for i in range(359):
    res = sess.run(y_sm, feed_dict={x:x_unlabeled})
    y_autolabeled = res.argmax(axis=1)
    pmax = np.amax(res, axis=1)
    pidx = np.argsort(pmax)
    x_unlabeled = x_unlabeled[pidx]
    y_autolabeled = y_autolabeled[pidx]
    x_labeled = np.concatenate([x_labeled, x_unlabeled[-10:]])
    y_labeled = np.concatenate([y_labeled, y_autolabeled[-10:]])
    x_unlabeled = x_unlabeled[:-10]
    fit()
    
# 출력 값
'''
Labels: 420 	Accuracy: 0.8975
Labels: 430 	Accuracy: 0.8975
Labels: 440 	Accuracy: 0.8975
Labels: 450 	Accuracy: 0.8975
Labels: 460 	Accuracy: 0.8975
Labels: 470 	Accuracy: 0.8975
Labels: 480 	Accuracy: 0.8975
Labels: 490 	Accuracy: 0.8975
Labels: 500 	Accuracy: 0.8975
Labels: 510 	Accuracy: 0.8975
Labels: 520 	Accuracy: 0.8975

...

Labels: 3850 	Accuracy: 0.9
Labels: 3860 	Accuracy: 0.9
Labels: 3870 	Accuracy: 0.9
Labels: 3880 	Accuracy: 0.905
Labels: 3890 	Accuracy: 0.905
Labels: 3900 	Accuracy: 0.905
Labels: 3910 	Accuracy: 0.905
Labels: 3920 	Accuracy: 0.905
Labels: 3930 	Accuracy: 0.905
Labels: 3940 	Accuracy: 0.905
Labels: 3950 	Accuracy: 0.905
Labels: 3960 	Accuracy: 0.905
Labels: 3970 	Accuracy: 0.905
Labels: 3980 	Accuracy: 0.905
Labels: 3990 	Accuracy: 0.905
Labels: 4000 	Accuracy: 0.905
'''
```

이 과정을 통해  0.8%의 정확도 향상을 얻었습니다.



## 결과

Experiment Accuracy
4000 샘플 92.25%
400 랜덤 샘플 83.75%
400 active learned 샘플 89.75%
\+ auto-labeling 90.50%



## 결론

물론 이 접근방식은 연산 자원의 과도한 사용과 초기 모델 평가와 혼합된 데이터 라벨에 특별한 절차가 필요하다는 사실과 같은 단점을 가지고 있다. 또한 테스트를 위한 데이터에도 라벨을 붙여야 한다. 그러나 라벨의 비용이 높은 경우(특히 NLP, CV 프로젝트의 경우) 이 방법은 상당한 양의 자원을 절약하고 더 나은 프로젝트 결과를 가져올 수 있다.'



> 해당 포스트는 타 블로그 포스트를 번역한 내용입니다. https://towardsdatascience.com/active-learning-on-mnist-saving-on-labeling-f3971994c7ba

