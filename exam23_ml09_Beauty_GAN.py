#!/usr/bin/env python
# coding: utf-8


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat')


img = dlib.load_rgb_image('../imgs/02.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()


img_result = img.copy()
dets = detector(img)
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1, figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='w', facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()


fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections() # 얼굴을 정면으로 만들어줌
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()


faces = dlib.get_face_chips(img, objs, size=256, padding=0.3) #이미지 불러오고, 얼굴 정면으로
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


def align_faces(img):
    dets = detector(img, 1) #이미지 가져옴
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
        faces = dlib.get_face_chips(img, objs, size=256, padding=0.35) #padding 사진에 여백?을 줌
        return faces
test_img = dlib.load_rgb_image('../imgs/03.jpg')
test_faces = align_faces(test_img)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(26, 16))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)
plt.show()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('../models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('../models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


def preprocess(img):
    return (img / 255 - 0.5) * 2
def deprocess(img):
    return (img + 1) / 2


img1 = dlib.load_rgb_image('../imgs/no_makeup/xfsy_0405.png')
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('../imgs/makeup/002.jpg')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1, 2, figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()
