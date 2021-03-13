# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow.keras.layers as layers 
import time
import provider
# import dataset
import glob
# import utils

import pointnet_util as utils


# %%
# physical_devices=tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0],True)


# %%
TRAIN_FILES=glob.glob("modelnet40_ply_hdf5_2048/*train*.h5")
TEST_FILES=glob.glob("modelnet40_ply_hdf5_2048/*test*.h5")


# %%
class self_attention_all(keras.Model):
    def __init__(self,out_dim,k_head,**key):
       super(self_attention_all,self).__init__(**key)
       self.out_dim=out_dim
       self.k=k_head
    def build(self,input_shape):
        self.w= self.add_weight(
            shape=(input_shape[-1], self.out_dim*3),
            initializer="random_normal",
            trainable=True,name="W"
        )
        self.proj=layers.Dense(self.out_dim,activation="relu")
        self.C=self.out_dim
        self.soft=layers.Softmax()
    def call(self,input_feature,training=True):
        input_shape=tf.shape(input_feature)
        self.B=input_shape[0]
        self.N=input_shape[1]
        qkv=tf.matmul(input_feature,self.w)
        qkv=tf.reshape(qkv,[self.B,self.N,3,self.k,self.C//self.k])
        qkv=tf.transpose(qkv,[2,0,3,1,4])
        q,k,v=tf.split(qkv,3,axis=0)
        # q=tf.reduce_max(q,axis=-2,keepdims=True)
        attn=tf.matmul(q,tf.transpose(k,[0,1,2,4,3]))*(self.out_dim**-0.5)
        attn=self.soft(attn)
        x=tf.matmul(attn,v)
        x=tf.reshape(x,[self.B,self.N,self.out_dim])
        x=self.proj(x)
        return x 
        

        

c=np.random.rand(32,1024,128)
a=self_attention_all(256,4)
b=a(c)
print(a.summary())


# %%
class self_attention(keras.Model):
    def __init__(self,out_dim,k_head,**key):
       super(self_attention,self).__init__(**key)
       self.out_dim=out_dim
       self.k=k_head
    def build(self,input_shape):
        self.w= self.add_weight(
            shape=(input_shape[-1], self.out_dim*3),
            initializer="random_normal",
            trainable=True,name="W"
        )
        self.proj=layers.Dense(self.out_dim,activation="relu")
        self.C=self.out_dim
        self.soft=layers.Softmax()
    def call(self,input_feature,training=True):
        input_shape=tf.shape(input_feature)
        self.B=input_shape[0]
        self.N=input_shape[1]
        self.N2=input_shape[2]
        qkv=tf.matmul(input_feature,self.w)
        qkv=tf.reshape(qkv,[self.B,self.N,self.N2,3,self.k,self.C//self.k])
        qkv=tf.transpose(qkv,[3,0,1,4,2,5])
        q,k,v=tf.split(qkv,3,axis=0)
        q=tf.reduce_max(q,axis=-2,keepdims=True)
        attn=tf.matmul(q,tf.transpose(k,[0,1,2,3,5,4]))*(self.out_dim**-0.5)
        attn=self.soft(attn)
        x=tf.matmul(attn,v)
        x=tf.squeeze(x,axis=[0,-2])
        x=tf.reshape(x,[self.B,self.N,self.out_dim])
        x=self.proj(x)
        return x 
        

        





# %%
class Transform(keras.Model):
    def __init__(self,out_dim,k_head,ds,**key):
        super(Transform,self ).__init__(**key)
        self.out_dim=out_dim
        self.k=k_head
        self.ds=ds
    def build(self,input_shape):
        self.Dense=layers.Dense(self.out_dim,activation="relu")
        self.fc1=layers.Dense(2*self.out_dim,activation="relu")
        self.fc2=layers.Dense(self.out_dim)
        if self.ds:
            self.attn=self_attention(self.out_dim,self.k)
        else:
            self.attn=self_attention_all(self.out_dim,self.k)
        self.norm1=layers.LayerNormalization()
        self.norm2=layers.LayerNormalization()
    def call(self,xyz,newxyz,feature_input,group_num,training):
        if not self.ds:
            feature=self.attn(self.norm1(feature_input),training)
            feature_input=self.Dense(feature_input)
            feature=feature+feature_input
        if self.ds:
            _, group_feature, _,_=utils.sample_and_group(newxyz,0,group_num,xyz,feature_input,knn=True)
            feature=self.attn(self.norm1(group_feature),training)
            feature=feature+self.Dense(tf.reduce_max(group_feature,axis=-2))
        feature2=self.norm2(feature)
        feature2=self.fc1(feature2)
        feature2=self.fc2(feature2)
        feature2=feature2+feature
        return feature2

        
        


        
    


# %%
def random_sample(xyz,feature):
    num=tf.shape(xyz)[1]
    return xyz[:,0:tf.cast(num/2,tf.int32),:],feature[:,0:tf.cast(num/2,tf.int32),:]


# %%
class pointcloud_class(keras.Model):
    def __init__(self,class_num,laten_dim,group_num,**key):
        super(pointcloud_class,self).__init__(**key)
        self.class_num=class_num
        self.laten_dim=laten_dim
        self.group_num=group_num
    def build(self,inputshape):
        self.n=7
        self.tans_embed=[Transform(self.laten_dim[i],8,ds=False) for i in range(self.n)]
        self.tans_ds=[Transform(self.laten_dim[i],8,ds=True) for i in range(self.n)]
        self.Dense=[layers.Dense(64,activation="relu"),layers.Dense(128,activation="relu"),layers.Dense(256,activation="relu"),layers.Dense(128,activation="relu")]
        self.out=layers.Dense(self.class_num)
        self.soft=layers.Softmax()
        self.norm=[layers.LayerNormalization() for i in range(4)]
    def call(self, points_xyz,training=True):
        # Density=utils.kernel_density_estimation(points_xyz,1.0)
        # featrue=points_xyz
        featrue=self.Dense[0](points_xyz)
        featrue=self.norm[0](featrue,training=training)
        featrue=self.Dense[1](featrue)
        featrue=self.norm[1](featrue,training=training)
        newpoints=points_xyz
        for i in range(self.n):
            featrue=self.tans_embed[i](newpoints,newpoints,featrue,self.group_num[i],training)
            sample_points,_ = random_sample(newpoints,featrue)
            featrue=self.tans_ds[i](newpoints,sample_points,featrue,self.group_num[i],training)
            newpoints=sample_points
        out=self.Dense[2](featrue)
        out=self.norm[2](out,training=training)
        out=layers.Dropout(0.5)(out,training=training)
        out=self.Dense[3](out)
        out=self.norm[3](out,training=training)
        out=layers.Dropout(0.5)(out,training=training)
        out=self.out(out)
        out=tf.reduce_max(out,axis=-2)
        out=self.soft(out)
        return out


# %%
lr_schedule=keras.optimizers.schedules.ExponentialDecay(0.0001,100000,0.3,staircase=True)


# %%

optimizer=keras.optimizers.Adam(0.001)
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model=pointcloud_class(40,[64,64,128,128,128,256,256,256],[16,32,16,8,16,8,8])#1024，512，256，128，64，32，16，8，
mtric1=keras.metrics.SparseCategoricalAccuracy()
mtric2=keras.metrics.SparseCategoricalCrossentropy()


# %%
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)


# %%
LOG_FOUT = open('log_train2.txt', 'a+')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# %%
data=[];label=[]
BATCH_SIZE=32
for i in range(len(TRAIN_FILES)):
    data1,label1=provider.loadDataFile(TRAIN_FILES[i])
    data.append(data1[:,0:1024,:])
    label.append(label1)
data=np.concatenate(data,axis=0)
label=np.concatenate(label,axis=0)
datasets=tf.data.Dataset.from_tensor_slices((data,label))
datasets=datasets.shuffle(buffer_size=1024)
func=lambda x,y:(tf.random.shuffle(x) ,y)
datasets=datasets.map(func,num_parallel_calls=tf.data.experimental.AUTOTUNE)
datasets=datasets.batch(BATCH_SIZE)
# datasets.cache()

datasets=datasets.prefetch(1)

data=[];label=[]

for i in range(len(TEST_FILES)):
    data1,label1=provider.loadDataFile(TEST_FILES[i])
    data.append(data1[:,0:1024,:])
    label.append(label1)
data=np.concatenate(data,axis=0)
label=np.concatenate(label,axis=0)
test_datasets=tf.data.Dataset.from_tensor_slices((data,label))
test_datasets=test_datasets.shuffle(buffer_size=1024)
func=lambda x,y:(tf.random.shuffle(x) ,y)
test_datasets=test_datasets.map(func,num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_datasets=test_datasets.batch(BATCH_SIZE)
# datasets.cache()
test_datasets=test_datasets.prefetch(1)






# %%
# def shuffle(data):
#     shuffed_data= np.zeros(data.shape, dtype=np.float32)
#     for k in range(data.shape[0]):
#         shuffed_data[k,...]=np.random.shuffle(data[k,...])
#     return shuffed_data

def callmodel(input,current_label):
    with tf.GradientTape() as tape:
            logits=model(input,training=True)
            if model.losses :
                regularization_loss=tf.math.add_n(model.losses)
            else:
                regularization_loss=0
            loss_value = loss(current_label, logits)+regularization_loss
            # print(loss_value)
    grads=tape.gradient(loss_value,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    mtric1.update_state(current_label, logits)
    mtric2.update_state(current_label, logits)

def evaluate_model(input,current_label):
    logits=model(input,training=False)
    mtric1.update_state(current_label, logits)
    mtric2.update_state(current_label, logits)


# %%
def shuffledata(data):
    idx=np.arange(data.shape[1])
    for k in range(data.shape[0]):
        np.random.shuffle(idx)
        data[k,...]=data[k,idx,:]
    return data
        


# %%

def train_one_epoch(epoch):
    # train_file_idxs = np.arange(0, len(TRAIN_FILES))
    # np.random.shuffle(train_file_idxs)
    NUM_POINT=1024
    global BATCH_SIZE
    global BATCH
    for step,(current_data,current_label) in enumerate(datasets):
        # print(current_data.shape[0])
        BATCH=BATCH+1
        ckpt.batch.assign_add(1)
        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(current_data.numpy())
        jittered_data = provider.jitter_point_cloud(rotated_data)
        jittered_data= jittered_data.astype(np.float32)
        jittered_data=shuffledata(jittered_data)
        # jittered_data=shuffle(jittered_data)
        # print(jittered_data.dtype)
        # print(jittered_data.shape)
        tf.keras.backend.set_value(optimizer.lr, lr_schedule(BATCH*BATCH_SIZE))
        callmodel(jittered_data,current_label)
        # pred_val = np.argmax(logits, 1)
        # correct = np.sum(pred_val == current_label[start_idx:end_idx])
        # total_correct += correct
        # total_seen += BATCH_SIZE
        # # print(loss_value)
        # loss_sum += float(loss_value)
def test_one_epoch(epoch):
    for setp ,(data,label) in enumerate(test_datasets):
        evaluate_model(data,label)

        


# %%
BATCH=1
# print(model.summary())

# %%
ckpt=tf.train.Checkpoint(model=model,opti=optimizer,batch=tf.Variable(1))
ckpt_mana=tf.train.CheckpointManager(ckpt,"pointtransform",max_to_keep=3)
epochs=200


# %%
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    train_one_epoch(epoch)
    ckpt_mana.save()
    log_string("-----{}------".format(epoch))
    log_string('train_mean loss: %f' % float(mtric2.result())) 
    log_string('train_accuracy: %f' % float(mtric1.result()))
    mtric1.reset_states()
    mtric2.reset_states()
    test_one_epoch(epoch)
    log_string('test_mean loss: %f' % float(mtric2.result()))
    log_string('test_accuracy: %f' % float(mtric1.result()))
    mtric1.reset_states()
    mtric2.reset_states()


# %%
point,label=next(iter(datasets))
points=point[0]


# %%
get_ipython().run_line_magic('matplotlib', '')
n=1024
fig=plt.figure()
for i in range(1,8):
    ax = fig.add_subplot(2,4,i, projection='3d')
    ax.scatter(points[0:int(n/pow(2,i)),0],points[0:int(n/pow(2,i)),1],points[0:int(n/pow(2,i)),2])
    ax.set_title(names[label[0,0]])


# %%
x=open("F:\点云深度学习代码\Dataset\modelnet40_ply_hdf5_2048\shape_names.txt",'r')


# %%
names=x.read().split("\n")


# %%



# %%
ckpt.restore(tf.train.latest_checkpoint('pointtransform'))
model=ckpt.model
optimizer=ckpt.opti

BATCH=ckpt.batch.numpy()


# %%
print(optimizer.lr)


# %%



