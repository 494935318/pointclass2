# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import provider
import tensorflow as tf 
import tf_ops.grouping.tf_grouping as grouping
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow.keras.layers as layers 
import time
# import dataset
import glob
import pointnet_util as utils


# %%
physical_devices=tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0],True)
TRAIN_FILES=glob.glob("modelnet40_ply_hdf5_2048/*train*.h5")
TEST_FILES=glob.glob("modelnet40_ply_hdf5_2048/*test*.h5")


# %%
class self_attention(keras.Model):
    def __init__(self,out_dim,k_head):
       super(self_attention,self).__init__()
       self.out_dim=out_dim
       self.k=k_head
    def build(self,input_shape):
        self.convq=[layers.Conv2D(self.out_dim,1,1) for i in range(self.k)]
        self.convk=[layers.Conv2D(self.out_dim,1,1) for i in range(self.k)]
        self.convv=[layers.Conv2D(self.out_dim,1,1) for i in range(self.k)]
        self.Dense_out=layers.Dense(self.out_dim,activation="relu",kernel_regularizer=keras.regularizers.l2(0.001))
        self.norm=keras.layers.LayerNormalization()
        self.soft=keras.layers.Softmax()
        # self.drop=keras.layers.Dropout(0.2)
    def call(self,input_feature,training=True):
        out=[]
        for i in range(self.k):
            q=self.convq[i](input_feature)
            v=self.convv[i](input_feature)
            k=self.convk[i](input_feature)
            q=tf.reduce_mean(q,axis=-2,keepdims=True)
            # q=tf.tile(q,[1,1,tf.shape(v)[2],1])
            k=tf.transpose(k,[0,1,3,2])
            qk=tf.matmul(q,k)/tf.sqrt(tf.cast(self.out_dim,tf.float32))
            out.append(tf.matmul(self.soft(qk),v))
        out=tf.concat(out,axis=-1)
        out=self.norm(self.Dense_out(tf.squeeze(out,axis=-2)),training=training)
        # out=self.drop(out,training=training)
        return  out 


# %%
def random_sample(xyz):
    num=tf.shape(xyz)[1]
    return xyz[:,0:tf.cast(num/2,tf.int32),:]


# %%
class pointcloud_class(keras.Model):
    def __init__(self,class_num,laten_dim,group_num):
        super(pointcloud_class,self).__init__()
        self.class_num=class_num
        self.laten_dim=laten_dim
        self.group_num=group_num
    def build(self,inputshape):
        self.n=7
        self.Dense1=keras.layers.Dense(32,activation="relu") 
        self.Dense2=keras.layers.Dense(64,activation="relu") 
        self.norm=[layers.LayerNormalization() for i in range(4)]
        self.self_attention=[self_attention(self.laten_dim[i],3) for i in range(self.n+1)]
        self.self_attention2=[self_attention(self.laten_dim[i],3) for i in range(self.n)]
        self.Dense_=[layers.Dense(self.laten_dim[i],activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)) for i in range(self.n)]
        self.Dense_2=[layers.Dense(self.laten_dim[i],activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)) for i in range(self.n)]
        self.normal_1=[layers.LayerNormalization() for _ in range(self.n)]
        self.normal_2=[layers.LayerNormalization() for _ in range(self.n)]
        self.Dense3=layers.Dense(256,activation="relu")
        self.Dense4=layers.Dense(256,activation="relu")
        self.Dense5=layers.Dense(self.class_num)
        self.soft=keras.layers.Softmax()
        self.drop1=layers.Dropout(0.5)
        self.drop2=layers.Dropout(0.5)
    def call(self , points_xyz,training=True):
        # featrue=tf.concat([points_xyz,Density],axis=-1)
        featrue=self.Dense1(points_xyz)
        featrue=self.norm[0](featrue,training=training)
        featrue=self.Dense2(featrue)
        featrue=self.norm[1](featrue,training=training)
        old_xyz=points_xyz
        for i in range(self.n):
            _, grouped_feature, idx, _=utils.sample_and_group(old_xyz,0,self.group_num[i],old_xyz,featrue,knn=True)
            local_fature=grouped_feature-tf.reduce_mean(grouped_feature,axis=-2,keepdims=True)
            global_fature=tf.reduce_max(grouped_feature,axis=-2,keepdims=False)
            local_featrue=self.self_attention2[i](local_fature,training=training)
            featrue=self.Dense_2[i](tf.concat([local_featrue,global_fature],axis=-1))
            featrue=self.normal_1[i](featrue,training=training)


            new_xyz = random_sample(old_xyz)
            _, grouped_feature, idx, grouped_xyz=utils.sample_and_group(new_xyz,0,self.group_num[i],old_xyz,featrue,knn=True)
            # new_xyz=tf.reduce_mean(grouped_xyz,axis=-2)
            local_fature=grouped_feature-tf.reduce_mean(grouped_feature,axis=-2,keepdims=True)
            global_fature=tf.reduce_max(grouped_feature,axis=-2,keepdims=False)
            local_featrue=self.self_attention[i](local_fature,training=training)
            featrue=self.Dense_[i](tf.concat([local_featrue,global_fature],axis=-1))
            featrue=self.normal_2[i](featrue,training=training)
            old_xyz=new_xyz
        _,new_points,_,_=utils.sample_and_group_all(old_xyz,featrue)
        out=self.self_attention[self.n](new_points,training=training)
        out=tf.squeeze(out,axis=1)
        
        out=self.Dense3(out)
        out=self.norm[2](out,training=training)
        out=self.drop1(out,training=training)
        out=self.Dense4(out)
        out=self.norm[3](out,training=training)
        out=self.drop2(out,training=training)
        out=self.Dense5(out)
        out=self.soft(out)
        return out


# %%
lr_schedule=keras.optimizers.schedules.ExponentialDecay(0.001,100000,0.5,staircase=True)


# %%

optimizer=keras.optimizers.Adam(0.001)
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model=pointcloud_class(40,[64,64,128,128,256,256,512,512],[32,32,32,16,16,16,8])
mtric1=keras.metrics.SparseCategoricalAccuracy()
mtric2=keras.metrics.SparseCategoricalCrossentropy()


# %%
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)


# %%
LOG_FOUT = open('log_train3.txt', 'a+')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# %%
# def shuffle(data):
#     shuffed_data= np.zeros(data.shape, dtype=np.float32)
#     for k in range(data.shape[0]):
#         shuffed_data[k,...]=np.random.shuffle(data[k,...])
#     return shuffed_data

def callmodel(input,current_label,epoch):
    with tf.GradientTape() as tape:
            logits=model(input,training=True)
            if model.losses is not None and epoch>80:
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
    
    global BATCH
    global BATCH_SIZE
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
        callmodel(jittered_data,current_label,epoch)
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
BATCH_SIZE=16

# %%
ckpt=tf.train.Checkpoint(model=model,opti=optimizer,batch=tf.Variable(1))
ckpt_mana=tf.train.CheckpointManager(ckpt,"pointcloud_class2",max_to_keep=3)
epochs=200


# %%
data=[];label=[]

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
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    train_one_epoch(epoch)
    ckpt_mana.save()
    # print(lr_schedule)
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

#%%
print(model.losses)

# %%
ckpt.restore(tf.train.latest_checkpoint('pointcloud_class2'))
model=ckpt.model
optimizer=ckpt.opti
BATCH=ckpt.batch.numpy()


# %%



