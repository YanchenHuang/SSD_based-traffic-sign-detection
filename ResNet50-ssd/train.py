import tensorflow as tf
import time

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, NUM_CLASSES, BATCH_SIZE, save_model_dir, \
    load_weights_before_training, load_weights_from_epoch, save_frequency, test_images_during_training, \
    test_images_dir_list
from core.ground_truth import ReadDataset, MakeGT
from core.loss import SSDLoss
from core.make_dataset import TFDataset
from core.ssd import SSD, ssd_prediction
from utils.visualize import visualize_training_results


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


if __name__ == '__main__':
    print("Train start\n")
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print("Generate the Data Set\n")
    dataset = TFDataset()
    train_data, train_count = dataset.generate_datatset()

    ssd = SSD()
    print_model_summary(network=ssd)

    if load_weights_before_training:
        ssd.load_weights(filepath=save_model_dir+"epoch-{}".format(load_weights_from_epoch))
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1

    # loss
    loss = SSDLoss()

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                                 decay_steps=20000,
                                                                 decay_rate=0.96)
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    # metrics
    loss_metric = tf.metrics.Mean()
    cls_loss_metric = tf.metrics.Mean()
    reg_loss_metric = tf.metrics.Mean()

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = ssd(batch_images, training=True)#调用call函数得到最初的各个层的feature map，每一个feature Map为[  1  19  19 150]
            print("pred",tf.shape(pred[1]),"\n")
            #print("prediction finished")
            output = ssd_prediction(feature_maps=pred, num_classes=NUM_CLASSES + 1)#将featureMap的预测值经行一定得排列，shape:[1,8732,25]
            #print("out",tf.shape(output))
            gt = MakeGT(batch_labels, pred)#计算ground Truth
            gt_boxes = gt.generate_gt_boxes()#产生gt_box
            loss_value, cls_loss, reg_loss = loss(y_true=gt_boxes, y_pred=output)
        gradients = tape.gradient(loss_value, ssd.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, ssd.trainable_variables))
        loss_metric.update_state(values=loss_value)
        cls_loss_metric.update_state(values=cls_loss)
        reg_loss_metric.update_state(values=reg_loss)

    print("train step start\n")
    for epoch in range(load_weights_from_epoch + 1, EPOCHS):
        start_time = time.time()
        for step, batch_data in enumerate(train_data):
            #print(tf.size(batch_data),"\n")
            images, labels = ReadDataset().read(batch_data)#images是一个[batch_size, 300,300,3],lable是[batch_size,20,5]的tensor
            #print(tf.shape(labels))
            train_step(batch_images=images, batch_labels=labels)
            time_per_step = (time.time() - start_time) / (step + 1)
            print("Epoch: {}/{}, step: {}/{}, {:.2f}s/step, loss: {:.5f}, "
                  "cls loss: {:.5f}, reg loss: {:.5f}".format(epoch,
                                                              EPOCHS,
                                                              step,
                                                              tf.math.ceil(train_count / BATCH_SIZE),
                                                              time_per_step,
                                                              loss_metric.result(),
                                                              cls_loss_metric.result(),
                                                              reg_loss_metric.result()))
        loss_metric.reset_states()
        cls_loss_metric.reset_states()
        reg_loss_metric.reset_states()

        if epoch % save_frequency == 0:
            ssd.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format="tf")

        if test_images_during_training:
            visualize_training_results(pictures=test_images_dir_list, model=ssd, epoch=epoch)

    ssd.save_weights(filepath=save_model_dir+"saved_model", save_format="tf")
