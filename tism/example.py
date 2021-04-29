import tensorflow as tf
import model as tism
import architecture
import backbone

model = tism.get(architecture=architecture.UNet(input_shape=(256, 256, 1)))

model = tism.get(architecture=architecture.LinkNet(input_shape=(None, None, 1)))

model = tism.get(architecture=architecture.LinkNet(input_shape=(64, 64, 64, 1), op_dim=3),
                 backbone_encoder=backbone.VGG(),
                 backbone_decoder=backbone.VGG(initial_block_length=1))

model = tism.get(architecture=architecture.LinkNet(input_shape=(64, 64, 64, 1), op_dim=3),
                 backbone_encoder=backbone.ResBlock(backbone.VGG()),
                 backbone_decoder=backbone.VGG(initial_block_length=1))

model = tism.get(architecture=architecture.LinkNet(input_shape=(256, 256, 1), op_dim=2),
                 backbone_encoder=backbone.ResBlock(backbone.VGG()),
                 backbone_decoder=backbone.VGG(initial_block_length=1))

model = tism.get(architecture=architecture.LinkNet(input_shape=(None, None, 1), op_dim=2),
                 backbone_encoder=backbone.ResBlock(backbone.VGG()),
                 backbone_decoder=backbone.VGG(initial_block_length=1))

model.summary(line_length=120)

# create tensorboard visible logs
'''
@tf.function
def tf_trace(x):
    return model(x)

logdir = "logs"
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)
tf_trace(tf.zeros((1, 256, 256, 1)))
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
'''
